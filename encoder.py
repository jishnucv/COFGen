"""
encoder.py
==========
Graph Transformer VAE encoder for COF crystal graphs.

Architecture
------------
Input: COF crystal graph (atom features + fractional coords + periodic edges)

1. Atom embedding MLP: atom_feat → hidden_dim
2. Fractional coord Fourier embedding → appended to atom embedding
3. L graph transformer layers with:
   - SE(3)-inspired message passing (relative fractional coords under PBC)
   - Multi-head self-attention over atom neighbourhood
4. Building-block pooling:
   - Mean-pool atom embeddings within each BB → BB embeddings
   - Stack BB embeddings → sequence input for a small BB-level transformer
5. Crystal-level pooling:
   - Mean-pool BB embeddings → crystal embedding h_crys
   - Concatenate topology token embedding → h_crys_topo
6. VAE head: h_crys_topo → μ, log σ² → z ~ N(μ, σ²)

The decoder MLP (in decoder_net.py) maps z → (linkage logits, topology logits,
stacking logits) and optionally predicted properties.

Dimensions
----------
  atom_feat_dim  = 36   (from featurisation.py)
  hidden_dim     = 256
  latent_dim     = 256
  n_layers       = 6    (graph transformer layers)
  n_heads        = 8
  bb_layers      = 2    (BB-level transformer layers)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from utils.featurisation import (
    ATOM_FEAT_DIM, BOND_FEAT_DIM,
    N_LINKAGE_TYPES, N_TOPOLOGIES, N_STACKING,
)

LATENT_DIM  = 256
HIDDEN_DIM  = 256
N_HEADS     = 8
N_LAYERS    = 6   # graph transformer layers
BB_LAYERS   = 2   # BB-level transformer layers
FOURIER_K   = 16  # Fourier features per coord → 3 × 2K = 96 dims


# ─────────────────────────────────────────────────────────────────────────────
# Fourier positional embedding for fractional coordinates
# ─────────────────────────────────────────────────────────────────────────────

class FourierEmbedding(nn.Module):
    """
    Maps fractional coords (N, 3) → (N, 6K) using sin/cos Fourier features.
    Periodic by construction: f(x + 1) = f(x).
    """
    def __init__(self, K: int = FOURIER_K):
        super().__init__()
        self.K = K
        # frequencies 1, 2, ..., K  (registered as buffer, not parameter)
        freqs = torch.arange(1, K + 1, dtype=torch.float32) * 2 * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, frac: torch.Tensor) -> torch.Tensor:
        # frac: (N, 3)
        # out:  (N, 3 × 2K)
        f = frac.unsqueeze(-1) * self.freqs       # (N, 3, K)
        return torch.cat([f.sin(), f.cos()], dim=-1).reshape(frac.shape[0], -1)

    @property
    def out_dim(self) -> int:
        return 3 * 2 * self.K


# ─────────────────────────────────────────────────────────────────────────────
# Equivariant message passing layer (simplified PaiNN-style)
# ─────────────────────────────────────────────────────────────────────────────

class PeriodicMessageLayer(nn.Module):
    """
    One layer of message passing over the COF crystal graph.
    Messages include:
      - sender atom embedding
      - bond features
      - relative fractional displacement (Fourier-embedded, PBC-aware)

    Not fully SE(3)-equivariant (that requires tensor products / NequIP),
    but rotationally invariant via Fourier distance features.
    Extending to full equivariance (MACE backbone) is TODO for v2.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, bond_feat_dim: int = BOND_FEAT_DIM):
        super().__init__()
        self.fourier = FourierEmbedding(K=FOURIER_K)
        msg_in = hidden_dim + bond_feat_dim + self.fourier.out_dim
        self.msg_net = nn.Sequential(
            nn.Linear(msg_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,           # (N, hidden)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, bond_feat_dim)
        frac_coords: torch.Tensor, # (N, 3)
        edge_shift: torch.Tensor,  # (E, 3)  integer PBC shifts
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]

        # Relative fractional displacement with PBC
        diff_frac = frac_coords[dst] - frac_coords[src] + edge_shift   # (E, 3)
        # Wrap to [-0.5, 0.5]
        diff_frac = diff_frac - diff_frac.round()
        fourier_disp = self.fourier(diff_frac)   # (E, 6K)

        msg_in = torch.cat([h[src], edge_attr, fourier_disp], dim=-1)
        msg    = self.msg_net(msg_in)            # (E, hidden)

        # Aggregate messages (mean pooling per destination atom)
        agg = torch.zeros_like(h)
        count = torch.zeros(h.shape[0], 1, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(len(dst), 1, device=h.device))
        count = count.clamp(min=1)
        agg = agg / count

        combined = torch.cat([h, agg], dim=-1)  # (N, 2*hidden)
        gate     = self.update_gate(combined)
        h_new    = self.update_net(combined)
        h_out    = self.norm(h + gate * h_new)  # residual gated update
        return h_out


# ─────────────────────────────────────────────────────────────────────────────
# Building-block pooling
# ─────────────────────────────────────────────────────────────────────────────

def pool_by_building_block(
    h: torch.Tensor,        # (N, hidden)
    bb_index: torch.Tensor, # (N,)  BB id per atom
    n_bbs: int,
) -> torch.Tensor:          # (n_bbs, hidden)
    """Mean-pool atom embeddings within each building block."""
    device = h.device
    bb_h = torch.zeros(n_bbs, h.shape[-1], device=device)
    count = torch.zeros(n_bbs, 1, device=device)
    expanded = bb_index.unsqueeze(-1).expand_as(h)
    bb_h.scatter_add_(0, expanded, h)
    count.scatter_add_(0, bb_index.unsqueeze(-1),
                       torch.ones(len(bb_index), 1, device=device))
    count = count.clamp(min=1)
    return bb_h / count


# ─────────────────────────────────────────────────────────────────────────────
# Transformer block for BB-level reasoning
# ─────────────────────────────────────────────────────────────────────────────

class BBTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, n_heads: int = N_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, n_bbs, hidden)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full encoder (VAE)
# ─────────────────────────────────────────────────────────────────────────────

class COFEncoder(nn.Module):
    """
    Graph transformer VAE encoder.
    Input: one COF crystal graph (batched via collate_cof_graphs)
    Output: μ (B, latent_dim), log_var (B, latent_dim)
    """

    def __init__(
        self,
        atom_feat_dim:  int = ATOM_FEAT_DIM,
        bond_feat_dim:  int = BOND_FEAT_DIM,
        hidden_dim:     int = HIDDEN_DIM,
        latent_dim:     int = LATENT_DIM,
        n_layers:       int = N_LAYERS,
        n_heads:        int = N_HEADS,
        bb_layers:      int = BB_LAYERS,
        n_topologies:   int = N_TOPOLOGIES,
        topo_embed_dim: int = 32,
        max_bbs:        int = 32,          # max BBs per crystal (padding limit)
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.latent_dim    = latent_dim
        self.max_bbs       = max_bbs

        # ── Atom embedding ────────────────────────────────────────────────────
        fourier_dim = 3 * 2 * FOURIER_K   # 96
        self.atom_embed = nn.Sequential(
            nn.Linear(atom_feat_dim + fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fourier = FourierEmbedding(K=FOURIER_K)

        # ── Graph transformer layers ──────────────────────────────────────────
        self.gnn_layers = nn.ModuleList([
            PeriodicMessageLayer(hidden_dim, bond_feat_dim)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # ── BB-level transformer ──────────────────────────────────────────────
        self.bb_layers = nn.ModuleList([
            BBTransformerLayer(hidden_dim, n_heads)
            for _ in range(bb_layers)
        ])

        # ── Topology token embedding ──────────────────────────────────────────
        # Appended to the crystal embedding before the VAE head
        self.topo_embed = nn.Embedding(n_topologies, topo_embed_dim)

        # ── Lattice MLP ───────────────────────────────────────────────────────
        self.lattice_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )

        # ── VAE head ──────────────────────────────────────────────────────────
        crys_dim = hidden_dim + topo_embed_dim + 32   # crystal + topo + lattice
        self.mu_head      = nn.Linear(crys_dim, latent_dim)
        self.logvar_head  = nn.Linear(crys_dim, latent_dim)

    # ── Forward ──────────────────────────────────────────────────────────────

    def encode(
        self,
        atoms:        torch.Tensor,   # (N_total, atom_feat_dim)
        frac_coords:  torch.Tensor,   # (N_total, 3)
        lattice:      torch.Tensor,   # (B, 6)
        edge_index:   torch.Tensor,   # (2, E_total)
        edge_attr:    torch.Tensor,   # (E_total, bond_feat_dim)
        edge_shift:   torch.Tensor,   # (E_total, 3)
        bb_index:     torch.Tensor,   # (N_total,)   BB id within each graph
        batch:        torch.Tensor,   # (N_total,)   graph id per atom
        topology_idx: torch.Tensor,   # (B,)
        n_atoms_per_graph: torch.Tensor,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns mu (B, latent_dim), log_var (B, latent_dim).
        """
        B = lattice.shape[0]
        device = atoms.device

        # 1. Atom embedding with Fourier coords
        fourier_pos = self.fourier(frac_coords)       # (N, 6K)
        h = self.atom_embed(torch.cat([atoms, fourier_pos], dim=-1))  # (N, hidden)

        # 2. Graph transformer layers
        for layer in self.gnn_layers:
            h = self.dropout(layer(h, edge_index, edge_attr, frac_coords, edge_shift))

        # 3. Per-graph BB pooling
        # Compute global bb_id offsets so bb_index is unique per graph
        # (within a batch, bb_index resets to 0 for each graph — we add graph offset)
        bb_crystal_id = torch.zeros(h.shape[0], dtype=torch.long, device=device)
        atom_offset = 0
        bb_offset   = 0
        bb_per_graph = []
        for g in range(B):
            mask = (batch == g)
            n_atoms_g = int(mask.sum())
            bbs_g = bb_index[mask]
            n_bbs_g = int(bbs_g.max()) + 1 if n_atoms_g > 0 else 1
            bb_crystal_id[mask] = bbs_g + bb_offset
            bb_per_graph.append(n_bbs_g)
            bb_offset += n_bbs_g

        N_bbs_total = bb_offset
        bb_h = pool_by_building_block(h, bb_crystal_id, N_bbs_total)  # (N_bbs_total, hidden)

        # 4. BB-level transformer per graph (padded to max_bbs)
        max_bbs_in_batch = max(bb_per_graph)
        bb_seqs   = torch.zeros(B, max_bbs_in_batch, self.hidden_dim, device=device)
        key_masks = torch.ones(B, max_bbs_in_batch, dtype=torch.bool, device=device)

        bb_ptr = 0
        for g in range(B):
            n = bb_per_graph[g]
            bb_seqs[g, :n] = bb_h[bb_ptr:bb_ptr + n]
            key_masks[g, :n] = False   # False = "attend to this position"
            bb_ptr += n

        for layer in self.bb_layers:
            bb_seqs = layer(bb_seqs, key_padding_mask=key_masks)

        # 5. Crystal-level embedding: mean-pool valid BB positions
        # Use key_masks (True = padding) to zero out padded positions
        valid_mask = ~key_masks   # (B, max_bbs) True = valid
        bb_pool = (bb_seqs * valid_mask.unsqueeze(-1).float()).sum(dim=1)
        bb_pool = bb_pool / valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        # → (B, hidden)

        # 6. Append topology token + lattice
        topo_emb = self.topo_embed(topology_idx)       # (B, topo_embed_dim)
        lat_emb  = self.lattice_mlp(lattice)           # (B, 32)
        crystal_emb = torch.cat([bb_pool, topo_emb, lat_emb], dim=-1)  # (B, hidden+topo+32)

        # 7. VAE head
        mu      = self.mu_head(crystal_emb)
        log_var = self.logvar_head(crystal_emb)
        return mu, log_var

    def reparameterise(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(self, batch_dict: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (z, mu, log_var).
        batch_dict is the output of collate_cof_graphs.
        """
        mu, log_var = self.encode(
            atoms             = batch_dict["atoms"],
            frac_coords       = batch_dict["frac_coords"],
            lattice           = batch_dict["lattice"],
            edge_index        = batch_dict["edge_index"],
            edge_attr         = batch_dict["edge_attr"],
            edge_shift        = batch_dict["edge_shift"],
            bb_index          = batch_dict["bb_index"],
            batch             = batch_dict["batch"],
            topology_idx      = batch_dict["topology_idx"],
            n_atoms_per_graph = batch_dict["n_atoms_per_graph"],
        )
        z = self.reparameterise(mu, log_var)
        return z, mu, log_var


# ─────────────────────────────────────────────────────────────────────────────
# KL divergence loss
# ─────────────────────────────────────────────────────────────────────────────

def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Standard VAE KL: -0.5 * sum(1 + log_var - mu² - exp(log_var))"""
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Model summary (no torch needed for printing)
# ─────────────────────────────────────────────────────────────────────────────

def model_summary(model: nn.Module) -> str:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"COFEncoder\n"
        f"  Trainable parameters: {n_params:,}\n"
        f"  Hidden dim:  {model.hidden_dim}\n"
        f"  Latent dim:  {model.latent_dim}\n"
        f"  GNN layers:  {len(model.gnn_layers)}\n"
        f"  BB layers:   {len(model.bb_layers)}\n"
    )


if __name__ == "__main__" and HAS_TORCH:
    # Minimal smoke test with random batch
    import torch
    enc = COFEncoder()
    print(model_summary(enc))

    B, N, E = 2, 80, 320
    n_bbs_per = 3
    batch_dict = {
        "atoms":             torch.randn(N, ATOM_FEAT_DIM),
        "frac_coords":       torch.rand(N, 3),
        "lattice":           torch.randn(B, 6),
        "edge_index":        torch.randint(0, N, (2, E)),
        "edge_attr":         torch.randn(E, BOND_FEAT_DIM),
        "edge_shift":        torch.zeros(E, 3),
        "bb_index":          torch.randint(0, n_bbs_per, (N,)),
        "batch":             torch.cat([torch.zeros(N//2, dtype=torch.long),
                                        torch.ones(N//2, dtype=torch.long)]),
        "topology_idx":      torch.randint(0, N_TOPOLOGIES, (B,)),
        "n_atoms_per_graph": torch.tensor([N//2, N//2]),
    }

    z, mu, log_var = enc(batch_dict)
    print(f"z shape: {z.shape}  (expected {B} × {LATENT_DIM})")
    print(f"KL: {kl_divergence(mu, log_var).item():.4f}")
