"""
flow_matching.py
================
Conditional Flow Matching (CFM) generative model over the COF latent space.

Why flow matching over DDPM:
  - Straight-line ODE trajectories → faster inference (fewer NFE)
  - Better latent space coverage than DDPM for structured discrete+continuous mix
  - Exact likelihood tractable via continuous normalising flows

Architecture
------------
  Noise z_0 ~ N(0, I)  →  vector field v_θ(z_t, t, c)  →  z_1 ≈ latent of COF

  The vector field network is a transformer over the flattened latent vector,
  conditioned on:
    - time t ∈ [0, 1]   (sinusoidal embedding)
    - property condition c  (via ControlNet-style adapters in adapters.py)

Training objective (conditional flow matching):
  L = E_{t, z_0, z_1} [ ||v_θ(z_t, t, c) - (z_1 - z_0)||² ]
  where z_t = (1-t)·z_0 + t·z_1  (simple linear interpolant)

Inference: solve ODE  dz/dt = v_θ(z_t, t, c)  from t=0 to t=1
  using Euler or RK4 with ~50 steps.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Minimal shim so class definitions work without torch
    class _FakeModule:
        def __init__(self, *a, **kw): pass
        def parameters(self): return iter([])
        def to(self, *a, **kw): return self
        def eval(self): return self
        def train(self, m=True): return self
        def state_dict(self): return {}
        def load_state_dict(self, sd): pass
        def __call__(self, *a, **kw): raise RuntimeError("torch required")
    class nn:
        Module = _FakeModule
        Linear = lambda *a, **kw: None
        Sequential = lambda *a, **kw: None
        Dropout = lambda *a, **kw: None
        Embedding = lambda *a, **kw: None
        LayerNorm = lambda *a, **kw: None
        MultiheadAttention = lambda *a, **kw: None
        ModuleList = lambda *a, **kw: []
        ModuleDict = lambda *a, **kw: {}
        Parameter = lambda *a, **kw: None

LATENT_DIM = 256
HIDDEN_DIM = 512   # wider than encoder — flow net is the expensive part
N_HEADS    = 8
N_LAYERS   = 8


# ─────────────────────────────────────────────────────────────────────────────
# Time and property embeddings
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal time embedding for diffusion / flow models."""
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)  values in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.proj(emb)


class PropertyEmbedding(nn.Module):
    """
    Embeds a dict of property conditions into a fixed-dim vector.
    Missing properties are masked out (set to 0 with a mask flag).
    """
    PROPERTY_NAMES = [
        "bet_surface_area",
        "pore_limiting_diameter",
        "void_fraction",
        "co2_uptake_298k_1bar",
        "band_gap",
        "linkage_type",   # integer → embedded
        "topology",       # integer → embedded
    ]

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        n_scalar = 5   # continuous properties
        self.scalar_proj  = nn.Linear(n_scalar * 2, hidden_dim // 2)
        # Discrete embeddings
        from utils.featurisation import N_LINKAGE_TYPES, N_TOPOLOGIES
        self.linkage_embed = nn.Embedding(N_LINKAGE_TYPES, 32)
        self.topo_embed    = nn.Embedding(N_TOPOLOGIES,    32)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32 + 32, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        props: Dict[str, torch.Tensor],   # property name → (B,) or None
        linkage_idx: Optional[torch.Tensor] = None,  # (B,)
        topology_idx: Optional[torch.Tensor] = None, # (B,)
    ) -> torch.Tensor:
        B = next(iter(props.values())).shape[0] if props else 1
        device = next(iter(props.values())).device if props else torch.device("cpu")

        scalar_keys = [
            "bet_surface_area", "pore_limiting_diameter",
            "void_fraction", "co2_uptake_298k_1bar", "band_gap",
        ]
        vals  = torch.zeros(B, len(scalar_keys), device=device)
        masks = torch.zeros(B, len(scalar_keys), device=device)
        for i, key in enumerate(scalar_keys):
            if key in props and props[key] is not None:
                v = props[key]
                vals[:, i]  = v
                masks[:, i] = 1.0

        scalar_emb = self.scalar_proj(torch.cat([vals, masks], dim=-1))  # (B, H/2)

        lk_emb = self.linkage_embed(
            linkage_idx if linkage_idx is not None
            else torch.zeros(B, dtype=torch.long, device=device)
        )  # (B, 32)
        tp_emb = self.topo_embed(
            topology_idx if topology_idx is not None
            else torch.zeros(B, dtype=torch.long, device=device)
        )  # (B, 32)

        cond = torch.cat([scalar_emb, lk_emb, tp_emb], dim=-1)
        return self.out_proj(cond)  # (B, hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Flow matching vector field network
# ─────────────────────────────────────────────────────────────────────────────

class FlowTransformerBlock(nn.Module):
    """Transformer block with adaptive layer norm conditioned on (time, property)."""
    def __init__(self, hidden_dim: int = HIDDEN_DIM, n_heads: int = N_HEADS):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        # Adaptive LayerNorm scale/shift from conditioning signal
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim * 4)  # → 2×scale + 2×shift

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, hidden)  cond: (B, hidden)
        params = self.cond_proj(cond).unsqueeze(1)  # (B, 1, 4H)
        s1, b1, s2, b2 = params.chunk(4, dim=-1)   # each (B, 1, H)

        h  = self.norm1(x)
        h  = h * (1 + s1) + b1
        attn_out, _ = self.attn(h, h, h)
        x  = x + attn_out

        h  = self.norm2(x)
        h  = h * (1 + s2) + b2
        x  = x + self.ff(h)
        return x


class FlowMatchingNetwork(nn.Module):
    """
    Vector field v_θ(z_t, t, c) for conditional flow matching.

    The latent z (LATENT_DIM) is treated as a short sequence of tokens
    (latent_dim // token_size tokens) so the transformer can attend across
    different parts of the latent.
    """
    TOKEN_SIZE = 16   # LATENT_DIM must be divisible by this

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_layers:   int = N_LAYERS,
        n_heads:    int = N_HEADS,
        time_dim:   int = 128,
    ):
        super().__init__()
        assert latent_dim % self.TOKEN_SIZE == 0
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_tokens   = latent_dim // self.TOKEN_SIZE

        # Input projection: latent tokens → hidden_dim
        self.in_proj = nn.Linear(self.TOKEN_SIZE, hidden_dim)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj  = nn.Linear(time_dim, hidden_dim)

        # Property conditioning embedding
        self.prop_embed = PropertyEmbedding(hidden_dim)

        # Positional embedding for latent tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_tokens, hidden_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FlowTransformerBlock(hidden_dim, n_heads)
            for _ in range(n_layers)
        ])

        # Output projection → back to latent token size
        self.out_proj = nn.Linear(hidden_dim, self.TOKEN_SIZE)

        # Initialise output to ~0 (flow starts near zero prediction)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,                    # (B, latent_dim)
        t:   torch.Tensor,                    # (B,)
        props: Optional[Dict[str, torch.Tensor]] = None,
        linkage_idx:  Optional[torch.Tensor] = None,
        topology_idx: Optional[torch.Tensor] = None,
        adapter_fn: Optional[Callable] = None,   # from adapters.py
    ) -> torch.Tensor:                            # (B, latent_dim)
        B = z_t.shape[0]

        # Tokenise latent
        x = z_t.view(B, self.n_tokens, self.TOKEN_SIZE)
        x = self.in_proj(x)                   # (B, n_tokens, hidden)
        x = x + self.pos_embed

        # Conditioning signal
        t_emb = self.time_proj(self.time_embed(t))       # (B, hidden)
        p_emb = self.prop_embed(
            props or {}, linkage_idx, topology_idx
        )                                                  # (B, hidden)
        cond  = t_emb + p_emb                            # (B, hidden)

        # Apply adapter residuals if provided
        if adapter_fn is not None:
            cond = cond + adapter_fn(cond)

        # Transformer
        for block in self.blocks:
            x = block(x, cond)

        # Output
        v = self.out_proj(x)                              # (B, n_tokens, TOKEN_SIZE)
        return v.view(B, self.latent_dim)                 # (B, latent_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Training objective
# ─────────────────────────────────────────────────────────────────────────────

def cfm_loss(
    model: FlowMatchingNetwork,
    z_1: torch.Tensor,              # (B, latent_dim)  target latent from encoder
    props: Optional[Dict] = None,
    linkage_idx: Optional[torch.Tensor] = None,
    topology_idx: Optional[torch.Tensor] = None,
    sigma_min: float = 1e-4,        # small noise floor on interpolant
) -> torch.Tensor:
    """
    Conditional flow matching loss (Lipman et al. 2023, Liu et al. 2022).
    Simple linear interpolant:  z_t = (1-t)·z_0 + t·z_1
    Target vector field:        u_t = z_1 - z_0
    """
    B = z_1.shape[0]
    device = z_1.device

    # Sample noise and time
    z_0 = torch.randn_like(z_1)
    t   = torch.rand(B, device=device)

    # Interpolate
    t_b = t.view(B, 1)
    z_t = (1 - t_b) * z_0 + t_b * z_1

    # Target (straight-line velocity)
    target = z_1 - z_0   # (B, latent_dim)

    # Predicted velocity
    v_pred = model(z_t, t, props, linkage_idx, topology_idx)

    return F.mse_loss(v_pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# ODE sampling
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_ode(
    model: FlowMatchingNetwork,
    n_samples: int,
    n_steps: int = 50,
    props: Optional[Dict] = None,
    linkage_idx: Optional[torch.Tensor] = None,
    topology_idx: Optional[torch.Tensor] = None,
    device: str = "cpu",
    solver: str = "euler",   # "euler" or "rk4"
) -> torch.Tensor:
    """
    Generate n_samples latent vectors by integrating the ODE from t=0 to t=1.
    Returns (n_samples, latent_dim).
    """
    model.eval()
    z = torch.randn(n_samples, model.latent_dim, device=device)
    dt = 1.0 / n_steps
    ts = torch.linspace(0, 1 - dt, n_steps, device=device)

    for t_val in ts:
        t_batch = t_val.expand(n_samples)

        if solver == "euler":
            v = model(z, t_batch, props, linkage_idx, topology_idx)
            z = z + dt * v

        elif solver == "rk4":
            k1 = model(z,               t_batch,           props, linkage_idx, topology_idx)
            k2 = model(z + dt/2 * k1,   t_batch + dt/2,    props, linkage_idx, topology_idx)
            k3 = model(z + dt/2 * k2,   t_batch + dt/2,    props, linkage_idx, topology_idx)
            k4 = model(z + dt   * k3,   t_batch + dt,      props, linkage_idx, topology_idx)
            z = z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        else:
            raise ValueError(f"Unknown solver: {solver}")

    return z


# ─────────────────────────────────────────────────────────────────────────────
# Classifier-free guidance
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_cfg(
    model: FlowMatchingNetwork,
    n_samples: int,
    n_steps: int = 50,
    props: Optional[Dict] = None,
    linkage_idx: Optional[torch.Tensor] = None,
    topology_idx: Optional[torch.Tensor] = None,
    guidance_scale: float = 2.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Classifier-free guidance sampling.
    v_guided = v_uncond + scale * (v_cond - v_uncond)
    Requires the model to have been trained with ~10% null conditioning
    (drop props to None randomly during training).
    """
    model.eval()
    z = torch.randn(n_samples, model.latent_dim, device=device)
    dt = 1.0 / n_steps

    for t_val in torch.linspace(0, 1 - dt, n_steps, device=device):
        t_batch = t_val.expand(n_samples)

        v_cond   = model(z, t_batch, props, linkage_idx, topology_idx)
        v_uncond = model(z, t_batch, None,  None,         None)
        v = v_uncond + guidance_scale * (v_cond - v_uncond)
        z = z + dt * v

    return z


if __name__ == "__main__" and HAS_TORCH:
    net = FlowMatchingNetwork()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"FlowMatchingNetwork: {n_params:,} parameters")

    B = 4
    z1 = torch.randn(B, LATENT_DIM)
    loss = cfm_loss(net, z1)
    print(f"CFM loss: {loss.item():.4f}")

    z_gen = sample_ode(net, n_samples=4, n_steps=10)
    print(f"Generated latents shape: {z_gen.shape}")
