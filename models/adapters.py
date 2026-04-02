"""
adapters.py
===========
ControlNet-style property adapters for COFGen flow matching model.

Each adapter is a small MLP that adds a conditioning residual to the
flow network's conditioning signal. Only the adapter parameters are
trained during fine-tuning; the base flow model is frozen.

Adapter types implemented:
  - ScalarPropertyAdapter  (BET surface area, CO2 uptake, band gap, etc.)
  - LinkageAdapter         (condition on linkage chemistry)
  - TopologyAdapter        (condition on desired net type)
  - MultiAdapter           (compose any subset of the above)

Usage
-----
    base_model  = FlowMatchingNetwork.load("checkpoints/base.pt")
    adapter     = ScalarPropertyAdapter("co2_uptake_298k_1bar", hidden_dim=512)

    # Fine-tune: freeze base_model, train adapter only
    optimizer = Adam(adapter.parameters(), lr=1e-4)
    for batch in loader:
        z1 = encoder(batch)
        loss = cfm_loss_with_adapter(base_model, adapter, z1, batch)
        loss.backward(); optimizer.step()

    # Inference with adapter
    z_gen = sample_ode(base_model, ..., adapter_fn=adapter)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
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

from utils.featurisation import N_LINKAGE_TYPES, N_TOPOLOGIES, normalise_property

HIDDEN_DIM = 512


# ─────────────────────────────────────────────────────────────────────────────
# Base adapter interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseAdapter(nn.Module):
    """
    An adapter maps a conditioning vector (B, hidden_dim) →
    a residual (B, hidden_dim) added to the flow network's cond.

    Subclasses define how they encode property values.
    """
    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "BaseAdapter":
        state = torch.load(path, map_location="cpu")
        adapter = cls(**state["config"])
        adapter.load_state_dict(state["weights"])
        return adapter

    def save(self, path: Path, config: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": config, "weights": self.state_dict()}, path)


# ─────────────────────────────────────────────────────────────────────────────
# Scalar property adapter
# ─────────────────────────────────────────────────────────────────────────────

class ScalarPropertyAdapter(BaseAdapter):
    """
    Conditions the flow on a single scalar property value (normalised to [0,1]).
    Includes a masking token for classifier-free guidance training:
    ~10% of training samples have the property masked (set to learned null embedding).
    """

    def __init__(
        self,
        property_name: str,
        hidden_dim:    int = HIDDEN_DIM,
        bottleneck:    int = 64,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.property_name = property_name
        self.hidden_dim    = hidden_dim

        # Property value encoder: (value, mask_flag) → bottleneck
        self.value_encoder = nn.Sequential(
            nn.Linear(2, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),
        )

        # Null embedding for classifier-free guidance
        self.null_embed = nn.Parameter(torch.zeros(bottleneck))

        # Residual MLP: (cond + bottleneck) → residual
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim + bottleneck, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Initialise residual output to zero (start = identity adapter)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def encode_value(
        self,
        value: Optional[torch.Tensor],  # (B,) normalised, or None
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            return self.null_embed.unsqueeze(0).expand(B, -1)

        val  = value.to(device)
        mask = (val >= 0).float()   # 0 = masked/missing
        x    = torch.stack([val, mask], dim=-1)   # (B, 2)
        return self.value_encoder(x)

    def forward(
        self,
        cond: torch.Tensor,                          # (B, hidden_dim)
        value: Optional[torch.Tensor] = None,        # (B,)  normalised property
        drop_prob: float = 0.0,                      # for CFG training
    ) -> torch.Tensor:
        B, device = cond.shape[0], cond.device

        # Randomly drop conditioning during training (CFG)
        if self.training and drop_prob > 0:
            mask = torch.rand(B, device=device) > drop_prob
            if not mask.any():
                value_emb = self.null_embed.unsqueeze(0).expand(B, -1)
            else:
                value_emb = self.encode_value(value, B, device)
                null_exp  = self.null_embed.unsqueeze(0).expand(B, -1)
                value_emb = torch.where(mask.unsqueeze(-1), value_emb, null_exp)
        else:
            value_emb = self.encode_value(value, B, device)

        combined = torch.cat([cond, value_emb], dim=-1)
        return self.residual(combined)


# ─────────────────────────────────────────────────────────────────────────────
# Discrete adapters
# ─────────────────────────────────────────────────────────────────────────────

class LinkageAdapter(BaseAdapter):
    """Conditions the flow on target linkage chemistry."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, embed_dim: int = 64):
        super().__init__()
        self.embed   = nn.Embedding(N_LINKAGE_TYPES + 1, embed_dim)  # +1 for null
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        cond: torch.Tensor,
        linkage_idx: Optional[torch.Tensor] = None,
        drop_prob: float = 0.0,
    ) -> torch.Tensor:
        B = cond.shape[0]
        if linkage_idx is None:
            idx = torch.full((B,), N_LINKAGE_TYPES, dtype=torch.long, device=cond.device)
        else:
            idx = linkage_idx.clone()
            if self.training and drop_prob > 0:
                drop_mask = torch.rand(B, device=cond.device) < drop_prob
                idx[drop_mask] = N_LINKAGE_TYPES  # null token

        emb      = self.embed(idx)
        combined = torch.cat([cond, emb], dim=-1)
        return self.residual(combined)


class TopologyAdapter(BaseAdapter):
    """Conditions the flow on desired COF net type."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, embed_dim: int = 64):
        super().__init__()
        self.embed   = nn.Embedding(N_TOPOLOGIES + 1, embed_dim)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        cond: torch.Tensor,
        topology_idx: Optional[torch.Tensor] = None,
        drop_prob: float = 0.0,
    ) -> torch.Tensor:
        B = cond.shape[0]
        if topology_idx is None:
            idx = torch.full((B,), N_TOPOLOGIES, dtype=torch.long, device=cond.device)
        else:
            idx = topology_idx.clone()
            if self.training and drop_prob > 0:
                drop_mask = torch.rand(B, device=cond.device) < drop_prob
                idx[drop_mask] = N_TOPOLOGIES

        emb      = self.embed(idx)
        combined = torch.cat([cond, emb], dim=-1)
        return self.residual(combined)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-adapter: compose any combination
# ─────────────────────────────────────────────────────────────────────────────

class MultiAdapter(nn.Module):
    """
    Composes multiple adapters; their residuals are summed.
    Allows e.g. conditioning simultaneously on CO2 uptake + linkage type.
    """

    def __init__(self, adapters: Dict[str, BaseAdapter]):
        super().__init__()
        self.adapters = nn.ModuleDict(adapters)

    def forward(
        self,
        cond:         torch.Tensor,
        values:       Optional[Dict[str, Optional[torch.Tensor]]] = None,
        linkage_idx:  Optional[torch.Tensor] = None,
        topology_idx: Optional[torch.Tensor] = None,
        drop_prob:    float = 0.0,
    ) -> torch.Tensor:
        residual = torch.zeros_like(cond)
        values = values or {}

        for name, adapter in self.adapters.items():
            if isinstance(adapter, ScalarPropertyAdapter):
                residual = residual + adapter(
                    cond, values.get(adapter.property_name), drop_prob
                )
            elif isinstance(adapter, LinkageAdapter):
                residual = residual + adapter(cond, linkage_idx, drop_prob)
            elif isinstance(adapter, TopologyAdapter):
                residual = residual + adapter(cond, topology_idx, drop_prob)

        return residual

    def make_adapter_fn(
        self,
        values: Optional[Dict[str, Optional[torch.Tensor]]] = None,
        linkage_idx:  Optional[torch.Tensor] = None,
        topology_idx: Optional[torch.Tensor] = None,
        drop_prob: float = 0.0,
    ) -> Callable:
        """
        Returns a closure compatible with FlowMatchingNetwork's adapter_fn argument.
        """
        def adapter_fn(cond: torch.Tensor) -> torch.Tensor:
            return self.forward(cond, values, linkage_idx, topology_idx, drop_prob)
        return adapter_fn


# ─────────────────────────────────────────────────────────────────────────────
# CFM loss with adapter
# ─────────────────────────────────────────────────────────────────────────────

def cfm_loss_with_adapter(
    base_model,         # FlowMatchingNetwork  (frozen)
    adapter: MultiAdapter,
    z_1: torch.Tensor,  # (B, latent_dim) from encoder
    batch_dict: dict,
    cfg_drop_prob: float = 0.1,
) -> torch.Tensor:
    """
    CFM loss when fine-tuning an adapter on top of a frozen base model.
    The adapter_fn is passed to base_model.forward().
    """
    import torch.nn.functional as F

    B = z_1.shape[0]
    device = z_1.device

    # Gather property values from batch
    values = {}
    for key in batch_dict:
        if key.startswith("prop_"):
            prop_name = key[5:]
            values[prop_name] = batch_dict[key].to(device)

    linkage_idx  = batch_dict.get("linkage_idx",  None)
    topology_idx = batch_dict.get("topology_idx", None)
    if linkage_idx  is not None: linkage_idx  = linkage_idx.to(device)
    if topology_idx is not None: topology_idx = topology_idx.to(device)

    adapter_fn = adapter.make_adapter_fn(values, linkage_idx, topology_idx, cfg_drop_prob)

    # Standard CFM interpolant
    z_0 = torch.randn_like(z_1)
    t   = torch.rand(B, device=device)
    t_b = t.view(B, 1)
    z_t = (1 - t_b) * z_0 + t_b * z_1
    target = z_1 - z_0

    v_pred = base_model(z_t, t, adapter_fn=adapter_fn)
    return F.mse_loss(v_pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_adapter(config: dict) -> MultiAdapter:
    """
    Build a MultiAdapter from a config dict, e.g.:
    {
        "scalar_properties": ["co2_uptake_298k_1bar", "bet_surface_area"],
        "linkage": true,
        "topology": true,
        "hidden_dim": 512,
    }
    """
    hidden_dim = config.get("hidden_dim", HIDDEN_DIM)
    adapters   = {}

    for prop in config.get("scalar_properties", []):
        adapters[f"scalar_{prop}"] = ScalarPropertyAdapter(prop, hidden_dim)

    if config.get("linkage", False):
        adapters["linkage"] = LinkageAdapter(hidden_dim)

    if config.get("topology", False):
        adapters["topology"] = TopologyAdapter(hidden_dim)

    return MultiAdapter(adapters)
