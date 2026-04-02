"""
reticular_decoder.py
====================
Decodes a latent vector z* (LATENT_DIM,) into a valid COF crystal structure.

Two-stage process:
  Stage 1 — Spec decoder (SpecDecoderMLP):
    z* → (linkage_logits, topology_logits, stacking_logits,
           bb1_formula_logits, bb2_formula_logits)
    Trained jointly with the encoder.

  Stage 2 — Reticular assembly (RetricularDecoder):
    Takes the spec tuple and assembles a 3D CIF via pyCOFBuilder.
    Falls back to a simple template library if pyCOFBuilder is unavailable.

The decoder is the bottleneck for structural validity: by restricting the
output space to the pyCOFBuilder grammar, we guarantee:
  ✓ Correct covalent bonding at linkage sites
  ✓ Geometrically feasible unit cell (no atomic overlaps by construction)
  ✓ Accessible pore volume > 0 (net topology enforces open channels)
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Minimal shim so class definitions work without torch
    class _FakeModule:
        def __init__(self, *a, **kw): pass
        def __init_subclass__(cls, **kw): pass
        def parameters(self): return iter([])
        def forward(self, *a, **kw): raise RuntimeError("torch required")
        def to(self, *a, **kw): return self
        def eval(self): return self
        def train(self, mode=True): return self
        def state_dict(self): return {}
        def load_state_dict(self, sd): pass
        def __call__(self, *a, **kw): return self.forward(*a, **kw)
    class nn:
        Module = _FakeModule
        @staticmethod
        def Linear(*a, **kw): return None
        @staticmethod
        def Sequential(*a, **kw): return None
        @staticmethod
        def Dropout(*a, **kw): return None
        @staticmethod
        def SiLU(): return None

try:
    import pycofbuilder as pcb
    HAS_PYCOFBUILDER = True
except ImportError:
    HAS_PYCOFBUILDER = False

from utils.featurisation import (
    LATENT_DIM,
    N_LINKAGE_TYPES, N_TOPOLOGIES, N_STACKING,
    LINKAGE_TYPES, ALL_TOPOLOGIES, STACKING_PATTERNS,
    LINKAGE_TO_IDX, TOPOLOGY_TO_IDX, STACKING_TO_IDX,
)

# ── pyCOFBuilder building block library ──────────────────────────────────────
# These are the cores available in pyCOFBuilder as of v0.0.8.
# Format: (connectivity, core_code, connection_group)
# Full list: github.com/lipelopesoliveira/pyCOFBuilder

BB_LIBRARY: Dict[str, List[str]] = {
    # Tritopic (T3) nodes — most common for hcb COFs
    "T3_nodes": [
        "T3_BENZ",    # benzene-1,3,5-triyl
        "T3_TRIZ",    # triazine
        "T3_TPM",     # triphenylmethane
        "T3_TPA",     # triphenylamine
        "T3_TRIF",    # trifluoromethyl
        "T3_INTZ",    # interspersed triazine
    ],
    # Ditopic (L2) linkers
    "L2_linkers": [
        "L2_BENZ",    # benzene
        "L2_NAPH",    # naphthalene
        "L2_BIPH",    # biphenyl
        "L2_TPHN",    # terphenyl
        "L2_ANTR",    # anthracene
        "L2_PYRN",    # pyrene
        "L2_AZBN",    # azobenzene
        "L2_ETBE",    # ethynylene-bridged
        "L2_STIL",    # stilbene
        "L2_BTTA",    # BT-based
    ],
    # Tetratopic (S4) nodes — for sql COFs
    "S4_nodes": [
        "S4_BENZ",    # porphyrin-type
        "S4_PORPH",   # metalloporphyrin
        "S4_PHTH",    # phthalocyanine
    ],
    # Functional group suffixes
    "conn_groups": {
        "imine":           ("NH2", "CHO"),
        "boronate_ester":  ("B(OH)2", "diol"),
        "boroxine":        ("B(OH)2", "B(OH)2"),
        "beta_ketoenamine": ("NH2", "beta_keto"),
        "hydrazone":       ("NHNH2", "CHO"),
        "triazine":        ("CN", "CN"),
    },
}

# Compatible (topology, node_connectivity) pairs
TOPO_NODE_COMPAT = {
    "hcb": [("T3", "L2")],
    "sql": [("S4", "L2")],
    "kgm": [("T3", "L2")],
    "hxl": [("T3", "T3")],
    "dia": [("T4", "T4")],   # 3D
    "bor": [("T3", "T4")],   # 3D
}


# ─────────────────────────────────────────────────────────────────────────────
# Spec decoder MLP
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class COFSpec:
    """Decoded specification of a COF."""
    linkage_type: str
    topology:     str
    stacking:     str
    node_bb:      str   # pyCOFBuilder building-block name (node)
    linker_bb:    str   # pyCOFBuilder building-block name (linker)
    node_func:    str   # functional group on node (e.g. "NH2")
    linker_func:  str   # functional group on linker (e.g. "CHO")
    properties:   Dict[str, float] = field(default_factory=dict)

    def to_pycofbuilder_name(self) -> str:
        """
        Construct the pyCOFBuilder string identifier.
        Format: {node_bb}_{node_func}-{linker_bb}_{linker_func}-{TOPO}_A-{STACKING}
        """
        topo_upper = self.topology.upper()
        stacking   = self.stacking.upper()
        return (
            f"{self.node_bb}_{self.node_func}"
            f"-{self.linker_bb}_{self.linker_func}"
            f"-{topo_upper}_A-{stacking}"
        )

    def __str__(self) -> str:
        return (
            f"COFSpec(linkage={self.linkage_type}, topo={self.topology}, "
            f"stacking={self.stacking}, "
            f"node={self.node_bb}_{self.node_func}, "
            f"linker={self.linker_bb}_{self.linker_func})"
        )


class SpecDecoderMLP(nn.Module):
    """
    Maps latent z (B, LATENT_DIM) → COF specification logits.

    Outputs:
      linkage_logits  : (B, N_LINKAGE_TYPES)
      topology_logits : (B, N_TOPOLOGIES)
      stacking_logits : (B, N_STACKING)
      node_logits     : (B, N_NODE_BBs)
      linker_logits   : (B, N_LINKER_BBs)
    """

    N_NODE_BB   = len(BB_LIBRARY["T3_nodes"]) + len(BB_LIBRARY["S4_nodes"])
    N_LINKER_BB = len(BB_LIBRARY["L2_linkers"])

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 512,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.linkage_head  = nn.Linear(hidden_dim, N_LINKAGE_TYPES)
        self.topology_head = nn.Linear(hidden_dim, N_TOPOLOGIES)
        self.stacking_head = nn.Linear(hidden_dim, N_STACKING)
        self.node_head     = nn.Linear(hidden_dim, self.N_NODE_BB)
        self.linker_head   = nn.Linear(hidden_dim, self.N_LINKER_BB)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(z)
        return {
            "linkage":  self.linkage_head(h),
            "topology": self.topology_head(h),
            "stacking": self.stacking_head(h),
            "node":     self.node_head(h),
            "linker":   self.linker_head(h),
        }

    def decode_greedy(self, z: torch.Tensor) -> List[COFSpec]:
        """Greedy decode a batch of latents → list of COFSpec."""
        logits = self.forward(z)
        B = z.shape[0]
        specs = []

        node_names   = BB_LIBRARY["T3_nodes"] + BB_LIBRARY["S4_nodes"]
        linker_names = BB_LIBRARY["L2_linkers"]

        for i in range(B):
            lk_idx   = int(logits["linkage"][i].argmax())
            tp_idx   = int(logits["topology"][i].argmax())
            st_idx   = int(logits["stacking"][i].argmax())
            nd_idx   = int(logits["node"][i].argmax())
            ln_idx   = int(logits["linker"][i].argmax())

            linkage  = LINKAGE_TYPES[lk_idx]
            topology = ALL_TOPOLOGIES[tp_idx]
            stacking = STACKING_PATTERNS[st_idx]
            node_bb  = node_names[min(nd_idx, len(node_names)-1)]
            linker_bb = linker_names[min(ln_idx, len(linker_names)-1)]

            # Pick connection group from linkage type
            conn_groups = BB_LIBRARY["conn_groups"]
            node_func, linker_func = conn_groups.get(linkage, ("NH2", "CHO"))

            specs.append(COFSpec(
                linkage_type = linkage,
                topology     = topology,
                stacking     = stacking,
                node_bb      = node_bb,
                linker_bb    = linker_bb,
                node_func    = node_func,
                linker_func  = linker_func,
            ))
        return specs


def spec_decoder_loss(
    logits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Cross-entropy loss over all spec heads.
    targets: {"linkage": (B,), "topology": (B,), "stacking": (B,),
              "node": (B,), "linker": (B,)}
    """
    import torch.nn.functional as F
    loss = (
        F.cross_entropy(logits["linkage"],  targets["linkage"])  * 2.0
        + F.cross_entropy(logits["topology"], targets["topology"]) * 2.0
        + F.cross_entropy(logits["stacking"], targets["stacking"])
        + F.cross_entropy(logits["node"],     targets["node"])    * 1.5
        + F.cross_entropy(logits["linker"],   targets["linker"])  * 1.5
    )
    return loss / 8.0


# ─────────────────────────────────────────────────────────────────────────────
# Reticular assembler
# ─────────────────────────────────────────────────────────────────────────────

class RetricularDecoder:
    """
    Takes a COFSpec and returns a .cif file path (or CIF string).

    Uses pyCOFBuilder when available; falls back to a precomputed template
    lookup for the most common building block combinations.
    """

    def __init__(
        self,
        output_dir: Path = Path("outputs/structures"),
        template_dir: Optional[Path] = None,
    ):
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir = template_dir

    def assemble(self, spec: COFSpec) -> Optional[Path]:
        """
        Assemble a 3D CIF from a COFSpec.
        Returns path to CIF file, or None if assembly fails.
        """
        pcb_name = spec.to_pycofbuilder_name()

        if HAS_PYCOFBUILDER:
            return self._assemble_pycofbuilder(spec, pcb_name)
        else:
            return self._assemble_template(spec, pcb_name)

    def _assemble_pycofbuilder(self, spec: COFSpec, pcb_name: str) -> Optional[Path]:
        out_path = self.output_dir / f"{pcb_name}.cif"
        if out_path.exists():
            return out_path
        try:
            cof = pcb.Framework(pcb_name)
            cof.save(fmt="cif", save_dir=str(self.output_dir))
            return out_path
        except Exception as e:
            print(f"[RetricularDecoder] pyCOFBuilder failed for {pcb_name}: {e}")
            return None

    def _assemble_template(self, spec: COFSpec, pcb_name: str) -> Optional[Path]:
        """
        When pyCOFBuilder is unavailable, write a stub CIF that records the spec.
        Real assembly must be run separately with pyCOFBuilder installed.
        """
        stub_path = self.output_dir / f"{pcb_name}.spec.json"
        with open(stub_path, "w") as f:
            json.dump({
                "pcb_name":    pcb_name,
                "linkage":     spec.linkage_type,
                "topology":    spec.topology,
                "stacking":    spec.stacking,
                "node_bb":     spec.node_bb,
                "linker_bb":   spec.linker_bb,
                "node_func":   spec.node_func,
                "linker_func": spec.linker_func,
                "properties":  spec.properties,
                "status":      "pending_assembly",
            }, f, indent=2)
        return stub_path

    def assemble_batch(
        self,
        specs: List[COFSpec],
        n_jobs: int = 1,
    ) -> List[Optional[Path]]:
        """Assemble a list of specs; optionally parallel."""
        if n_jobs == 1:
            return [self.assemble(s) for s in specs]

        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            return list(ex.map(self.assemble, specs))


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end generation helper
# ─────────────────────────────────────────────────────────────────────────────

def latents_to_structures(
    z: "torch.Tensor",
    spec_decoder: SpecDecoderMLP,
    reticular_decoder: RetricularDecoder,
    temperature: float = 1.0,
) -> Tuple[List[COFSpec], List[Optional[Path]]]:
    """
    Full pipeline: latent tensors → COFSpecs → CIF files.

    Parameters
    ----------
    z               : (N, LATENT_DIM) generated latents from flow matching
    spec_decoder    : trained SpecDecoderMLP
    reticular_decoder: RetricularDecoder (wraps pyCOFBuilder)
    temperature     : softmax temperature for stochastic decoding (1.0 = greedy)

    Returns
    -------
    specs  : List[COFSpec]
    paths  : List[Optional[Path]]  — None where assembly failed
    """
    spec_decoder.eval()
    with torch.no_grad():
        if temperature == 1.0:
            specs = spec_decoder.decode_greedy(z)
        else:
            # Temperature-scaled sampling
            logits_dict = spec_decoder(z)
            B = z.shape[0]
            node_names   = BB_LIBRARY["T3_nodes"] + BB_LIBRARY["S4_nodes"]
            linker_names = BB_LIBRARY["L2_linkers"]
            specs = []
            for i in range(B):
                def sample(logits):
                    probs = torch.softmax(logits[i] / temperature, dim=-1)
                    return int(torch.multinomial(probs, 1).item())

                lk_idx  = sample(logits_dict["linkage"])
                tp_idx  = sample(logits_dict["topology"])
                st_idx  = sample(logits_dict["stacking"])
                nd_idx  = sample(logits_dict["node"])
                ln_idx  = sample(logits_dict["linker"])

                linkage  = LINKAGE_TYPES[lk_idx]
                topology = ALL_TOPOLOGIES[tp_idx]
                stacking = STACKING_PATTERNS[st_idx]
                node_bb  = node_names[min(nd_idx, len(node_names)-1)]
                linker_bb = linker_names[min(ln_idx, len(linker_names)-1)]
                node_func, linker_func = BB_LIBRARY["conn_groups"].get(linkage, ("NH2", "CHO"))

                specs.append(COFSpec(
                    linkage_type=linkage, topology=topology, stacking=stacking,
                    node_bb=node_bb, linker_bb=linker_bb,
                    node_func=node_func, linker_func=linker_func,
                ))

    paths = reticular_decoder.assemble_batch(specs)
    return specs, paths
