"""
Atom and bond featurisation for COF crystal graphs.

Designed for COFs: heavy organic atoms (C, H, N, O, B, S, F, Cl, Br)
plus occasional metals in metallophthalocyanine or salen-type COFs.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── Atom vocabulary ──────────────────────────────────────────────────────────

# Elements present in COF literature; ordered by frequency in ReDD-COFFEE
ELEMENTS = [
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl",
    "Br", "I",
    # Metallophthalocyanines / salen COFs
    "Fe", "Co", "Ni", "Cu", "Zn", "Mn",
    # Catch-all
    "<UNK>",
]
ELEMENT_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(ELEMENTS)}
N_ELEMENTS = len(ELEMENTS)

# Hybridisation states
HYBRIDISATIONS = ["s", "sp", "sp2", "sp3", "sp3d", "sp3d2", "<UNK>"]
HYBRIDISATION_TO_IDX = {h: i for i, h in enumerate(HYBRIDISATIONS)}
N_HYBRID = len(HYBRIDISATIONS)

# ── Bond vocabulary ───────────────────────────────────────────────────────────

BOND_TYPES = [
    "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC",
    "LATTICE",     # periodic image edge (not a real bond)
    "<UNK>",
]
BOND_TO_IDX = {b: i for i, b in enumerate(BOND_TYPES)}
N_BOND_TYPES = len(BOND_TYPES)

# ── Linkage (connection group) vocabulary ────────────────────────────────────
# Each COF linkage type has a distinct chemistry and stability profile.

LINKAGE_TYPES = [
    "imine",             # -CH=N-  (Schiff base condensation)
    "boronate_ester",    # -B(OR)2  boronic acid + diol
    "boroxine",          # B3O3 ring  boronic acid + boronic acid
    "triazine",          # s-triazine  (CTF ionothermal)
    "hydrazone",         # -C(=O)-NH-N=
    "beta_ketoenamine",  # keto-enol tautomeric; very stable
    "imide",             # cyclic imide
    "urea",              # -NH-C(=O)-NH-
    "squaraine",         # donor-acceptor
    "olefin",            # C=C  Knoevenagel
    "amine",             # reductive amination
    "spiroborate",       # spiro B
    "<UNK>",
]
LINKAGE_TO_IDX = {l: i for i, l in enumerate(LINKAGE_TYPES)}
N_LINKAGE_TYPES = len(LINKAGE_TYPES)

# ── Topology vocabulary ───────────────────────────────────────────────────────

# 2D nets (most COFs)
TOPOLOGIES_2D = [
    "hcb",   # honeycomb — tritopic + ditopic, most common
    "sql",   # square lattice — tetratopic + ditopic
    "kgm",   # kagome
    "hxl",   # hexagonal — tritopic self-condensation
    "fxt",   # flux (augmented hxl)
    "kgd",   # kagome dual
    "bex",   # binodal expanded
    "tth",   # tritopic + hexatopic
    "htb",   # hexagonal tungsten bronze
    "pcu",   # primitive cubic (sometimes 2D analogue)
]

# 3D nets
TOPOLOGIES_3D = [
    "dia",   # diamond — tetratopic
    "bor",   # boracite — tri + tetra
    "ctn",   # carbon nitride
    "lon",   # lonsdaleite
    "srs",   # SrSi2
    "pts",   # PtS — tetratopic
    "rra",   # rare earth analogue
    "acs",   # aluminium chalcogenide
    "stp",   # snub trihexagonal tiling (3D)
    "ffc",   # face-fused cuboidal
    "qtz",   # quartz
]

ALL_TOPOLOGIES = TOPOLOGIES_2D + TOPOLOGIES_3D + ["<UNK>"]
TOPOLOGY_TO_IDX = {t: i for i, t in enumerate(ALL_TOPOLOGIES)}
N_TOPOLOGIES = len(ALL_TOPOLOGIES)

# Stacking patterns (2D COFs only)
STACKING_PATTERNS = ["AA", "AB", "ABC", "serrated", "inclined", "3D"]
STACKING_TO_IDX = {s: i for i, s in enumerate(STACKING_PATTERNS)}
N_STACKING = len(STACKING_PATTERNS)

# ── Model dimensions (centralised here so all modules import from one place) ──

LATENT_DIM = 256
HIDDEN_DIM_ENCODER = 256
HIDDEN_DIM_FLOW    = 512

# ── Feature dimensions ────────────────────────────────────────────────────────

ATOM_FEAT_DIM = (
    N_ELEMENTS      # one-hot element
    + N_HYBRID      # hybridisation
    + 1             # formal charge (normalised)
    + 1             # is in linkage site (bool)
    + 1             # is aromatic (bool)
    + 1             # degree (normalised)
)  # = 36 total

BOND_FEAT_DIM = (
    N_BOND_TYPES    # one-hot bond type
    + 1             # is in ring
    + 1             # is conjugated
    + 1             # distance (Å, normalised)
)  # = 9 total


def atom_features(
    element: str,
    hybridisation: str = "<UNK>",
    formal_charge: float = 0.0,
    is_linkage_site: bool = False,
    is_aromatic: bool = False,
    degree: int = 0,
) -> np.ndarray:
    """
    Returns a 1-D float32 array of atom features.
    All inputs are strings / scalars as parsed from the crystal graph builder.
    """
    feat = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)
    idx = 0

    # Element one-hot
    e_idx = ELEMENT_TO_IDX.get(element, ELEMENT_TO_IDX["<UNK>"])
    feat[idx + e_idx] = 1.0
    idx += N_ELEMENTS

    # Hybridisation one-hot
    h_idx = HYBRIDISATION_TO_IDX.get(hybridisation, HYBRIDISATION_TO_IDX["<UNK>"])
    feat[idx + h_idx] = 1.0
    idx += N_HYBRID

    # Scalar features — normalised to roughly [0, 1]
    feat[idx] = np.clip(formal_charge / 4.0, -1.0, 1.0)
    idx += 1
    feat[idx] = float(is_linkage_site)
    idx += 1
    feat[idx] = float(is_aromatic)
    idx += 1
    feat[idx] = np.clip(degree / 6.0, 0.0, 1.0)
    idx += 1

    assert idx == ATOM_FEAT_DIM
    return feat


def bond_features(
    bond_type: str = "SINGLE",
    is_in_ring: bool = False,
    is_conjugated: bool = False,
    distance_angstrom: float = 1.5,
) -> np.ndarray:
    """Returns a 1-D float32 array of bond features."""
    feat = np.zeros(BOND_FEAT_DIM, dtype=np.float32)
    idx = 0

    b_idx = BOND_TO_IDX.get(bond_type, BOND_TO_IDX["<UNK>"])
    feat[idx + b_idx] = 1.0
    idx += N_BOND_TYPES

    feat[idx] = float(is_in_ring)
    idx += 1
    feat[idx] = float(is_conjugated)
    idx += 1
    # Bond distance: typical COF bonds 1.2–2.0 Å, normalise by 3.0
    feat[idx] = np.clip(distance_angstrom / 3.0, 0.0, 1.0)
    idx += 1

    assert idx == BOND_FEAT_DIM
    return feat


def linkage_onehot(linkage: str) -> np.ndarray:
    """One-hot over LINKAGE_TYPES."""
    feat = np.zeros(N_LINKAGE_TYPES, dtype=np.float32)
    feat[LINKAGE_TO_IDX.get(linkage, LINKAGE_TO_IDX["<UNK>"])] = 1.0
    return feat


def topology_onehot(topology: str) -> np.ndarray:
    """One-hot over ALL_TOPOLOGIES."""
    feat = np.zeros(N_TOPOLOGIES, dtype=np.float32)
    feat[TOPOLOGY_TO_IDX.get(topology, TOPOLOGY_TO_IDX["<UNK>"])] = 1.0
    return feat


def stacking_onehot(stacking: str) -> np.ndarray:
    feat = np.zeros(N_STACKING, dtype=np.float32)
    feat[STACKING_TO_IDX.get(stacking, 0)] = 1.0
    return feat


# ── Lattice featurisation ─────────────────────────────────────────────────────

def lattice_features(
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
) -> np.ndarray:
    """
    6-dim lattice vector: [a, b, c, cos α, cos β, cos γ].
    Lengths normalised by 50 Å (large pore COFs can hit 40+ Å).
    Angles as cosines → [-1, 1].
    """
    return np.array([
        a / 50.0, b / 50.0, c / 50.0,
        np.cos(np.radians(alpha)),
        np.cos(np.radians(beta)),
        np.cos(np.radians(gamma)),
    ], dtype=np.float32)


# ── Property normalisation ───────────────────────────────────────────────────

# Approximate ranges from ReDD-COFFEE statistics.
PROPERTY_STATS: Dict[str, Tuple[float, float]] = {
    "bet_surface_area":    (0.0,    8000.0),   # m²/g
    "pore_limiting_diameter": (0.0, 30.0),     # Å
    "largest_cavity_diameter": (0.0, 50.0),    # Å
    "void_fraction":       (0.0,    1.0),       # dimensionless
    "co2_uptake_298k_1bar": (0.0,   5.0),      # mmol/g
    "ch4_uptake_298k_65bar": (0.0,  200.0),    # v/v
    "h2_uptake_77k_100bar": (0.0,   50.0),     # g/L
    "band_gap":            (0.0,    6.0),       # eV
    "formation_energy":    (-5.0,   2.0),       # eV/atom (UFF)
}


def normalise_property(name: str, value: float) -> float:
    lo, hi = PROPERTY_STATS.get(name, (0.0, 1.0))
    return float(np.clip((value - lo) / (hi - lo + 1e-8), 0.0, 1.0))


def denormalise_property(name: str, value: float) -> float:
    lo, hi = PROPERTY_STATS.get(name, (0.0, 1.0))
    return float(value * (hi - lo) + lo)
