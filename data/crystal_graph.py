"""
crystal_graph.py
================
Converts a COF crystal structure (CIF or POSCAR) into a COF-native crystal graph.

The key innovation vs. a generic crystal graph:
  - Building-block (BB) subgraphs are identified via connectivity flood-fill
    after stripping linkage bonds
  - Each BB gets a canonical SMILES + connectivity integer
  - The graph carries both atom-level edges (for the encoder GNN) and
    BB-level edges (for the coarse-grained topology token)

Works without pymatgen if only numpy/networkx are available (uses a lightweight
fallback CIF parser). Full pymatgen integration enabled automatically.

Output schema
-------------
CrystalGraph (dataclass):
    atoms           : np.ndarray  shape (N, ATOM_FEAT_DIM)
    frac_coords     : np.ndarray  shape (N, 3)
    lattice         : np.ndarray  shape (6,)          [a,b,c,cosα,cosβ,cosγ]
    edge_index      : np.ndarray  shape (2, E)         atom-level graph (undirected→both dirs)
    edge_attr       : np.ndarray  shape (E, BOND_FEAT_DIM)
    edge_shift      : np.ndarray  shape (E, 3)         periodic image shifts
    bb_index        : np.ndarray  shape (N,)           building-block id per atom
    bb_smiles       : List[str]                        canonical SMILES per BB (may be approx)
    linkage_type    : str
    topology        : str
    stacking        : str
    name            : str                              structure identifier
    properties      : Dict[str, float]                 optional property labels
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from utils.featurisation import (
    ATOM_FEAT_DIM, BOND_FEAT_DIM,
    ELEMENT_TO_IDX, HYBRIDISATION_TO_IDX, BOND_TO_IDX,
    atom_features, bond_features, lattice_features,
    LINKAGE_TO_IDX, TOPOLOGY_TO_IDX, STACKING_TO_IDX,
)

# ── try importing pymatgen (optional) ────────────────────────────────────────
try:
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import CrystalNN
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    from data.bb_decomposer import replace_bb_index_with_decomposed
    HAS_BB_DECOMP = True
except ImportError:
    HAS_BB_DECOMP = False


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrystalGraph:
    atoms:       np.ndarray          # (N, ATOM_FEAT_DIM)
    frac_coords: np.ndarray          # (N, 3)
    lattice:     np.ndarray          # (6,)
    edge_index:  np.ndarray          # (2, E)
    edge_attr:   np.ndarray          # (E, BOND_FEAT_DIM)
    edge_shift:  np.ndarray          # (E, 3)  integer lattice shifts
    bb_index:    np.ndarray          # (N,)  building-block ID per atom
    bb_smiles:   List[str]           # one SMILES per unique BB
    linkage_type: str = "<UNK>"
    topology:    str  = "<UNK>"
    stacking:    str  = "AA"
    name:        str  = ""
    properties:  Dict[str, float] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def n_building_blocks(self) -> int:
        return int(self.bb_index.max()) + 1 if len(self.bb_index) else 0

    def to_dict(self) -> dict:
        return {
            "atoms":        self.atoms.tolist(),
            "frac_coords":  self.frac_coords.tolist(),
            "lattice":      self.lattice.tolist(),
            "edge_index":   self.edge_index.tolist(),
            "edge_attr":    self.edge_attr.tolist(),
            "edge_shift":   self.edge_shift.tolist(),
            "bb_index":     self.bb_index.tolist(),
            "bb_smiles":    self.bb_smiles,
            "linkage_type": self.linkage_type,
            "topology":     self.topology,
            "stacking":     self.stacking,
            "name":         self.name,
            "properties":   self.properties,
        }

    @staticmethod
    def from_dict(d: dict) -> "CrystalGraph":
        return CrystalGraph(
            atoms        = np.array(d["atoms"],       dtype=np.float32),
            frac_coords  = np.array(d["frac_coords"], dtype=np.float32),
            lattice      = np.array(d["lattice"],     dtype=np.float32),
            edge_index   = np.array(d["edge_index"],  dtype=np.int64),
            edge_attr    = np.array(d["edge_attr"],   dtype=np.float32),
            edge_shift   = np.array(d["edge_shift"],  dtype=np.float32),
            bb_index     = np.array(d["bb_index"],    dtype=np.int64),
            bb_smiles    = d["bb_smiles"],
            linkage_type = d.get("linkage_type", "<UNK>"),
            topology     = d.get("topology",     "<UNK>"),
            stacking     = d.get("stacking",     "AA"),
            name         = d.get("name",         ""),
            properties   = d.get("properties",   {}),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load(path: Path) -> "CrystalGraph":
        with open(path) as f:
            return CrystalGraph.from_dict(json.load(f))


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight CIF parser (no pymatgen)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cif_minimal(cif_text: str) -> dict:
    """
    Extract atom sites and cell parameters from a CIF without pymatgen.
    Returns a dict with keys: elements, frac_x, frac_y, frac_z, a, b, c, alpha, beta, gamma.
    """
    lines = cif_text.splitlines()

    def grab(tag: str) -> Optional[str]:
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith(tag.lower()):
                parts = stripped.split(None, 1)
                if len(parts) == 2:
                    return parts[1].strip().strip("'\"")
        return None

    def to_float(s: Optional[str]) -> float:
        if s is None:
            return 90.0
        return float(re.sub(r"\(.*?\)", "", s))

    a     = to_float(grab("_cell_length_a"))
    b     = to_float(grab("_cell_length_b"))
    c     = to_float(grab("_cell_length_c"))
    alpha = to_float(grab("_cell_angle_alpha"))
    beta  = to_float(grab("_cell_angle_beta"))
    gamma = to_float(grab("_cell_angle_gamma"))

    # Find atom_site loop
    in_loop = False
    headers: List[str] = []
    rows: List[List[str]] = []
    collecting_headers = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower() == "loop_":
            if headers and any("_atom_site" in h for h in headers) and rows:
                break
            in_loop = True
            collecting_headers = True
            headers = []
            rows = []
            continue
        if in_loop and collecting_headers and stripped.lower().startswith("_atom_site"):
            headers.append(stripped.lower())
            continue
        if in_loop and headers and not stripped.startswith("_") and stripped.lower() != "loop_":
            collecting_headers = False
            if stripped:
                rows.append(stripped.split())

    def col(name_fragment: str) -> Optional[int]:
        for i, h in enumerate(headers):
            if name_fragment in h:
                return i
        return None

    type_col  = col("type_symbol")
    if type_col is None:
        type_col = col("label")
    x_col     = col("fract_x")
    y_col     = col("fract_y")
    z_col     = col("fract_z")

    if type_col is None or x_col is None:
        return {}

    elements, fx, fy, fz = [], [], [], []
    for row in rows:
        if len(row) <= max(type_col, x_col, y_col or 0, z_col or 0):
            continue
        raw_el = row[type_col]
        # Strip oxidation state etc.
        el = re.sub(r"[^A-Za-z]", "", raw_el)
        el = el.capitalize()
        elements.append(el)
        try:
            fx.append(float(re.sub(r"\(.*?\)", "", row[x_col])))
            fy.append(float(re.sub(r"\(.*?\)", "", row[y_col or x_col + 1])))
            fz.append(float(re.sub(r"\(.*?\)", "", row[z_col or x_col + 2])))
        except (ValueError, IndexError):
            elements.pop()

    return dict(elements=elements, frac_x=fx, frac_y=fy, frac_z=fz,
                a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)


# ─────────────────────────────────────────────────────────────────────────────
# Distance computation under periodic boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

def _cell_matrix(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Returns the 3×3 cell matrix (row vectors = lattice vectors)."""
    alpha_r = math.radians(alpha)
    beta_r  = math.radians(beta)
    gamma_r = math.radians(gamma)
    cos_a, cos_b, cos_g = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
    sin_g = math.sin(gamma_r)
    vol_factor = math.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2
                           + 2 * cos_a * cos_b * cos_g)
    M = np.array([
        [a,             0,           0           ],
        [b * cos_g,     b * sin_g,   0           ],
        [c * cos_b,  c * (cos_a - cos_b * cos_g) / sin_g,
                     c * vol_factor / sin_g       ],
    ], dtype=np.float64)
    return M


def _pbc_edges(
    frac_coords: np.ndarray,    # (N, 3)
    cell: np.ndarray,           # (3, 3)
    cutoff: float = 5.0,        # Å — generous for large COF unit cells
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a neighbour list under PBC.
    Returns (edge_index (2, E), shifts (E, 3), distances (E,)).
    Fully vectorised — ~100× faster than the nested loop version.
    """
    N = len(frac_coords)
    if N == 0:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.float32))

    n_images = int(math.ceil(cutoff / max(np.linalg.norm(cell, axis=1).min(), 0.1))) + 1

    # All image shifts to check: (K, 3)
    rng_1d = range(-n_images, n_images + 1)
    shifts_all = np.array([[nx, ny, nz]
                            for nx in rng_1d
                            for ny in rng_1d
                            for nz in rng_1d], dtype=np.float32)  # (K, 3)

    # All pairwise fractional differences: (N, N, 3)
    diff_all = frac_coords[np.newaxis, :, :] - frac_coords[:, np.newaxis, :]

    src_list, dst_list, shift_list, dist_list = [], [], [], []

    for shift in shifts_all:
        # (N, N, 3) displaced fractional → Cartesian
        displaced = (diff_all + shift) @ cell          # (N, N, 3)
        d_mat = np.linalg.norm(displaced, axis=-1)     # (N, N)

        # Valid: distance ≤ cutoff and not (self + zero shift)
        is_self_zero = np.zeros((N, N), dtype=bool)
        if np.all(shift == 0):
            np.fill_diagonal(is_self_zero, True)

        valid = (d_mat <= cutoff) & (d_mat > 1e-8) & ~is_self_zero
        ii, jj = np.where(valid)

        if len(ii):
            src_list.append(ii)
            dst_list.append(jj)
            shift_list.append(np.tile(shift, (len(ii), 1)))
            dist_list.append(d_mat[ii, jj])

    if not src_list:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.float32))

    src    = np.concatenate(src_list).astype(np.int64)
    dst    = np.concatenate(dst_list).astype(np.int64)
    shifts = np.concatenate(shift_list).astype(np.float32)
    dists  = np.concatenate(dist_list).astype(np.float32)

    return np.array([src, dst], dtype=np.int64), shifts, dists


# ─────────────────────────────────────────────────────────────────────────────
# Building-block identification via flood-fill
# ─────────────────────────────────────────────────────────────────────────────

# Bond-length cutoffs (Å) for recognising covalent bonds in each element pair.
# Keys are frozensets of two element symbols.
COVALENT_CUTOFFS: Dict[frozenset, float] = {
    frozenset({"C","C"}):  1.6,
    frozenset({"C","H"}):  1.2,
    frozenset({"C","N"}):  1.5,
    frozenset({"C","O"}):  1.5,
    frozenset({"C","B"}):  1.7,
    frozenset({"C","S"}):  1.9,
    frozenset({"C","F"}):  1.5,
    frozenset({"C","Cl"}): 1.9,
    frozenset({"C","Br"}): 2.0,
    frozenset({"N","H"}):  1.1,
    frozenset({"N","N"}):  1.5,
    frozenset({"N","B"}):  1.7,
    frozenset({"O","H"}):  1.1,
    frozenset({"O","B"}):  1.6,
    frozenset({"B","B"}):  1.8,
}
DEFAULT_COVALENT_CUTOFF = 2.2   # fallback


def _covalent_cutoff(el1: str, el2: str) -> float:
    return COVALENT_CUTOFFS.get(frozenset({el1, el2}), DEFAULT_COVALENT_CUTOFF)


def _identify_building_blocks(
    elements: List[str],
    edge_index: np.ndarray,  # (2, E)
    dists: np.ndarray,       # (E,)
) -> np.ndarray:             # (N,) BB ids
    """
    Flood-fill connected components using only short covalent bonds.
    Lattice / long contacts are not traversed, so the COF linker and
    node building blocks are separated.
    """
    N = len(elements)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        d = float(dists[e])
        cutoff = _covalent_cutoff(elements[i], elements[j])
        if d <= cutoff:
            G.add_edge(i, j)

    bb_index = np.full(N, -1, dtype=np.int64)
    for comp_id, component in enumerate(nx.connected_components(G)):
        for atom in component:
            bb_index[atom] = comp_id

    return bb_index


def _approximate_smiles_for_bb(
    elements: List[str],
    atom_indices: List[int],
) -> str:
    """
    Returns a simplified formula string (not true SMILES) when RDKit is absent.
    """
    from collections import Counter
    counts = Counter(elements[i] for i in atom_indices)
    return "".join(f"{el}{n}" for el, n in sorted(counts.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Linkage type heuristics
# ─────────────────────────────────────────────────────────────────────────────

def _infer_linkage_from_elements(elements: List[str]) -> str:
    """
    Very rough heuristic based on element composition.
    Proper inference requires RDKit SMARTS matching.
    """
    el_set = set(elements)
    if "B" in el_set and "O" in el_set and "N" not in el_set:
        return "boronate_ester"
    if "B" in el_set and "O" in el_set and "N" not in el_set:
        return "boroxine"
    if "N" in el_set and "C" in el_set and "O" not in el_set:
        return "imine"
    if "N" in el_set and "C" in el_set and "O" in el_set:
        # Could be hydrazone, beta-ketoenamine, imide, urea…
        return "beta_ketoenamine"  # most stable, common in functional COFs
    return "<UNK>"


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# pyCOFBuilder name parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_pcb_name(stem: str) -> dict:
    """
    Parse a pyCOFBuilder structure name into its components.
    Format: {Node_BB_FG}-{Linker_BB_FG}-{NET}_A-{STACKING}
    Returns dict with node_bb, linker_bb, topology, stacking, node_func, linker_func.
    Works for both ReDD-COFFEE and manually named structures.
    """
    parts = stem.split("-")
    if len(parts) < 4:
        return {}

    stacking = parts[-1].upper()
    net_part = parts[-2].upper()
    topology = net_part.split("_")[0].lower()

    CONNECTION_GROUPS = {
        "NH2", "CHO", "B(OH)2", "OH", "CN", "NHNH2",
        "H", "F", "NO2", "OMe", "Me", "BETA_KETO", "DIOL",
    }

    def parse_bb(bb_str: str):
        tokens = bb_str.split("_")
        conn_idx = len(tokens)
        for i, t in enumerate(tokens):
            if t.upper() in CONNECTION_GROUPS:
                conn_idx = i
                break
        core = "_".join(tokens[:conn_idx]) if conn_idx > 0 else bb_str
        func = "_".join(tokens[conn_idx:]) if conn_idx < len(tokens) else "NH2"
        return core, func

    node_bb,   node_fg   = parse_bb(parts[0])
    linker_bb, linker_fg = parse_bb(parts[1])

    return {
        "node_bb":    node_bb,
        "node_func":  node_fg,
        "linker_bb":  linker_bb,
        "linker_func": linker_fg,
        "topology":   topology,
        "stacking":   stacking,
    }


def cif_to_crystal_graph(
    cif_path: Path,
    cutoff: float = 5.0,
    linkage_type: Optional[str] = None,
    topology: Optional[str] = None,
    stacking: Optional[str] = None,
    properties: Optional[Dict[str, float]] = None,
) -> CrystalGraph:
    """
    Parse a CIF file and return a CrystalGraph.

    Parameters
    ----------
    cif_path      : Path to .cif file
    cutoff        : Neighbour cutoff in Å (default 5 Å covers all covalent bonds
                    and some π-stacking contacts in 2D COFs)
    linkage_type  : Override inferred linkage (if known from database metadata)
    topology      : Net name (e.g. "hcb") — from pyCOFBuilder filename if known
    stacking      : Stacking pattern (e.g. "AA", "AB")
    properties    : Dict of pre-computed property labels to attach

    Returns
    -------
    CrystalGraph
    """
    cif_path = Path(cif_path)
    name = cif_path.stem
    # Also try to use the data_ block name from inside the CIF (more reliable
    # than the filename when using temp files or renamed downloads)
    _cif_text = cif_path.read_text()
    for _line in _cif_text.splitlines():
        _stripped = _line.strip()
        if _stripped.lower().startswith("data_"):
            _block_name = _stripped[5:].strip()
            if _block_name:
                name = _block_name
            break

    # ── Parse structure ──────────────────────────────────────────────────────
    if HAS_PYMATGEN:
        struct = CifParser(str(cif_path)).get_structures(primitive=False)[0]
        elements  = [str(s.specie.symbol) for s in struct]
        frac_coords = np.array([s.frac_coords for s in struct], dtype=np.float32)
        latt = struct.lattice
        a, b, c = latt.a, latt.b, latt.c
        alpha, beta, gamma = latt.alpha, latt.beta, latt.gamma
    else:
        parsed = _parse_cif_minimal(cif_path.read_text())
        if not parsed:
            raise ValueError(f"Could not parse CIF: {cif_path}")
        elements   = parsed["elements"]
        frac_coords = np.array(
            list(zip(parsed["frac_x"], parsed["frac_y"], parsed["frac_z"])),
            dtype=np.float32,
        )
        a, b, c = parsed["a"], parsed["b"], parsed["c"]
        alpha, beta, gamma = parsed["alpha"], parsed["beta"], parsed["gamma"]

    cell = _cell_matrix(a, b, c, alpha, beta, gamma).astype(np.float32)

    # ── Build neighbour list ─────────────────────────────────────────────────
    edge_index, shifts, dists = _pbc_edges(frac_coords, cell, cutoff=cutoff)

    # ── Atom features ────────────────────────────────────────────────────────
    atoms = np.stack([atom_features(el) for el in elements], axis=0)

    # ── Bond features ────────────────────────────────────────────────────────
    edge_attr = np.stack([
        bond_features(distance_angstrom=float(d))
        for d in dists
    ], axis=0) if len(dists) else np.zeros((0, BOND_FEAT_DIM), dtype=np.float32)

    # ── Building-block identification ────────────────────────────────────────
    bb_index = _identify_building_blocks(elements, edge_index, dists)

    # Collect SMILES (or formula) per unique BB
    n_bbs = int(bb_index.max()) + 1 if len(bb_index) else 0
    bb_smiles_list: List[str] = []
    for bb_id in range(n_bbs):
        indices = [i for i, bid in enumerate(bb_index) if bid == bb_id]
        bb_smiles_list.append(_approximate_smiles_for_bb(elements, indices))

    # ── Linkage / topology inference ─────────────────────────────────────────
    if linkage_type is None:
        linkage_type = _infer_linkage_from_elements(elements)

    # Parse pyCOFBuilder filename to extract topology, stacking, and BB identity
    _pcb = parse_pcb_name(name)

    if topology is None:
        topology = _pcb.get("topology", "<UNK>") or "<UNK>"
    if stacking is None:
        stacking = _pcb.get("stacking", "AA") or "AA"

    # Attach BB metadata to properties so dataset can build node/linker indices
    if _pcb:
        bb_meta = {
            "pcb_node_bb":    _pcb.get("node_bb",    ""),
            "pcb_linker_bb":  _pcb.get("linker_bb",  ""),
            "pcb_node_func":  _pcb.get("node_func",  ""),
            "pcb_linker_func": _pcb.get("linker_func", ""),
        }
        if properties is None:
            properties = {}
        properties.update({k: v for k, v in bb_meta.items() if k not in properties})

    return CrystalGraph(
        atoms        = atoms,
        frac_coords  = frac_coords,
        lattice      = lattice_features(a, b, c, alpha, beta, gamma),
        edge_index   = edge_index,
        edge_attr    = edge_attr,
        edge_shift   = shifts,
        bb_index     = bb_index,
        bb_smiles    = bb_smiles_list,
        linkage_type = linkage_type,
        topology     = topology,
        stacking     = stacking,
        name         = name,
        properties   = properties or {},
    )


def from_dict_spec(
    bb_smiles: List[str],
    linkage_type: str,
    topology: str,
    stacking: str = "AA",
    properties: Optional[Dict[str, float]] = None,
) -> "CrystalGraphSpec":
    """
    Lightweight spec object used by the decoder — no actual atom positions,
    just the building-block specification tuple.
    Used as the *target* for the encoder and the *output* of the decoder.
    """
    return CrystalGraphSpec(
        bb_smiles    = bb_smiles,
        linkage_type = linkage_type,
        topology     = topology,
        stacking     = stacking,
        properties   = properties or {},
    )


@dataclass
class CrystalGraphSpec:
    """Coarse-grained COF specification — the decoder's target output."""
    bb_smiles:    List[str]
    linkage_type: str
    topology:     str
    stacking:     str = "AA"
    properties:   Dict[str, float] = field(default_factory=dict)

    def to_label_vector(self) -> np.ndarray:
        """
        Encode spec as a compact integer label vector for classifier training:
        [linkage_idx, topology_idx, stacking_idx]
        """
        from utils.featurisation import LINKAGE_TO_IDX, TOPOLOGY_TO_IDX, STACKING_TO_IDX
        return np.array([
            LINKAGE_TO_IDX.get(self.linkage_type, 0),
            TOPOLOGY_TO_IDX.get(self.topology, 0),
            STACKING_TO_IDX.get(self.stacking, 0),
        ], dtype=np.int64)
