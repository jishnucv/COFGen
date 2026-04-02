"""
bb_decomposer.py
================
Identifies building-block subgraphs and linkage bonds in COF crystal structures
using SMARTS pattern matching (RDKit) or a distance+element heuristic fallback.

The core problem: the flood-fill in crystal_graph.py uses a single distance
cutoff, which over-segments (each atom becomes its own BB) when imine bonds
(C=N, ~1.28 Å) and aromatic C-C bonds (~1.40 Å) have similar lengths.

Correct approach:
  1. Identify all linkage bonds by chemistry (SMARTS)
  2. Remove them from the connectivity graph
  3. Flood-fill the remaining graph → each component = one BB

Linkage SMARTS patterns (covers all major COF chemistries):
  Imine:           [C;R0]=[N;R0]     (non-ring C=N)
  Hydrazone:       [C;R0]=N-N
  Boronate ester:  [B](-O)-O
  Boroxine:        [B]1-O-[B]-O-[B]-O-1
  Beta-ketoenamine: [NH]-[C]=[C]-[C]=O (tautomeric — treat C-N as linkage)
  Triazine:        c1ncncn1           (CTF linkage is C-C to triazine ring)
  Imide:           [C](=O)-N-[C](=O)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ── Linkage SMARTS ────────────────────────────────────────────────────────────

LINKAGE_SMARTS: Dict[str, List[str]] = {
    "imine": [
        "[C;!R]=[N;!R]",           # imine C=N (non-ring)
        "[CH]=[N;!R]",             # aldehyde-derived imine
    ],
    "hydrazone": [
        "[C;!R]=N-N",
    ],
    "boronate_ester": [
        "[B](-[OH])-[OH]",         # boronic acid (before condensation)
        "[B]1-O-C-C-O-1",          # catechol boronate ester ring
        "[B](-O)-O",               # general boronate
    ],
    "boroxine": [
        "[B]1-O-[B]-O-[B]-O-1",
    ],
    "beta_ketoenamine": [
        "[NH]-[C]=[C]",            # enamine part
        "[N]-[c]",                 # N to aromatic ring (simplified)
    ],
    "triazine": [
        "c1ncncn1",                # 1,3,5-triazine
    ],
    "imide": [
        "[C](=O)[NH][C](=O)",
    ],
    "squaraine": [
        "[C](=O)-[c]",
    ],
    "olefin": [
        "[CH]=[CH]",               # Knoevenagel / Wittig
    ],
}


def smarts_linkage_bonds(
    mol,            # rdkit.Chem.Mol
    linkage_hint: Optional[str] = None,
) -> Set[Tuple[int, int]]:
    """
    Find all linkage bonds in an RDKit molecule using SMARTS matching.
    Returns a set of (i, j) pairs with i < j.
    """
    patterns_to_try = []
    if linkage_hint and linkage_hint in LINKAGE_SMARTS:
        patterns_to_try = LINKAGE_SMARTS[linkage_hint]
    else:
        # Try all patterns and union
        for pats in LINKAGE_SMARTS.values():
            patterns_to_try.extend(pats)

    linkage_pairs: Set[Tuple[int, int]] = set()
    for smarts in patterns_to_try:
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                # The bond between first two atoms of the match is the linkage
                if len(match) >= 2:
                    i, j = match[0], match[1]
                    linkage_pairs.add((min(i,j), max(i,j)))
        except Exception:
            continue
    return linkage_pairs


# ── Heuristic fallback (no RDKit) ────────────────────────────────────────────

# These element pairs and distance ranges signal linkage bonds
# Distinguished from intra-BB bonds by being longer / at ring boundaries
LINKAGE_BOND_HEURISTICS: List[Dict] = [
    # Imine C=N: shorter than C-N single but non-ring
    {"elements": frozenset({"C","N"}), "d_min": 1.24, "d_max": 1.32},
    # Boronate C-O-B: O-B bond
    {"elements": frozenset({"O","B"}), "d_min": 1.30, "d_max": 1.50},
    # Boroxine B-O
    {"elements": frozenset({"B","O"}), "d_min": 1.35, "d_max": 1.42},
]


def heuristic_linkage_bonds(
    elements:   List[str],
    edge_index: np.ndarray,   # (2, E)
    dists:      np.ndarray,   # (E,)
    linkage_hint: Optional[str] = None,
) -> Set[Tuple[int, int]]:
    """
    Identify likely linkage bonds using element-pair distance heuristics.
    This is an approximation — SMARTS via RDKit is much more reliable.
    """
    linkage_pairs: Set[Tuple[int, int]] = set()
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0,e]), int(edge_index[1,e])
        d    = float(dists[e])
        ep   = frozenset({elements[i], elements[j]})
        for rule in LINKAGE_BOND_HEURISTICS:
            if ep == rule["elements"] and rule["d_min"] <= d <= rule["d_max"]:
                linkage_pairs.add((min(i,j), max(i,j)))
                break
    return linkage_pairs


# ── Main decomposer ───────────────────────────────────────────────────────────

def decompose_building_blocks(
    elements:     List[str],
    edge_index:   np.ndarray,   # (2, E)
    dists:        np.ndarray,   # (E,)
    linkage_hint: Optional[str] = None,
    mol = None,                 # optional pre-built RDKit mol
) -> Tuple[np.ndarray, Set[Tuple[int,int]]]:
    """
    Decompose a COF crystal graph into building-block subgraphs by removing
    linkage bonds.

    Returns
    -------
    bb_index     : (N,) array of building-block IDs per atom
    linkage_bonds: set of (i,j) bond pairs that are linkage bonds
    """
    N = len(elements)

    # Identify linkage bonds
    if HAS_RDKIT and mol is not None:
        linkage_bonds = smarts_linkage_bonds(mol, linkage_hint)
    else:
        linkage_bonds = heuristic_linkage_bonds(elements, edge_index, dists, linkage_hint)

    # Build graph with only non-linkage covalent bonds
    from data.crystal_graph import _covalent_cutoff
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0,e]), int(edge_index[1,e])
        d    = float(dists[e])
        pair = (min(i,j), max(i,j))
        if pair in linkage_bonds:
            continue
        cutoff = _covalent_cutoff(elements[i], elements[j])
        if d <= cutoff:
            G.add_edge(i, j)

    bb_index = np.full(N, -1, dtype=np.int64)
    for comp_id, comp in enumerate(nx.connected_components(G)):
        for atom in comp:
            bb_index[atom] = comp_id

    return bb_index, linkage_bonds


# ── Integration with crystal_graph.py ─────────────────────────────────────────

def replace_bb_index_with_decomposed(
    graph,              # CrystalGraph (mutates bb_index in place)
    elements: List[str],
    dists:    np.ndarray,
) -> None:
    """
    Replace the naive flood-fill BB index with the linkage-aware decomposition.
    Called from cif_to_crystal_graph when RDKit is available.
    """
    bb_index, linkage_bonds = decompose_building_blocks(
        elements, graph.edge_index, dists,
        linkage_hint=graph.linkage_type,
    )
    graph.bb_index = bb_index

    # Rebuild bb_smiles using decomposed BBs
    from data.crystal_graph import _approximate_smiles_for_bb
    n_bbs = int(bb_index.max()) + 1 if len(bb_index) else 0
    graph.bb_smiles = [
        _approximate_smiles_for_bb(elements, [i for i,b in enumerate(bb_index) if b == bb_id])
        for bb_id in range(n_bbs)
    ]
