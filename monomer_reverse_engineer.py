"""
monomer_reverse_engineer.py
============================
Reverse-engineers the monomers (building blocks) from a COF crystal structure.

Two-path strategy:
  PATH A — Name-based (pyCOFBuilder convention):
    Parse the data_ block or filename → extract BB names directly
    Works for: ReDD-COFFEE, pyCOFBuilder outputs, our synthetic CIFs
    Confidence: HIGH (exact)

  PATH B — Atom-graph based (unknown/real CIFs):
    1. Build intra-cell bond graph from fractional coords + covalent radii
    2. Detect linkage bonds by element pair + distance heuristics
    3. Flood-fill remaining graph → one connected component per BB
    4. Fingerprint each component (C/N/H/O counts + total size)
    5. Match against known BB library by composition similarity

Output:
  ReverseEngineerResult with:
    - node_bb, linker_bb (names from our library, or "unknown")
    - confidence (0–1)
    - linkage_type
    - topology guess
    - fragment SMILES (approximate, rule-based)
    - monomer molecular formulas
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Covalent radii (Å) for bond detection
# ─────────────────────────────────────────────────────────────────────────────

COVALENT_RADII: Dict[str, float] = {
    'H': 0.31, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'F': 0.57, 'Si':1.11, 'P': 1.07, 'S': 1.05, 'Cl':1.02,
    'Br':1.20, 'Zn':1.22, 'Fe':1.32, 'Co':1.26, 'Ni':1.24,
    'Cu':1.32, 'Mn':1.61, 'default': 1.0,
}

def cov_cutoff(el1: str, el2: str, tolerance: float = 1.20) -> float:
    r1 = COVALENT_RADII.get(el1, COVALENT_RADII['default'])
    r2 = COVALENT_RADII.get(el2, COVALENT_RADII['default'])
    return (r1 + r2) * tolerance


# ─────────────────────────────────────────────────────────────────────────────
# Known building block compositions
# ─────────────────────────────────────────────────────────────────────────────

# (C, N, H, O, total_heavy) — H excluded from total_heavy
BB_LIBRARY_COMP: Dict[str, Tuple[int,int,int,int,int]] = {
    #          C    N   H   O  heavy
    "T3_BENZ": (6,  0,  3,  0,  6),
    "T3_TRIZ": (3,  3,  0,  0,  6),
    "T3_TPM":  (22, 0, 15,  0, 22),
    "T3_TPA":  (18, 1, 12,  0, 19),
    "T3_TRIF": (9,  0,  6,  0,  9),
    "T3_INTZ": (12, 2,  8,  0, 14),
    "S4_BENZ": (6,  0,  2,  0,  6),
    "S4_PORPH":(44, 4, 26,  0, 48),
    "S4_PHTH": (32, 8, 16,  0, 40),
    "L2_BENZ": (6,  0,  4,  0,  6),
    "L2_NAPH": (10, 0,  6,  0, 10),
    "L2_BIPH": (12, 0,  8,  0, 12),
    "L2_TPHN": (18, 0, 12,  0, 18),
    "L2_ANTR": (14, 0,  8,  0, 14),
    "L2_PYRN": (16, 0,  8,  0, 16),
    "L2_AZBN": (12, 2,  8,  0, 14),
    "L2_ETBE": (10, 0,  6,  0, 10),
    "L2_STIL": (14, 0, 10,  0, 14),
    "L2_BTTA": (15, 3,  9,  0, 18),
    # Dianhydride linkers for imide-linked COFs
    # (C, N, H, O, total_heavy_atoms)
    "L2_PMDA":  (10, 0,  2,  6, 16),   # pyromellitic dianhydride       C10H2O6
    "L2_NTCDA": (14, 0,  4,  6, 20),   # naphthalene-1,4,5,8-DA         C14H4O6
    "L2_BTDA":  (17, 0,  6,  7, 24),   # benzophenone tetracarboxylic DA C17H6O7
}

# Approximate SMILES for each BB (for display/export)
BB_SMILES: Dict[str, str] = {
    "T3_BENZ": "Nc1cc(N)cc(N)c1",
    "T3_TRIZ": "Nc1nc(N)nc(N)n1",
    "T3_TPM":  "NC(c1ccccc1)(c1ccccc1)c1ccccc1",
    "T3_TPA":  "Nc1ccc(N(c2ccccc2)c2ccccc2)cc1",
    "T3_TRIF": "Nc1cccc(N)c1-c1cccc(N)c1",
    "T3_INTZ": "Nc1nc2cc(N)ccc2n1N",
    "S4_BENZ": "Nc1cc(N)cc(N)c1N",
    "S4_PORPH":"Nc1ccc(-c2cc3cc(-c4ccc(N)cc4)c4ccc(N)cc4c3n2)cc1N",
    "S4_PHTH": "Nc1ccc(-c2nc3ccc(N)cc3n2-c2nc3ccc(N)cc3n2)cc1",
    "L2_BENZ": "O=Cc1ccc(C=O)cc1",
    "L2_NAPH": "O=Cc1ccc2cc(C=O)ccc2c1",
    "L2_BIPH": "O=Cc1ccc(-c2ccc(C=O)cc2)cc1",
    "L2_TPHN": "O=Cc1ccc(-c2ccc(-c3ccc(C=O)cc3)cc2)cc1",
    "L2_ANTR": "O=Cc1ccc2cc3ccc(C=O)cc3cc2c1",
    "L2_PYRN": "O=Cc1ccc2ccc3ccc(C=O)cc3c2c1",
    "L2_AZBN": "O=Cc1ccc(/N=N/c2ccc(C=O)cc2)cc1",
    "L2_ETBE": "O=Cc1ccc(C#Cc2ccc(C=O)cc2)cc1",
    "L2_STIL": "O=Cc1ccc(/C=C/c2ccc(C=O)cc2)cc1",
    "L2_BTTA": "O=Cc1cc(-n2ccnn2)cc(-n2ccnn2)c1",
    # Dianhydride linkers — SMILES show the anhydride functional groups
    "L2_PMDA":  "O=C1OC(=O)c2cc3c(cc21)C(=O)OC3=O",
    "L2_NTCDA": "O=C1OC(=O)c2ccc3c(c21)C(=O)OC3=O",
    "L2_BTDA":  "O=C1OC(=O)c2ccc(-c3ccc4C(=O)OC(=O)c4c3)cc21",
}

# Connection group SMILES that get replaced in actual COF linkage
CONN_GROUP_SMILES: Dict[str, str] = {
    "NH2":    "-N",
    "CHO":    "-C=O",
    "B(OH)2": "-B(O)O",
    "OH":     "-O",
}


def match_bb_by_composition(
    n_c: int, n_n: int, n_h: int, n_o: int, n_heavy: int
) -> List[Tuple[str, float]]:
    """
    Match a fragment's element counts to the closest building block(s).
    Returns list of (bb_name, confidence) sorted by confidence descending.
    """
    results = []
    for bb, (bc, bn, bh, bo, bheavy) in BB_LIBRARY_COMP.items():
        # Weighted similarity — heavy atom count is most discriminating
        dc = abs(bc - n_c)
        dn = abs(bn - n_n) * 2.0   # N count strongly discriminating
        dh = abs(bh - n_h) * 0.3   # H is less reliable (CIF often omits H)
        do = abs(bo - n_o)
        dheavy = abs(bheavy - n_heavy) * 1.5

        total_diff = dc + dn + dh + do + dheavy
        # Convert to confidence (exponential decay)
        conf = math.exp(-0.15 * total_diff)
        results.append((bb, conf))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CIF parser (minimal, for unknown CIFs)
# ─────────────────────────────────────────────────────────────────────────────

def parse_cif_atoms(cif_text: str) -> Tuple[List[str], np.ndarray, Dict]:
    """
    Parse atom positions and cell parameters from CIF text.
    Returns (elements, frac_coords, cell_params_dict).
    """
    elements, fracs = [], []
    cell = {"a":1,"b":1,"c":1,"alpha":90,"beta":90,"gamma":90}

    lines = cif_text.splitlines()
    in_loop = False
    col_map = {}
    col_order = []

    for i, line in enumerate(lines):
        line = line.strip()

        # Cell parameters
        for key, ckey in [("_cell_length_a","a"),("_cell_length_b","b"),
                           ("_cell_length_c","c"),("_cell_angle_alpha","alpha"),
                           ("_cell_angle_beta","beta"),("_cell_angle_gamma","gamma")]:
            if line.lower().startswith(key):
                try:
                    # Handle values like "22.49(3)"
                    val = line.split()[-1].split('(')[0]
                    cell[ckey] = float(val)
                except: pass

        # Loop header
        if line == "loop_":
            in_loop = True; col_map = {}; col_order = []
            continue

        if in_loop and line.startswith("_atom_site"):
            tag = line.lower().strip()
            col_order.append(tag)
            col_map[tag] = len(col_order) - 1
            continue

        # Atom data row
        if in_loop and col_order and not line.startswith("_") and not line.startswith("loop_"):
            parts = line.split()
            if not parts or len(parts) < 3:
                in_loop = False; continue
            try:
                # Find element
                el = None
                for k in ("_atom_site_type_symbol", "_atom_site_label"):
                    if k in col_map:
                        raw = parts[col_map[k]]
                        el = re.sub(r'[^A-Za-z]', '', raw)
                        if len(el) > 2: el = el[:2]
                        el = el.capitalize()
                        if el in COVALENT_RADII: break

                if el is None or el not in COVALENT_RADII:
                    continue

                # Find fractional coordinates
                fx = fy = fz = None
                for tag, attr in [("_atom_site_fract_x","x"),
                                   ("_atom_site_fract_y","y"),
                                   ("_atom_site_fract_z","z")]:
                    if tag in col_map:
                        val = parts[col_map[tag]].split('(')[0]
                        if attr == 'x': fx = float(val)
                        elif attr == 'y': fy = float(val)
                        else: fz = float(val)

                if fx is None:
                    # try positional: assume type x y z
                    if len(parts) >= 4:
                        el = parts[0].capitalize()
                        fx,fy,fz = float(parts[1]),float(parts[2]),float(parts[3])

                if fx is not None and el in COVALENT_RADII:
                    elements.append(el)
                    fracs.append([fx % 1.0, fy % 1.0, fz % 1.0])
            except Exception:
                continue

    frac_coords = np.array(fracs) if fracs else np.zeros((0,3))
    return elements, frac_coords, cell


def build_bond_graph(
    elements:    List[str],
    cart_coords: np.ndarray,   # (N, 3) Cartesian
    tolerance:   float = 1.15,
) -> Dict[int, List[Tuple[int, float]]]:
    """Build intra-cell bond graph from Cartesian coordinates."""
    N = len(elements)
    graph: Dict[int, List[Tuple[int,float]]] = defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            d = float(np.linalg.norm(cart_coords[i] - cart_coords[j]))
            cut = cov_cutoff(elements[i], elements[j], tolerance)
            if 0.3 < d <= cut:
                graph[i].append((j, d))
                graph[j].append((i, d))
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# Linkage bond detection
# ─────────────────────────────────────────────────────────────────────────────

LINKAGE_BOND_SIGNATURES: List[Dict] = [
    {"pair": frozenset({"C","N"}), "d_min":1.24, "d_max":1.33, "type":"imine"},
    {"pair": frozenset({"C","N"}), "d_min":1.33, "d_max":1.42, "type":"hydrazone"},
    {"pair": frozenset({"B","O"}), "d_min":1.30, "d_max":1.42, "type":"boronate_ester"},
    {"pair": frozenset({"C","N"}), "d_min":1.42, "d_max":1.50, "type":"beta_ketoenamine_N"},
    {"pair": frozenset({"C","C"}), "d_min":1.48, "d_max":1.52, "type":"olefin_junction"},
    # Imide C-N bond: 1.38-1.41 Å (5-membered ring, slightly longer than imine C=N)
    # Distinguished from imine by: adjacent C=O groups on both sides of N
    {"pair": frozenset({"C","N"}), "d_min":1.37, "d_max":1.42, "type":"imide"},
]

def detect_linkage_bonds(
    elements: List[str],
    graph:    Dict[int, List[Tuple[int,float]]],
) -> Tuple[set, str]:
    """
    Detect linkage bonds in the molecular graph.
    Returns (set of (i,j) pairs, detected linkage type).
    """
    bond_counts: Counter = Counter()
    linkage_bonds: set = set()

    for i, nbrs in graph.items():
        for j, d in nbrs:
            if j <= i: continue
            pair = frozenset({elements[i], elements[j]})
            for sig in LINKAGE_BOND_SIGNATURES:
                if pair == sig["pair"] and sig["d_min"] <= d <= sig["d_max"]:
                    linkage_bonds.add((min(i,j), max(i,j)))
                    bond_counts[sig["type"]] += 1
                    break

    # Most common linkage type
    if bond_counts:
        detected = bond_counts.most_common(1)[0][0]
    else:
        # Fallback: check C-N bonds by degree (imine C has degree 3, N has degree 2)
        detected = "imine"   # most common default
        for i, nbrs in graph.items():
            if elements[i] == "N" and len(nbrs) == 2:
                for j, d in nbrs:
                    if elements[j] == "C" and len(graph[j]) == 3:
                        linkage_bonds.add((min(i,j), max(i,j)))
                        detected = "imine"

    # Normalise linkage type name
    if "boronate" in detected: detected = "boronate_ester"
    elif "hydrazone" in detected: detected = "hydrazone"
    elif "beta_keto" in detected: detected = "beta_ketoenamine"
    elif "olefin" in detected: detected = "olefin"
    else: detected = "imine"

    return linkage_bonds, detected


def fragment_graph(
    n_atoms:       int,
    graph:         Dict[int, List[Tuple[int,float]]],
    remove_bonds:  set,   # set of (i,j) pairs to sever
) -> List[frozenset]:
    """Flood-fill the graph minus severed bonds → list of components."""
    G_pruned: Dict[int, set] = defaultdict(set)
    for i, nbrs in graph.items():
        for j, d in nbrs:
            if (min(i,j), max(i,j)) not in remove_bonds:
                G_pruned[i].add(j)

    visited = set()
    components = []
    for start in range(n_atoms):
        if start in visited: continue
        comp = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in comp: continue
            comp.add(node)
            for nb in G_pruned[node]:
                if nb not in comp: stack.append(nb)
        visited |= comp
        components.append(frozenset(comp))
    return components


# ─────────────────────────────────────────────────────────────────────────────
# Topology inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_topology(
    node_bb:   str,
    linker_bb: str,
    a: float, b: float, c: float, gamma: float,
    n_node_fragments: int, n_linker_fragments: int,
) -> str:
    """Infer 2D topology from BB types and unit cell shape."""
    if abs(gamma - 120.0) < 5 and node_bb.startswith("T3_"):
        if n_node_fragments == 2 and n_linker_fragments == 3:
            return "hcb"
        if n_node_fragments == 1 and n_linker_fragments == 2:
            return "kgm_unit"
        return "hcb"
    if abs(gamma - 90.0) < 5 and node_bb.startswith("S4_"):
        return "sql"
    if node_bb.startswith("T3_") and abs(gamma - 120.0) < 5:
        return "hcb"
    if node_bb.startswith("S4_"):
        return "sql"
    return "<UNK>"


# ─────────────────────────────────────────────────────────────────────────────
# Main result class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReverseEngineerResult:
    # Identified building blocks
    node_bb:       str
    linker_bb:     str
    node_bb_conf:  float   # 0–1
    linker_bb_conf: float

    # Chemical identity
    linkage_type:  str
    topology:      str

    # Monomer information
    node_formula:   str    # e.g. "C6H3N3"
    linker_formula: str
    node_smiles:    str
    linker_smiles:  str

    # Fragment data
    n_node_frags:   int = 0
    n_linker_frags: int = 0

    # Source
    method:        str = "atom_graph"   # "name_parser" or "atom_graph"
    confidence:    float = 0.0

    def summary(self) -> str:
        conf_bar = "█" * int(self.confidence * 10) + "░" * (10 - int(self.confidence * 10))
        return "\n".join([
            f"┌─ COF Monomer Reverse Engineering ─────────────────────┐",
            f"│  Method:    {self.method:<42}│",
            f"│  Confidence:[{conf_bar}] {self.confidence:.0%:<20}│",
            f"├─────────────────────────────────────────────────────────┤",
            f"│  Node BB:   {self.node_bb:<30} ({self.node_bb_conf:.0%}) │",
            f"│  Linker BB: {self.linker_bb:<30} ({self.linker_bb_conf:.0%}) │",
            f"│  Linkage:   {self.linkage_type:<42}│",
            f"│  Topology:  {self.topology:<42}│",
            f"├─────────────────────────────────────────────────────────┤",
            f"│  Node formula:   {self.node_formula:<36}│",
            f"│  Linker formula: {self.linker_formula:<36}│",
            f"│  Node SMILES:                                           │",
            f"│    {self.node_smiles[:52]:<52}│",
            f"│  Linker SMILES:                                         │",
            f"│    {self.linker_smiles[:52]:<52}│",
            f"└─────────────────────────────────────────────────────────┘",
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def reverse_engineer_cif(
    cif_path:   Path,
    cif_text:   Optional[str] = None,
    verbose:    bool = False,
) -> ReverseEngineerResult:
    """
    Reverse-engineer the building blocks from a COF CIF file.

    Tries PATH A (name-based) first; falls back to PATH B (atom-graph).
    """
    cif_path = Path(cif_path)
    if cif_text is None:
        cif_text = cif_path.read_text()

    # ── PATH A: Name-based (fast, exact) ─────────────────────────────────
    from data.crystal_graph import parse_pcb_name
    # Try data_ block name first, then filename
    block_name = None
    for line in cif_text.splitlines():
        if line.strip().lower().startswith("data_"):
            block_name = line.strip()[5:].strip()
            break
    name_to_parse = block_name or cif_path.stem

    pcb = parse_pcb_name(name_to_parse)
    if pcb and pcb.get("node_bb") and pcb.get("linker_bb"):
        node_bb   = pcb["node_bb"]
        linker_bb = pcb["linker_bb"]
        linkage   = pcb.get("linkage", "imine")
        topology  = pcb.get("topology", "<UNK>")

        # Infer linkage from node/linker func groups
        func_map = {
            "NH2+CHO":      "imine",
            "NH2+B(OH)2":   "boronate_ester",
            "NH2+beta_keto":"beta_ketoenamine",
            "OH+B(OH)2":    "boronate_ester",
            "NHNH2+CHO":    "hydrazone",
            "NH2+CN":       "triazine",
            "NH2+OH":       "beta_ketoenamine",
        }
        lk_key = f"{pcb.get('node_func','NH2')}+{pcb.get('linker_func','CHO')}"
        linkage = func_map.get(lk_key, "imine")

        return ReverseEngineerResult(
            node_bb        = node_bb,
            linker_bb      = linker_bb,
            node_bb_conf   = 1.0,
            linker_bb_conf = 1.0,
            linkage_type   = linkage,
            topology       = topology,
            node_formula   = _formula_from_bb(node_bb),
            linker_formula = _formula_from_bb(linker_bb),
            node_smiles    = BB_SMILES.get(node_bb,   "unknown"),
            linker_smiles  = BB_SMILES.get(linker_bb, "unknown"),
            method         = "name_parser",
            confidence     = 1.0,
        )

    # ── PATH B: Atom-graph decomposition ─────────────────────────────────
    elements, frac_coords, cell = parse_cif_atoms(cif_text)

    if len(elements) < 4:
        return _unknown_result("insufficient atoms in CIF")

    # Cartesian coordinates
    a,b,c = cell["a"],cell["b"],cell["c"]
    alpha,beta,gamma_ = cell["alpha"],cell["beta"],cell["gamma"]
    M = _make_cell_matrix(a, b, c, alpha, beta, gamma_)
    cart = frac_coords @ M

    # Build bond graph
    graph = build_bond_graph(elements, cart, tolerance=1.15)

    # Detect linkage bonds
    linkage_bonds, linkage_type = detect_linkage_bonds(elements, graph)

    if verbose:
        print(f"  Atoms: {len(elements)}, linkage bonds: {len(linkage_bonds)}")

    # Fragment decomposition
    components = fragment_graph(len(elements), graph, linkage_bonds)
    if verbose:
        print(f"  Fragments: {len(components)}")

    # Skip tiny fragments (isolated H or connection atoms)
    components = [c for c in components if len(c) >= 4]

    if not components:
        return _unknown_result("fragmentation produced no significant components")

    # Classify each fragment
    frag_matches = []
    for comp in components:
        comp_els = [elements[i] for i in comp]
        el_count = Counter(comp_els)
        n_c = el_count.get("C",0)
        n_n = el_count.get("N",0)
        n_h = el_count.get("H",0)
        n_o = el_count.get("O",0)
        n_heavy = sum(v for k,v in el_count.items() if k != "H")

        matches = match_bb_by_composition(n_c, n_n, n_h, n_o, n_heavy)
        formula = _make_formula(el_count)
        frag_matches.append({
            "comp": comp,
            "n_atoms": len(comp),
            "n_heavy": n_heavy,
            "formula": formula,
            "best_bb": matches[0][0],
            "best_conf": matches[0][1],
            "all_matches": matches[:3],
        })

    if not frag_matches:
        return _unknown_result("no fragments matched")

    # Separate node and linker fragments by BB type prefix
    node_frags   = [f for f in frag_matches if f["best_bb"].startswith(("T3_","S4_"))]
    linker_frags = [f for f in frag_matches if f["best_bb"].startswith("L2_")]

    if not node_frags and not linker_frags:
        # Try largest vs smallest fragment heuristic
        by_size = sorted(frag_matches, key=lambda f: f["n_heavy"])
        linker_frags = by_size[:len(by_size)//2]
        node_frags   = by_size[len(by_size)//2:]

    # Pick best representative (highest confidence)
    node_frag   = max(node_frags,   key=lambda f: f["best_conf"]) if node_frags   else None
    linker_frag = max(linker_frags, key=lambda f: f["best_conf"]) if linker_frags else None

    if node_frag is None and linker_frag:
        node_frag = linker_frag

    node_bb       = node_frag["best_bb"]    if node_frag   else "T3_BENZ"
    node_conf     = node_frag["best_conf"]  if node_frag   else 0.0
    node_formula  = node_frag["formula"]    if node_frag   else "C6H3"
    linker_bb     = linker_frag["best_bb"]  if linker_frag else "L2_BENZ"
    linker_conf   = linker_frag["best_conf"]if linker_frag else 0.0
    linker_formula= linker_frag["formula"]  if linker_frag else "C6H4"

    topology = infer_topology(
        node_bb, linker_bb, a, b, c, gamma_,
        len(node_frags), len(linker_frags),
    )

    overall_conf = 0.5 * (node_conf + linker_conf)

    return ReverseEngineerResult(
        node_bb        = node_bb,
        linker_bb      = linker_bb,
        node_bb_conf   = node_conf,
        linker_bb_conf = linker_conf,
        linkage_type   = linkage_type,
        topology       = topology,
        node_formula   = node_formula,
        linker_formula = linker_formula,
        node_smiles    = BB_SMILES.get(node_bb,   f"[Unknown:{node_bb}]"),
        linker_smiles  = BB_SMILES.get(linker_bb, f"[Unknown:{linker_bb}]"),
        n_node_frags   = len(node_frags),
        n_linker_frags = len(linker_frags),
        method         = "atom_graph",
        confidence     = overall_conf,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_cell_matrix(a,b,c,alpha,beta,gamma) -> np.ndarray:
    ar,br,gr = math.radians(alpha),math.radians(beta),math.radians(gamma)
    ca,cb,cg = math.cos(ar),math.cos(br),math.cos(gr)
    sg = math.sin(gr)
    vol = math.sqrt(max(1-ca**2-cb**2-cg**2+2*ca*cb*cg, 1e-10))
    return np.array([[a,0,0],[b*cg,b*sg,0],[c*cb,c*(ca-cb*cg)/sg,c*vol/sg]])


def _formula_from_bb(bb: str) -> str:
    if bb not in BB_LIBRARY_COMP:
        return "unknown"
    c,n,h,o,_ = BB_LIBRARY_COMP[bb]
    parts = []
    if c: parts.append(f"C{c}")
    if h: parts.append(f"H{h}")
    if n: parts.append(f"N{n}")
    if o: parts.append(f"O{o}")
    return "".join(parts)


def _make_formula(el_count: Counter) -> str:
    order = ["C","H","N","O","S","B","F","Cl","Br"]
    parts = []
    for el in order:
        if el_count.get(el,0):
            parts.append(f"{el}{el_count[el]}" if el_count[el]>1 else el)
    for el, cnt in sorted(el_count.items()):
        if el not in order and cnt:
            parts.append(f"{el}{cnt}" if cnt>1 else el)
    return "".join(parts)


def _unknown_result(reason: str) -> ReverseEngineerResult:
    return ReverseEngineerResult(
        node_bb="unknown", linker_bb="unknown",
        node_bb_conf=0.0, linker_bb_conf=0.0,
        linkage_type="unknown", topology="unknown",
        node_formula="?", linker_formula="?",
        node_smiles="?", linker_smiles="?",
        method="atom_graph", confidence=0.0,
    )
