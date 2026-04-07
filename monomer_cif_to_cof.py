"""
monomer_cif_to_cof.py
=====================
Predict COF properties directly from two monomer CIF files.

Given the CIF files of a node building block (amine) and a linker
building block (aldehyde, dianhydride, boronic acid, etc.), this module:

  1. Parses each monomer CIF → element composition + 3D coordinates
  2. Detects functional groups (NH2, CHO, anhydride, B(OH)2, NHNH2, CN)
     by analysing the bond graph with distance-based heuristics
  3. Infers linkage type from the detected functional group pair
  4. Determines connectivity (ditopic L2, tritopic T3, tetratopic S4)
     from the number of reactive sites
  5. Estimates arm length (centroid → reactive atom distance)
     OR matches to the nearest known BB by molecular composition
  6. Generates a synthetic COF CIF using the calibrated reticular geometry
  7. Runs the full property prediction pipeline

No external databases needed — works entirely offline with numpy/scipy.

Usage
-----
    result = predict_cof_from_monomers(
        node_cif   = "TAPB.cif",
        linker_cif = "PDA.cif",
        topology   = "hcb",      # optional — inferred from connectivity
        stacking   = "AA",       # optional
    )
    print(result.summary)

Supported functional groups
---------------------------
  Node functional groups:
    NH2     → imine, beta-ketoenamine, hydrazone, imide
    B(OH)2  → boronate ester, boroxine
    NHNH2   → hydrazone
    CN      → triazine (ionothermal)

  Linker functional groups:
    CHO       → imine, hydrazone
    beta_keto → beta-ketoenamine
    anhydride → imide
    B(OH)2    → boronate ester, boroxine
    CN        → triazine

Limitations
-----------
  - Arm length estimation is most accurate when monomer matches a known BB.
    For truly custom molecules, expect ±15% on pore geometry.
  - Functional group detection uses distance heuristics, not bond orders.
    Amide C-N bonds (1.35 Å) may be confused with imide C-N (1.38 Å).
  - For porphyrin/phthalocyanine nodes, CIF must contain the full macrocycle
    (not just a fragment) for correct arm-length estimation.
"""

from __future__ import annotations

import math
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Covalent radii (Å) — bond detection
# ─────────────────────────────────────────────────────────────────────────────

_CRAD: Dict[str, float] = {
    'H': 0.31, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'F': 0.57, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
    'Br': 1.20, 'Zn': 1.22, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24,
    'Cu': 1.32, 'Mn': 1.61, 'Mg': 1.41,
}

def _bond_cutoff(e1: str, e2: str, tol: float = 1.18) -> float:
    return (_CRAD.get(e1, 1.0) + _CRAD.get(e2, 1.0)) * tol


# ─────────────────────────────────────────────────────────────────────────────
# CIF parser — lightweight, no external deps
# ─────────────────────────────────────────────────────────────────────────────

def _parse_monomer_cif(cif_text: str) -> Tuple[List[str], np.ndarray, Dict]:
    """
    Parse a monomer CIF file.
    Returns (elements, frac_coords, cell_params).
    Handles most common CIF conventions including disorder flags.
    """
    import re
    elements, fracs = [], []
    cell = {"a": 20.0, "b": 20.0, "c": 10.0,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0}

    lines = cif_text.splitlines()
    col_map: Dict[str, int] = {}
    col_order: List[str] = []
    in_loop = False
    loop_has_atom = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Cell parameters
        for key, ckey in [
            ("_cell_length_a", "a"), ("_cell_length_b", "b"),
            ("_cell_length_c", "c"), ("_cell_angle_alpha", "alpha"),
            ("_cell_angle_beta", "beta"), ("_cell_angle_gamma", "gamma"),
        ]:
            if stripped.lower().startswith(key):
                try:
                    val = stripped.split()[-1].split("(")[0]
                    cell[ckey] = float(val)
                except (ValueError, IndexError):
                    pass

        # Loop header
        if stripped == "loop_":
            in_loop = True
            loop_has_atom = False
            col_map = {}
            col_order = []
            continue

        # Column headers
        if in_loop and stripped.startswith("_atom_site"):
            tag = stripped.lower()
            col_order.append(tag)
            col_map[tag] = len(col_order) - 1
            loop_has_atom = True
            continue

        # Data row in atom loop
        if in_loop and loop_has_atom and not stripped.startswith("_"):
            if stripped.startswith("loop_"):
                in_loop = False
                loop_has_atom = False
                col_map = {}
                continue

            parts = stripped.split()
            if len(parts) < 4:
                continue

            # Skip disordered sites (occupancy < 1 or label contains '?')
            if any(p == "?" for p in parts):
                continue

            try:
                # Find element
                el = None
                for col in ("_atom_site_type_symbol", "_atom_site_label"):
                    if col in col_map:
                        raw = parts[col_map[col]]
                        # Strip trailing numbers and special chars from labels
                        cleaned = re.sub(r"[^A-Za-z]", "", raw)
                        if cleaned[:2] in _CRAD:
                            el = cleaned[:2]
                        elif cleaned[:1] in _CRAD:
                            el = cleaned[:1]
                        if el:
                            break

                if el is None and len(parts) >= 4:
                    # Positional fallback: first token is element
                    raw = parts[0]
                    cleaned = re.sub(r"[^A-Za-z]", "", raw)
                    if cleaned[:2] in _CRAD:
                        el = cleaned[:2]
                    elif cleaned[:1] in _CRAD:
                        el = cleaned[:1]

                if el is None:
                    continue

                # Fractional coordinates
                fx = fy = fz = None
                for tag, attr in [
                    ("_atom_site_fract_x", "x"),
                    ("_atom_site_fract_y", "y"),
                    ("_atom_site_fract_z", "z"),
                ]:
                    if tag in col_map:
                        val = parts[col_map[tag]].split("(")[0]
                        v = float(val)
                        if attr == "x":
                            fx = v
                        elif attr == "y":
                            fy = v
                        else:
                            fz = v

                if fx is None and len(parts) >= 4:
                    # Try direct positional parsing
                    try:
                        fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
                    except ValueError:
                        pass

                if fx is not None and fy is not None and fz is not None:
                    elements.append(el)
                    fracs.append([fx % 1.0, fy % 1.0, fz % 1.0])

            except (ValueError, IndexError):
                continue

    frac_coords = np.array(fracs, dtype=np.float64) if fracs else np.zeros((0, 3))
    return elements, frac_coords, cell


def _cell_matrix(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Convert cell parameters to Cartesian transformation matrix."""
    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    ca, cb, cg = math.cos(ar), math.cos(br), math.cos(gr)
    sg = math.sin(gr)
    vol = math.sqrt(max(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg, 1e-10))
    return np.array([
        [a, 0, 0],
        [b * cg, b * sg, 0],
        [c * cb, c * (ca - cb * cg) / sg, c * vol / sg],
    ], dtype=np.float64)


def _build_graph(elements: List[str], cart: np.ndarray, tol=1.18) -> Dict:
    """Build covalent bond graph from Cartesian coordinates."""
    from collections import defaultdict
    graph = defaultdict(list)
    N = len(elements)
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(cart[i] - cart[j]))
            cut = _bond_cutoff(elements[i], elements[j], tol)
            if 0.3 < d <= cut:
                graph[i].append((j, d))
                graph[j].append((i, d))
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# Functional group detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FunctionalGroup:
    """A detected reactive functional group in a monomer."""
    group_type:  str    # "NH2", "CHO", "anhydride", "B(OH)2", "NHNH2", "CN", "beta_keto"
    atom_idx:    int    # index of the key atom (N for NH2, C for CHO, B for B(OH)2)
    cart_pos:    np.ndarray   # 3D position
    confidence:  float  # 0-1


def detect_functional_groups(
    elements: List[str],
    cart:     np.ndarray,
    graph:    Dict,
) -> List[FunctionalGroup]:
    """
    Detect all reactive functional groups in a monomer by analysing
    the bond graph with distance-based heuristics.

    Detects: NH2, CHO, beta_keto, anhydride, B(OH)2, NHNH2, CN
    """
    groups: List[FunctionalGroup] = []
    N = len(elements)

    for i in range(N):
        el = elements[i]
        nbrs = graph.get(i, [])
        nbr_els = [elements[j] for j, _ in nbrs]
        nbr_dist = {elements[j]: d for j, d in nbrs}

        # ── NH2 — primary amine ───────────────────────────────────────────
        # N with ≥1 H neighbor and ≥1 C neighbor, not bonded to other N
        if el == 'N':
            n_H = nbr_els.count('H')
            n_C = nbr_els.count('C')
            n_N = nbr_els.count('N')
            if n_H >= 1 and n_C >= 1 and n_N == 0:
                conf = 0.95 if n_H == 2 else 0.80
                groups.append(FunctionalGroup("NH2", i, cart[i], conf))

        # ── CHO — aldehyde ────────────────────────────────────────────────
        # C with one short C=O (d<1.28 Å) and one C-H
        if el == 'C':
            o_short = [(j, d) for j, d in nbrs if elements[j] == 'O' and d < 1.28]
            h_nbrs  = [(j, d) for j, d in nbrs if elements[j] == 'H']
            if o_short and h_nbrs:
                groups.append(FunctionalGroup("CHO", i, cart[i], 0.90))

        # ── beta-ketoenamine precursor — 1,3-diketone (triformylphloroglucinol) ──
        # C with two short C=O bonds separated by one carbon
        if el == 'C':
            o_short = [(j, d) for j, d in nbrs if elements[j] == 'O' and d < 1.28]
            c_nbrs  = [j for j, d in nbrs if elements[j] == 'C']
            if len(o_short) >= 2:
                groups.append(FunctionalGroup("beta_keto", i, cart[i], 0.75))

        # ── Anhydride: bridging O between two carbonyl C atoms ─────────────
        # O bonded to exactly two C, each C has a short C=O
        if el == 'O' and len(nbrs) == 2:
            both_C = all(elements[j] == 'C' for j, _ in nbrs)
            if both_C:
                c_have_CO = []
                for cj, _ in nbrs:
                    c_nbrs_sub = graph.get(cj, [])
                    has_CO = any(elements[k] == 'O' and d < 1.28 for k, d in c_nbrs_sub)
                    c_have_CO.append(has_CO)
                if all(c_have_CO):
                    # Find the carbonyl C (use one of them as the key atom)
                    key_c = nbrs[0][0]
                    groups.append(FunctionalGroup("anhydride", key_c, cart[key_c], 0.85))

        # ── B(OH)2 — boronic acid ──────────────────────────────────────────
        if el == 'B':
            n_O = nbr_els.count('O')
            if n_O >= 2:
                groups.append(FunctionalGroup("B(OH)2", i, cart[i], 0.95))

        # ── NHNH2 — hydrazide / hydrazine ──────────────────────────────────
        if el == 'N':
            nbr_N_list = [(j, d) for j, d in nbrs if elements[j] == 'N']
            nbr_H_list = [(j, d) for j, d in nbrs if elements[j] == 'H']
            if nbr_N_list and nbr_H_list:
                groups.append(FunctionalGroup("NHNH2", i, cart[i], 0.80))

        # ── CN — nitrile (C≡N: ~1.15-1.18 Å, N is terminal with no H) ────────
        if el == 'C':
            for j, d in nbrs:
                if elements[j] == 'N' and d < 1.20:
                    # True nitrile N has no H neighbors (terminal atom)
                    n_nbrs  = graph.get(j, [])
                    n_has_H = any(elements[k] == 'H' for k, _ in n_nbrs)
                    if not n_has_H:
                        groups.append(FunctionalGroup("CN", i, cart[i], 0.70))

    # Deduplicate: if an atom appears in multiple groups, keep highest confidence
    seen = {}
    for g in groups:
        key = (g.atom_idx, g.group_type)
        if key not in seen or g.confidence > seen[key].confidence:
            seen[key] = g
    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# Linkage inference
# ─────────────────────────────────────────────────────────────────────────────

# Maps (node_func, linker_func) → linkage_type
FUNCGROUP_TO_LINKAGE: Dict[Tuple[str, str], str] = {
    ("NH2",    "CHO"):       "imine",
    ("NH2",    "anhydride"): "imide",
    ("NH2",    "beta_keto"): "beta_ketoenamine",
    ("NHNH2",  "CHO"):       "hydrazone",
    ("B(OH)2", "B(OH)2"):    "boroxine",
    ("B(OH)2", "diol"):      "boronate_ester",
    ("CN",     "CN"):        "triazine",
    # Reversed pairs (in case user swaps node/linker CIFs)
    ("CHO",       "NH2"):    "imine",
    ("anhydride",  "NH2"):   "imide",
    ("beta_keto",  "NH2"):   "beta_ketoenamine",
}


def infer_linkage(node_func: str, linker_func: str) -> str:
    """Infer linkage type from the two functional groups."""
    key = (node_func, linker_func)
    if key in FUNCGROUP_TO_LINKAGE:
        return FUNCGROUP_TO_LINKAGE[key]
    # Fuzzy: any amine + any carbonyl → imine
    if "NH" in node_func and linker_func in ("CHO", "beta_keto"):
        return "imine"
    if "NH" in linker_func and node_func in ("CHO", "beta_keto"):
        return "imine"
    return "imine"   # safe default


# ─────────────────────────────────────────────────────────────────────────────
# Arm-length estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_arm_length(
    elements:  List[str],
    cart:      np.ndarray,
    fg_atoms:  List[int],    # indices of reactive atoms
) -> float:
    """
    Arm length = mean distance from molecular centroid to reactive atoms.
    Falls back to max extent / 2 if fg_atoms is empty.
    """
    heavy_mask = np.array([el != 'H' for el in elements])
    heavy_cart = cart[heavy_mask]
    if len(heavy_cart) == 0:
        return 4.0

    centroid = heavy_cart.mean(axis=0)

    if fg_atoms:
        fg_coords = cart[np.array(fg_atoms)]
        dists = np.linalg.norm(fg_coords - centroid, axis=1)
        return float(round(dists.mean(), 2))
    else:
        # Fallback: max extent from centroid
        dists = np.linalg.norm(heavy_cart - centroid, axis=1)
        return float(round(dists.max() * 0.85, 2))


def match_or_estimate_arm_length(
    n_C: int, n_N: int, n_O: int, n_heavy: int,
    connectivity: str,
    fg_atoms: List[int] = None,
    elements: List[str] = None,
    cart: np.ndarray = None,
) -> Tuple[float, str, float]:
    """
    Returns (arm_length_Å, matched_bb_name, match_confidence).

    Priority:
      1. Geometric measurement from CIF coordinates (most accurate)
      2. Exact composition match to known BB library
      3. Scaled estimate from nearest composition match
    """
    from analysis.monomer_reverse_engineer import BB_LIBRARY_COMP, match_bb_by_composition
    from data.synthetic_cif_generator import BB_SIZES

    # 1. Geometric measurement from coordinates
    if fg_atoms is not None and elements is not None and cart is not None and len(elements):
        geo_length = estimate_arm_length(elements, cart, fg_atoms)
        # Also try composition match for BB identification
        matches = match_bb_by_composition(n_C, n_N, 0, n_O, n_heavy)
        bb_name = matches[0][0] if matches else "custom"
        conf    = matches[0][1] if matches else 0.0
        # Use geometric length unless it's wildly different from known BB
        if bb_name in BB_SIZES and abs(geo_length - BB_SIZES[bb_name]) > 3.0:
            # Big discrepancy: trust composition match for size, keep bb_name
            return BB_SIZES[bb_name], bb_name, conf * 0.7
        return geo_length, bb_name, conf if conf > 0.5 else 0.6

    # 2. Pure composition matching
    matches = match_bb_by_composition(n_C, n_N, 0, n_O, n_heavy)
    if matches:
        bb_name, conf = matches[0]
        if bb_name in BB_SIZES:
            return BB_SIZES[bb_name], bb_name, conf

    # 3. Absolute fallback based on heavy atom count
    if connectivity == "L2":
        arm = max(3.0, 0.52 * n_heavy)
    elif connectivity in ("T3", "S4"):
        arm = max(4.0, 0.45 * n_heavy)
    else:
        arm = max(3.5, 0.50 * n_heavy)
    return float(round(arm, 2)), "custom", 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Full monomer analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MonomierAnalysis:
    """Result of analysing one monomer CIF."""
    cif_name:      str
    n_atoms:       int
    formula:       str
    elements:      List[str]
    cart_coords:   np.ndarray
    functional_groups: List[FunctionalGroup]
    primary_fg:    str      # "NH2", "CHO", "anhydride", etc.
    n_reactive:    int      # 2, 3, or 4
    connectivity:  str      # "L2", "T3", "S4"
    role:          str      # "node" or "linker"
    arm_length:    float    # Å
    matched_bb:    str      # closest known BB name
    match_conf:    float    # 0–1
    topology_hint: str      # "hcb", "sql", or ""


def analyse_monomer(cif_path: Path, cif_text: str = None) -> MonomierAnalysis:
    """Analyse one monomer CIF and return its key properties."""
    cif_path = Path(cif_path)
    if cif_text is None:
        cif_text = cif_path.read_text()

    name = cif_path.stem

    # 1. Parse atoms
    elements, frac, cell = _parse_monomer_cif(cif_text)
    if len(elements) == 0:
        raise ValueError(f"No atoms found in {name}")

    # 2. Cartesian coordinates
    M    = _cell_matrix(cell["a"], cell["b"], cell["c"],
                        cell["alpha"], cell["beta"], cell["gamma"])
    cart = frac @ M

    # 3. Bond graph
    graph = _build_graph(elements, cart, tol=1.18)

    # 4. Functional groups
    fgs = detect_functional_groups(elements, cart, graph)

    # 5. Primary functional group — use chemical priority, not just count
    # NH2 > B(OH)2 > NHNH2 > CHO > beta_keto > anhydride > CN > unknown
    FG_PRIORITY = {
        "NH2": 10, "B(OH)2": 9, "NHNH2": 8,
        "CHO": 7,  "beta_keto": 6, "anhydride": 5, "CN": 4,
    }
    from collections import Counter as _Counter
    fg_counts = _Counter(g.group_type for g in fgs)
    if fg_counts:
        primary_fg = max(fg_counts.keys(), key=lambda g: FG_PRIORITY.get(g, 0))
        n_reactive = fg_counts[primary_fg]
    else:
        primary_fg = "unknown"
        n_reactive = 0

    # 6. Connectivity from number of reactive sites
    if n_reactive <= 0:
        # Try to infer from element counts
        ec = _Counter(elements)
        n_N = ec.get('N', 0)
        if n_N == 2:
            n_reactive = 2; connectivity = "L2"; role = "linker"
        elif n_N == 3:
            n_reactive = 3; connectivity = "T3"; role = "node"
        elif n_N == 4:
            n_reactive = 4; connectivity = "S4"; role = "node"
        else:
            n_reactive = 2; connectivity = "L2"; role = "linker"
    else:
        if n_reactive == 2:
            connectivity = "L2"; role = "linker"
        elif n_reactive == 3:
            connectivity = "T3"; role = "node"
        elif n_reactive >= 4:
            n_reactive = 4
            connectivity = "S4"; role = "node"
        else:
            connectivity = "L2"; role = "linker"

    # Topology hint from connectivity
    topo_hint = {"T3": "hcb", "S4": "sql", "L2": ""}.get(connectivity, "")

    # 7. Formula
    ec = _Counter(elements)
    formula_parts = []
    for el in ["C", "H", "N", "O", "B", "S", "F", "Cl"]:
        if ec.get(el, 0) > 0:
            formula_parts.append(f"{el}{ec[el]}" if ec[el] > 1 else el)
    formula = "".join(formula_parts) or "Unknown"

    # 8. Arm length
    fg_atom_idxs = [g.atom_idx for g in fgs if g.group_type == primary_fg]
    n_heavy = sum(v for k, v in ec.items() if k != 'H')
    arm_len, matched_bb, match_conf = match_or_estimate_arm_length(
        n_C=ec.get('C', 0), n_N=ec.get('N', 0), n_O=ec.get('O', 0),
        n_heavy=n_heavy, connectivity=connectivity,
        fg_atoms=fg_atom_idxs, elements=elements, cart=cart,
    )

    return MonomierAnalysis(
        cif_name       = name,
        n_atoms        = len(elements),
        formula        = formula,
        elements       = elements,
        cart_coords    = cart,
        functional_groups = fgs,
        primary_fg     = primary_fg,
        n_reactive     = n_reactive,
        connectivity   = connectivity,
        role           = role,
        arm_length     = arm_len,
        matched_bb     = matched_bb,
        match_conf     = match_conf,
        topology_hint  = topo_hint,
    )


# ─────────────────────────────────────────────────────────────────────────────
# COF prediction from two monomers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MonomierCOFResult:
    """Full prediction result from two monomer CIFs."""
    node:      MonomierAnalysis
    linker:    MonomierAnalysis
    linkage:   str
    topology:  str
    stacking:  str

    # Predicted properties
    a_angstrom:  float
    c_angstrom:  float
    void_fraction:  float
    bet_m2g:     float
    pld:         float
    lcd:         float
    density:     float
    band_gap:    float
    co2_uptake:  float
    ch4_uptake:  float
    h2_uptake:   float
    co2_n2_sel:  float
    thermal_C:   int
    water_stab:  str
    e_inplane:   float
    stacking_prefs: Dict[str, float]
    layer_spacing:  float
    pxrd_peaks:  List[Dict]
    synth_score: float
    synth_solvent:   str
    synth_catalyst:  str
    synth_temp:      str
    warnings:    List[str]

    @property
    def summary(self) -> str:
        w_str = "\n  ".join(self.warnings) if self.warnings else "none"
        peaks_str = "\n  ".join(
            f"({p['hkl']}) 2θ={p['two_theta']:.3f}° d={p['d']:.4f}Å I={p['I']:.0f}"
            for p in self.pxrd_peaks[:6]
        )
        return f"""
{'='*62}
  COFGen — Prediction from Monomer CIFs
{'='*62}
  Node:    {self.node.cif_name}  [{self.node.formula}]
           {self.node.n_reactive}× {self.node.primary_fg}  →  {self.node.connectivity}
           Matched: {self.node.matched_bb} (conf={self.node.match_conf:.0%})
           Arm length: {self.node.arm_length:.2f} Å

  Linker:  {self.linker.cif_name}  [{self.linker.formula}]
           {self.linker.n_reactive}× {self.linker.primary_fg}  →  {self.linker.connectivity}
           Matched: {self.linker.matched_bb} (conf={self.linker.match_conf:.0%})
           Arm length: {self.linker.arm_length:.2f} Å

  Linkage: {self.linkage}
  Topology: {self.topology}
  Stacking (predicted): {self.stacking}  ({self.stacking_prefs.get(self.stacking, 0):.0%} probability)

{'─'*62}
  Pore Geometry
    Cell a:         {self.a_angstrom:.2f} Å
    Void fraction:  {self.void_fraction:.4f}
    BET:            {self.bet_m2g:.0f} m²/g
    PLD:            {self.pld:.2f} Å
    LCD:            {self.lcd:.2f} Å
    Density:        {self.density:.4f} g/cm³

  Electronics
    Band gap:       {self.band_gap:.2f} eV

  Gas Adsorption
    CO₂ @ 298K/1bar:   {self.co2_uptake:.2f} mmol/g
    CH₄ @ 298K/65bar:  {self.ch4_uptake:.2f} mmol/g
    H₂  @ 77K/100bar:  {self.h2_uptake:.2f} mmol/g
    CO₂/N₂ sel:        {self.co2_n2_sel:.0f}×

  Mechanical
    E (in-plane):   {self.e_inplane:.1f} GPa

  Stability
    Thermal decomp: {self.thermal_C} °C
    Water:          {self.water_stab}

  PXRD Top Peaks (Cu Kα)
  {peaks_str}

  Synthesis Conditions
    Score:          {self.synth_score:.0%}
    Solvent:        {self.synth_solvent}
    Catalyst:       {self.synth_catalyst}
    Temperature:    {self.synth_temp}

  Warnings: {w_str}
{'='*62}"""


def predict_cof_from_monomers(
    node_cif:    Path,
    linker_cif:  Path,
    topology:    str = "",       # "" = auto-detect
    stacking:    str = "",       # "" = auto-predict
    node_text:   str = None,     # CIF text (if already loaded)
    linker_text: str = None,
    verbose:     bool = True,
) -> MonomierCOFResult:
    """
    Predict COF properties from two monomer CIF files.

    Parameters
    ----------
    node_cif    : path to the node building block CIF
    linker_cif  : path to the linker building block CIF
    topology    : "hcb", "sql", "kgm" — inferred from node connectivity if empty
    stacking    : "AA", "AB", "ABC" — predicted if empty

    Returns
    -------
    MonomierCOFResult with all predicted properties
    """
    import warnings as _warnings
    warn_list = []

    node_cif   = Path(node_cif)
    linker_cif = Path(linker_cif)

    if verbose:
        print(f"\n  Parsing node monomer:   {node_cif.name}")
    node_ana = analyse_monomer(node_cif, node_text)

    if verbose:
        print(f"  Parsing linker monomer: {linker_cif.name}")
    link_ana = analyse_monomer(linker_cif, linker_text)

    # ── Sanity checks & role swap ─────────────────────────────────────────
    # If user gave them backwards, auto-swap
    if node_ana.role == "linker" and link_ana.role == "node":
        if verbose:
            print("  Note: swapping node/linker (detected role mismatch)")
        node_ana, link_ana = link_ana, node_ana
        warn_list.append("Node and linker CIFs were automatically swapped based on connectivity.")

    if node_ana.role == "linker" and link_ana.role == "linker":
        warn_list.append("Both monomers detected as linkers — treating the larger one as node.")
        if node_ana.arm_length < link_ana.arm_length:
            node_ana, link_ana = link_ana, node_ana

    # ── Linkage inference ────────────────────────────────────────────────
    linkage = infer_linkage(node_ana.primary_fg, link_ana.primary_fg)
    if verbose:
        print(f"  Detected: {node_ana.connectivity} node ({node_ana.primary_fg}) "
              f"+ {link_ana.connectivity} linker ({link_ana.primary_fg}) "
              f"→ {linkage} linkage")

    # ── Topology ─────────────────────────────────────────────────────────
    if not topology:
        topology = node_ana.topology_hint or "hcb"

    # ── Arm lengths & cell parameter ─────────────────────────────────────
    from data.synthetic_cif_generator import LINKAGE_BOND_LENGTH
    link_bond = LINKAGE_BOND_LENGTH.get(linkage, 1.30)
    r_node    = node_ana.arm_length
    r_linker  = link_ana.arm_length
    a = 2.0 * (r_node + r_linker + link_bond)
    c = 3.60  # default; will be overridden by BB-specific layer spacing

    if verbose:
        print(f"  r_node={r_node:.2f}Å  r_linker={r_linker:.2f}Å  → a={a:.2f}Å")

    # ── Generate CIF and compute geometric properties ──────────────────────
    from data.synthetic_cif_generator import (
        generate_hcb_cif, generate_sql_cif, LINKAGE_BOND_LENGTH as LBL
    )
    from data.crystal_graph import cif_to_crystal_graph
    from data.property_labels import compute_geometric_properties
    from utils.featurisation import ELEMENTS as ELS

    # Use matched BB names if available, otherwise fall back to T3_BENZ/L2_BENZ
    node_bb   = node_ana.matched_bb   if node_ana.matched_bb   != "custom" else "T3_BENZ"
    linker_bb = link_ana.matched_bb   if link_ana.matched_bb   != "custom" else "L2_BENZ"
    nf, lf    = node_ana.primary_fg, link_ana.primary_fg
    # Normalise func groups to pyCOFBuilder convention
    if lf == "anhydride":
        lf = "anhydride"
    if nf not in ("NH2", "NHNH2", "B(OH)2", "CN"):
        nf = "NH2"
    if lf not in ("CHO", "anhydride", "beta_keto", "B(OH)2", "diol", "CN"):
        lf = "CHO"

    # Generate synthetic CIF with calibrated arm lengths
    # Override BB_SIZES temporarily with our measured values
    import data.synthetic_cif_generator as _scg
    _orig_sizes = dict(_scg.BB_SIZES)
    _scg.BB_SIZES[node_bb]   = r_node
    _scg.BB_SIZES[linker_bb] = r_linker

    try:
        if topology in ("hcb", "kgm", "hxl") and node_ana.connectivity == "T3":
            cif_text = generate_hcb_cif(node_bb, linker_bb, linkage, nf, lf)
        else:
            cif_text = generate_sql_cif(node_bb, linker_bb, linkage, nf, lf)
    finally:
        _scg.BB_SIZES.update(_orig_sizes)  # restore

    with tempfile.NamedTemporaryFile(suffix=".cif", mode="w", delete=False) as f:
        f.write(cif_text)
        tmp_cif = Path(f.name)

    try:
        cg  = cif_to_crystal_graph(tmp_cif, cutoff=4.0)
        geo = compute_geometric_properties(cg, n_grid=20)
    finally:
        tmp_cif.unlink()

    n_el     = len(ELS)
    elements = [ELS[int(cg.atoms[i][:n_el].argmax())] for i in range(cg.n_atoms)]
    import math as _math
    lat      = cg.lattice
    a_cif    = float(lat[0]) * 50
    c_cif    = float(lat[2]) * 50
    gamma    = _math.degrees(_math.acos(float(np.clip(lat[5], -1, 1))))
    n_arom   = elements.count('C')

    # ── Stacking ──────────────────────────────────────────────────────────
    from analysis.pxrd_simulator import (
        analyse_stacking, predict_preferred_stacking, _BB_LAYER_D
    )
    stk  = analyse_stacking(cg.frac_coords, a_cif, a_cif, c_cif,
                             90.0, 90.0, gamma, node_bb=node_bb)
    # Use literature layer spacing if known BB
    layer_d = _BB_LAYER_D.get(node_bb, stk.layer_spacing)
    if linkage == "imide":
        layer_d = 3.48

    pref = predict_preferred_stacking(a_cif, a_cif, n_arom, linkage,
                                       node_bb=node_bb, linker_bb=linker_bb)
    if not stacking:
        stacking = max(pref, key=pref.get)

    # ── All properties ────────────────────────────────────────────────────
    from analysis.property_predictor import (
        estimate_band_gap, estimate_gas_uptake,
        estimate_mechanical_properties, predict_stability
    )
    from analysis.pxrd_simulator import simulate_pxrd
    from decoder.validity_checker import synthesizability_score
    from models.synthesis_condition_predictor import SynthesisConditionPredictor
    from decoder.reticular_decoder import COFSpec, BB_LIBRARY

    vf  = geo["void_fraction"]
    pld = geo["pore_limiting_diameter"]
    lcd = geo["largest_cavity_diameter"]
    bet = geo["bet_surface_area"]
    den = geo.get("density_g_cm3", 0.6)

    bg, _   = estimate_band_gap(linkage, node_bb, linker_bb, vf, n_arom)
    co2     = estimate_gas_uptake("CO2",  298, 1.0,   vf, pld, bet, linkage)
    ch4     = estimate_gas_uptake("CH4",  298, 65.0,  vf, pld, bet, linkage)
    h2      = estimate_gas_uptake("H2",   77,  100.0, vf, pld, bet, linkage)
    mech    = estimate_mechanical_properties(linkage, layer_d, vf)
    stab    = predict_stability(linkage, node_bb, linker_bb)

    crystal_sys = "hexagonal" if abs(gamma - 120) < 8 else "tetragonal"
    pxrd    = simulate_pxrd(
        elements, cg.frac_coords, a_cif, a_cif, c_cif, 90.0, 90.0, gamma,
        crystal_system=crystal_sys, hkl_max=5, fwhm=0.25,
    )

    nf2, lf2 = BB_LIBRARY["conn_groups"].get(linkage, ("NH2", "CHO"))
    spec      = COFSpec(linkage, topology, stacking, node_bb, linker_bb, nf2, lf2)
    synth_s   = synthesizability_score(linkage, node_bb, linker_bb, topology)

    try:
        pred      = SynthesisConditionPredictor.from_schema()
        prior     = pred.get_prior(spec, n_repeats=3)
        solvent   = prior.solvent_candidates[0] if prior.solvent_candidates else "—"
        catalyst  = prior.catalyst_candidates[0] if prior.catalyst_candidates else "—"
        temp_str  = (f"{prior.temperature_range[0]:.0f}–"
                     f"{prior.temperature_range[1]:.0f} °C")
    except Exception:
        solvent  = "o-DCB/n-BuOH (1:1)"
        catalyst = "6M AcOH (0.1 mL)"
        temp_str = "100–130 °C"

    # Warnings for low-confidence predictions
    if node_ana.match_conf < 0.5:
        warn_list.append(
            f"Node '{node_ana.cif_name}' did not match any known BB well "
            f"(conf={node_ana.match_conf:.0%}). Geometry may be approximate.")
    if link_ana.match_conf < 0.5:
        warn_list.append(
            f"Linker '{link_ana.cif_name}' did not match any known BB well "
            f"(conf={link_ana.match_conf:.0%}). Geometry may be approximate.")
    if node_ana.primary_fg == "unknown" or link_ana.primary_fg == "unknown":
        warn_list.append(
            "Could not detect functional groups from CIF coordinates. "
            "Check that your CIF contains complete atom positions including H atoms.")

    return MonomierCOFResult(
        node=node_ana, linker=link_ana,
        linkage=linkage, topology=topology, stacking=stacking,
        a_angstrom=round(a_cif, 2), c_angstrom=round(c_cif, 2),
        void_fraction=vf, bet_m2g=bet, pld=pld, lcd=lcd, density=den,
        band_gap=round(bg, 2),
        co2_uptake=co2["uptake_mmol_g"], ch4_uptake=ch4["uptake_mmol_g"],
        h2_uptake=h2["uptake_mmol_g"],
        co2_n2_sel=co2.get("co2_n2_selectivity", 30.0),
        thermal_C=stab["thermal_decomp_T_C"],
        water_stab=stab["water_stability"],
        e_inplane=mech["young_modulus_inplane_GPa"],
        stacking_prefs=pref,
        layer_spacing=round(layer_d, 3),
        pxrd_peaks=[
            {"hkl": f"{p.h}{p.k}{p.l}", "d": p.d,
             "two_theta": p.two_theta, "I": p.intensity}
            for p in pxrd.peaks[:10]
        ],
        synth_score=synth_s,
        synth_solvent=solvent,
        synth_catalyst=catalyst,
        synth_temp=temp_str,
        warnings=warn_list,
    )
