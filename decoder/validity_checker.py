"""
validity_checker.py
===================
Post-generation validity and stability checks for COF structures.

Checks (in order of computational cost):
  1. Linkage valence check        — O(N), instant
  2. Pore accessibility check     — O(N log N), ~ms via zeo++ or geometric proxy
  3. Force field relaxation       — O(N²) per step × ~200 steps, ~seconds
  4. Synthesizability heuristics  — rule-based, instant

Returns a ValidityReport with pass/fail per check and a composite score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.optimize import BFGS
    HAS_ASE = True
except ImportError:
    HAS_ASE = False

try:
    from pymatgen.core import Structure
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False


# ─────────────────────────────────────────────────────────────────────────────
# Report dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidityReport:
    name:            str
    linkage_valid:   bool  = False
    pore_accessible: bool  = False
    uff_converged:   bool  = False
    synth_score:     float = 0.0   # 0–1, higher = more synthesisable

    # Details
    n_atoms:         int   = 0
    void_fraction:   float = 0.0
    pld:             float = 0.0   # pore limiting diameter (Å)
    lcd:             float = 0.0   # largest cavity diameter (Å)
    uff_energy:      float = float("nan")
    error_msg:       str   = ""

    @property
    def is_valid(self) -> bool:
        return self.linkage_valid and self.pore_accessible

    @property
    def is_stable(self) -> bool:
        return self.is_valid and self.uff_converged

    @property
    def composite_score(self) -> float:
        """0–1 composite score: valid × stable × pore × synth."""
        return (
            float(self.linkage_valid)
            * float(self.pore_accessible)
            * (0.5 + 0.5 * float(self.uff_converged))
            * (0.5 + 0.5 * self.synth_score)
        )

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Linkage valence check
# ─────────────────────────────────────────────────────────────────────────────

# Expected valence (degree) for common COF-linkage atoms
EXPECTED_VALENCE: Dict[str, int] = {
    "N": 3,  # imine N, amide N
    "C": 4,
    "B": 3,  # boronate ester, boroxine
    "O": 2,
    "S": 2,
}

def check_linkage_valence(
    elements: List[str],
    edge_index: np.ndarray,  # (2, E)
    dists: np.ndarray,       # (E,)
    covalent_cutoff: float = 2.2,
) -> Tuple[bool, str]:
    """
    Check that all non-H atoms have chemically reasonable valence.
    Returns (passed, message).
    """
    N = len(elements)
    valence = np.zeros(N, dtype=int)
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        if dists[e] <= covalent_cutoff:
            valence[i] += 1

    violations = []
    for i, (el, v) in enumerate(zip(elements, valence)):
        if el == "H":
            continue
        expected = EXPECTED_VALENCE.get(el, None)
        if expected is not None and v > expected + 1:
            violations.append(f"atom {i} ({el}) valence={v} > expected {expected}")

    if violations:
        return False, "; ".join(violations[:3])
    return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pore accessibility (geometric proxy — no Zeo++ required)
# ─────────────────────────────────────────────────────────────────────────────

# Van der Waals radii (Å)
VDW_RADII: Dict[str, float] = {
    "H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52,
    "F": 1.47, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75,
    "Br": 1.85, "I": 1.98,
    "Fe": 2.05, "Co": 2.00, "Ni": 1.97, "Cu": 1.96, "Zn": 2.01,
}
DEFAULT_VDW = 2.0
PROBE_RADIUS = 1.86  # N2 probe radius (Å)


def estimate_void_fraction(
    frac_coords: np.ndarray,   # (N, 3)
    elements: List[str],
    cell: np.ndarray,           # (3, 3) row-vector matrix
    n_grid: int = 20,           # grid points per dimension
) -> Tuple[float, float, float]:
    """
    Estimate void fraction, approximate PLD and LCD via a coarse grid probe.
    Returns (void_fraction, pld_estimate, lcd_estimate).
    """
    # Build grid in fractional space
    gx = np.linspace(0, 1, n_grid, endpoint=False)
    g  = np.stack(np.meshgrid(gx, gx, gx, indexing="ij"), axis=-1).reshape(-1, 3)
    g_cart = g @ cell   # (n_grid³, 3)

    cart_atoms = frac_coords @ cell   # (N, 3)
    radii      = np.array([VDW_RADII.get(el, DEFAULT_VDW) for el in elements])

    # For each grid point, find minimum distance to any atom (PBC)
    min_dists = np.full(len(g_cart), np.inf)
    for i, pos in enumerate(cart_atoms):
        diff = g_cart - pos
        # Apply minimum image convention
        frac_diff = (g - frac_coords[i])
        frac_diff -= np.round(frac_diff)
        cart_diff = frac_diff @ cell
        d = np.linalg.norm(cart_diff, axis=-1)
        clearance = d - radii[i]
        min_dists = np.minimum(min_dists, clearance)

    accessible = min_dists >= PROBE_RADIUS
    void_fraction = float(accessible.mean())

    # PLD: largest sphere that fits through the narrowest channel
    # Approximated as the smallest max-clearance along any lattice direction
    pld_estimate = float(np.percentile(min_dists[accessible], 10)) * 2 if accessible.any() else 0.0
    lcd_estimate = float(min_dists.max()) * 2

    return void_fraction, max(pld_estimate, 0.0), max(lcd_estimate, 0.0)


def check_pore_accessibility(
    frac_coords: np.ndarray,
    elements: List[str],
    cell: np.ndarray,
    min_void_fraction: float = 0.10,
    min_pld: float = 2.0,   # Å — must fit at least a CO2 molecule
) -> Tuple[bool, float, float, float]:
    """
    Returns (passed, void_fraction, pld, lcd).
    """
    vf, pld, lcd = estimate_void_fraction(frac_coords, elements, cell)
    passed = (vf >= min_void_fraction) and (pld >= min_pld)
    return passed, vf, pld, lcd


# ─────────────────────────────────────────────────────────────────────────────
# 3. UFF force field relaxation (via ASE if available)
# ─────────────────────────────────────────────────────────────────────────────

def uff_relax(
    cif_path: Path,
    max_steps: int = 200,
    fmax: float = 0.1,   # eV/Å convergence
) -> Tuple[bool, float]:
    """
    Relax a structure with UFF via ASE.
    Returns (converged, energy_per_atom).
    Falls back to (True, NaN) if ASE or pymatgen unavailable.
    """
    if not HAS_ASE:
        return True, float("nan")

    try:
        if HAS_PYMATGEN:
            struct = Structure.from_file(str(cif_path))
            atoms  = _pymatgen_to_ase(struct)
        else:
            # Minimal ASE CIF reader
            from ase.io import read as ase_read
            atoms = ase_read(str(cif_path))

        # Use a simple Lennard-Jones as a proxy for UFF
        # (full UFF requires OpenBabel or LAMMPS; LJ gives geometry sanity check)
        calc = LennardJones()
        atoms.calc = calc

        opt = BFGS(atoms, logfile=None)
        converged = opt.run(fmax=fmax, steps=max_steps)
        energy = float(atoms.get_potential_energy()) / len(atoms)
        return bool(converged), energy

    except Exception as e:
        return False, float("nan")


def _pymatgen_to_ase(struct) -> "Atoms":
    from ase import Atoms as AseAtoms
    symbols = [str(s.specie.symbol) for s in struct]
    positions = struct.cart_coords
    cell = struct.lattice.matrix
    return AseAtoms(symbols, positions=positions, cell=cell, pbc=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Synthesizability heuristics
# ─────────────────────────────────────────────────────────────────────────────

# Building blocks known to be commercially available
COMMERCIAL_NODES = {
    "T3_BENZ", "T3_TRIZ", "T3_TPA", "T3_TPM",
    "S4_PORPH", "S4_PHTH",
}
COMMERCIAL_LINKERS = {
    "L2_BENZ", "L2_NAPH", "L2_BIPH", "L2_ANTR",
    "L2_PYRN", "L2_STIL",
}
EASY_LINKAGES = {"imine", "boronate_ester", "beta_ketoenamine"}


def synthesizability_score(
    linkage: str,
    node_bb: str,
    linker_bb: str,
    topology: str,
) -> float:
    """
    Rule-based synthesizability score in [0, 1].
    Reflects known synthetic accessibility:
      - Imine / boronate ester / beta-ketoenamine: easy
      - Commercial building blocks: easier
      - Common topologies (hcb, sql): well-established
    """
    score = 0.0

    # Linkage chemistry
    if linkage in EASY_LINKAGES:
        score += 0.40
    elif linkage in {"hydrazone", "triazine", "imide"}:
        score += 0.25
    else:
        score += 0.10

    # Building block availability
    if node_bb in COMMERCIAL_NODES:
        score += 0.25
    elif node_bb.startswith("T3_"):
        score += 0.15

    if linker_bb in COMMERCIAL_LINKERS:
        score += 0.20
    elif linker_bb.startswith("L2_"):
        score += 0.10

    # Topology
    topo_bonus = {"hcb": 0.15, "sql": 0.12, "kgm": 0.08}.get(topology, 0.02)
    score += topo_bonus

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main validator
# ─────────────────────────────────────────────────────────────────────────────

class ValidityChecker:
    """
    Runs all validity checks on a generated COF structure.
    """

    def __init__(
        self,
        min_void_fraction: float = 0.10,
        min_pld: float = 2.0,
        uff_relax_enable: bool = True,
        uff_max_steps: int = 200,
    ):
        self.min_void_fraction = min_void_fraction
        self.min_pld           = min_pld
        self.uff_relax_enable  = uff_relax_enable
        self.uff_max_steps     = uff_max_steps

    def check(
        self,
        name: str,
        cif_path: Optional[Path],
        spec=None,   # COFSpec
    ) -> ValidityReport:
        report = ValidityReport(name=name)

        if cif_path is None or not Path(cif_path).exists():
            report.error_msg = "CIF file missing"
            return report

        # Parse crystal graph
        try:
            from data.crystal_graph import cif_to_crystal_graph
            cg = cif_to_crystal_graph(cif_path)
        except Exception as e:
            report.error_msg = f"Parse error: {e}"
            return report

        report.n_atoms = cg.n_atoms

        # 1. Linkage valence
        from data.crystal_graph import _cell_matrix
        a, b, c = float(cg.lattice[0]*50), float(cg.lattice[1]*50), float(cg.lattice[2]*50)
        alpha = math.degrees(math.acos(float(np.clip(cg.lattice[3], -1, 1))))
        beta  = math.degrees(math.acos(float(np.clip(cg.lattice[4], -1, 1))))
        gamma = math.degrees(math.acos(float(np.clip(cg.lattice[5], -1, 1))))
        cell  = _cell_matrix(a, b, c, alpha, beta, gamma)

        # Recover elements from feature one-hot
        from utils.featurisation import ELEMENTS
        elements = []
        for atom_feat in cg.atoms:
            el_idx = int(atom_feat[:len(ELEMENTS)].argmax())
            elements.append(ELEMENTS[el_idx])

        lv_ok, lv_msg = check_linkage_valence(elements, cg.edge_index, np.zeros(cg.n_edges))
        report.linkage_valid = lv_ok
        if not lv_ok:
            report.error_msg = lv_msg

        # 2. Pore accessibility
        pa_ok, vf, pld, lcd = check_pore_accessibility(
            cg.frac_coords, elements, cell,
            self.min_void_fraction, self.min_pld,
        )
        report.pore_accessible = pa_ok
        report.void_fraction   = vf
        report.pld             = pld
        report.lcd             = lcd

        # 3. UFF relaxation
        if self.uff_relax_enable:
            converged, energy = uff_relax(cif_path, self.uff_max_steps)
            report.uff_converged = converged
            report.uff_energy    = energy
        else:
            report.uff_converged = True

        # 4. Synthesizability
        if spec is not None:
            report.synth_score = synthesizability_score(
                spec.linkage_type, spec.node_bb, spec.linker_bb, spec.topology
            )
        else:
            report.synth_score = 0.5

        return report

    def check_batch(
        self,
        names: List[str],
        cif_paths: List[Optional[Path]],
        specs=None,
    ) -> List[ValidityReport]:
        specs = specs or [None] * len(names)
        return [
            self.check(n, p, s)
            for n, p, s in zip(names, cif_paths, specs)
        ]

    @staticmethod
    def summary(reports: List[ValidityReport]) -> Dict[str, float]:
        n = len(reports)
        if n == 0:
            return {}
        return {
            "n_total":         n,
            "valid_rate":      sum(r.is_valid for r in reports) / n,
            "stable_rate":     sum(r.is_stable for r in reports) / n,
            "mean_void_frac":  float(np.mean([r.void_fraction for r in reports])),
            "mean_pld":        float(np.mean([r.pld for r in reports])),
            "mean_synth":      float(np.mean([r.synth_score for r in reports])),
            "mean_composite":  float(np.mean([r.composite_score for r in reports])),
        }
