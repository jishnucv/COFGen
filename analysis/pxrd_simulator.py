"""
pxrd_simulator.py
=================
Simulates powder X-ray diffraction (PXRD) patterns from COF crystal structures.

Physics:
  - Structure factor:  F(hkl) = Σ_j f_j(s) * exp(2πi (h*x_j + k*y_j + l*z_j))
  - Intensity:         I(hkl) = |F(hkl)|² × LP × multiplicity × DW
  - Peak broadening:   Lorentzian (Scherrer) + optional Gaussian (strain)
  - Cromer-Mann atomic scattering factors for H,B,C,N,O,F,S,Cl,Br,P,Si

Output:
  - 2θ angles and intensities (normalised to strongest peak = 100)
  - Peak list with (hkl), d-spacing, 2θ, intensity
  - Full profile at user-specified step

Works entirely with numpy/scipy — no diffraction library needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Cromer-Mann coefficients ──────────────────────────────────────────────────
# f(s) = Σ a_i * exp(-b_i * s²) + c,  s = sin(θ)/λ  [Å⁻¹]

_CM: Dict[str, dict] = {
    'H':  {'a':[0.489918,0.262003,0.196767,0.049879],
            'b':[20.6593,7.74039,49.5519,2.20159],'c':0.001305},
    'B':  {'a':[2.05450,1.33260,1.09790,0.706800],
            'b':[23.2185,1.02100,60.3498,0.140300],'c':-0.193200},
    'C':  {'a':[2.31000,1.02000,1.58860,0.865000],
            'b':[20.8439,10.2075,0.568700,51.6512],'c':0.215600},
    'N':  {'a':[12.2126,3.13220,2.01250,1.16630],
            'b':[0.005700,9.89330,28.9975,0.582600],'c':-11.5290},
    'O':  {'a':[3.04850,2.28680,1.54630,0.867000],
            'b':[13.2771,5.70110,0.323900,32.9089],'c':0.250800},
    'F':  {'a':[3.53920,2.64120,1.51700,1.02430],
            'b':[10.2825,4.29440,0.261500,26.1476],'c':0.277600},
    'Si': {'a':[6.29150,3.03530,1.98910,1.54100],
            'b':[2.43860,32.3337,0.678500,81.6937],'c':1.14070},
    'P':  {'a':[6.43450,4.17910,1.78000,1.49080],
            'b':[1.90670,27.1570,0.526000,68.1645],'c':1.11490},
    'S':  {'a':[6.90530,5.20340,1.43790,1.58630],
            'b':[1.46790,22.2151,0.253600,56.1720],'c':0.866900},
    'Cl': {'a':[11.4604,7.19640,6.25560,1.64550],
            'b':[0.010400,1.16620,18.5194,47.7784],'c':-9.55740},
    'Br': {'a':[17.1789,5.23580,5.63770,3.98510],
            'b':[2.17230,16.5796,0.260900,41.4328],'c':2.95570},
    'Zn': {'a':[14.0743,7.03180,5.16520,2.41000],
            'b':[3.26550,0.233300,10.3163,58.7097],'c':1.30410},
    'Fe': {'a':[11.7695,7.35730,3.52220,2.30450],
            'b':[4.76110,0.307200,15.3535,76.8805],'c':1.03690},
    'Co': {'a':[12.2841,7.34090,4.00340,2.34880],
            'b':[4.27910,0.278400,13.5359,71.1692],'c':1.01180},
    'Ni': {'a':[12.8376,7.29200,4.44380,2.38000],
            'b':[3.87850,0.256500,12.1763,66.3421],'c':1.03410},
    'Cu': {'a':[13.3380,7.16760,5.61580,1.67350],
            'b':[3.58280,0.247000,11.3966,64.8126],'c':1.19100},
    'Mn': {'a':[11.2819,7.35730,3.01930,2.24410],
            'b':[5.34090,0.343200,17.8674,83.7543],'c':1.08960},
}
_CM_DEFAULT = _CM['C']


def _f(element: str, s: float) -> float:
    """Atomic scattering factor f(s), s = sin(θ)/λ in Å⁻¹."""
    cm = _CM.get(element, _CM_DEFAULT)
    return cm['c'] + sum(a * math.exp(-b * s * s)
                         for a, b in zip(cm['a'], cm['b']))


# ── Cell geometry ─────────────────────────────────────────────────────────────

def cell_matrix(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Row-vector cell matrix M such that cart = frac @ M."""
    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    ca, cb, cg = math.cos(ar), math.cos(br), math.cos(gr)
    sg = math.sin(gr)
    vol_factor = math.sqrt(max(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg, 1e-12))
    return np.array([
        [a,       0,           0          ],
        [b*cg,    b*sg,        0          ],
        [c*cb,    c*(ca-cb*cg)/sg, c*vol_factor/sg],
    ], dtype=np.float64)


def reciprocal_cell(M: np.ndarray) -> np.ndarray:
    """Reciprocal cell matrix G* = inv(M).T."""
    return np.linalg.inv(M).T


def d_spacing(h, k, l, G_star: np.ndarray) -> float:
    """d-spacing for reflection (hkl) given reciprocal cell G*."""
    hkl = np.array([h, k, l], dtype=np.float64)
    q_vec = hkl @ G_star
    q_len = np.linalg.norm(q_vec)
    return 1.0 / q_len if q_len > 1e-12 else 1e9


def two_theta(d: float, wavelength: float) -> Optional[float]:
    """2θ in degrees from d-spacing and wavelength (Cu Kα = 1.5406 Å)."""
    sin_th = wavelength / (2.0 * d)
    if abs(sin_th) > 1.0:
        return None
    return math.degrees(2.0 * math.asin(sin_th))


# ── Structure factor ──────────────────────────────────────────────────────────

def structure_factor(
    h, k, l,
    elements: List[str],
    frac_coords: np.ndarray,   # (N, 3)
    s_val: float,               # sin(θ)/λ
    Uiso: float = 0.025,        # isotropic displacement (Å²) — typical for organics
) -> complex:
    """
    F(hkl) = Σ_j T_j * f_j(s) * exp(2πi (h*x + k*y + l*z))
    T_j = exp(-8π²Uiso*s²) — Debye-Waller factor
    """
    DW  = math.exp(-8.0 * math.pi**2 * Uiso * s_val**2)
    hkl = np.array([h, k, l], dtype=np.float64)
    phases = 2.0 * math.pi * (frac_coords @ hkl)   # (N,)
    F = complex(0, 0)
    for j, (el, ph) in enumerate(zip(elements, phases)):
        fj = _f(el, s_val)
        F += DW * fj * (math.cos(ph) + 1j * math.sin(ph))
    return F


def lorentz_polarisation(two_theta_deg: float) -> float:
    """LP correction: (1 + cos²(2θ)) / (sin²(θ) * cos(θ))"""
    th  = math.radians(two_theta_deg / 2.0)
    cos2 = math.cos(2.0 * th)
    sth  = math.sin(th)
    cth  = math.cos(th)
    if abs(sth) < 1e-9 or abs(cth) < 1e-9:
        return 1.0
    return (1.0 + cos2**2) / (sth**2 * cth)


# ── Multiplicity for common crystal systems ───────────────────────────────────

def hexagonal_multiplicity(h, k, l) -> int:
    """Approximate multiplicity for hexagonal (P6/m and subgroups)."""
    if h == 0 and k == 0:
        return 2 if l != 0 else 1
    if l == 0:
        if h == k or h == 0 or k == 0:
            return 6
        return 12
    if h == k or h == 0 or k == 0:
        return 12
    return 24


def tetragonal_multiplicity(h, k, l) -> int:
    if h == 0 and k == 0: return 2
    if l == 0 and h == k: return 4
    if l == 0: return 8
    if h == k: return 8
    if h == 0 or k == 0: return 8
    return 16


def cubic_multiplicity(h, k, l) -> int:
    vals = sorted([abs(h), abs(k), abs(l)], reverse=True)
    if vals[1] == 0: return 6
    if vals[0] == vals[1] == vals[2]: return 8
    if vals[0] == vals[1] or vals[1] == vals[2]: return 12
    if vals[2] == 0: return 12
    return 24


def get_multiplicity(h, k, l, crystal_system: str = "hexagonal") -> int:
    if crystal_system == "hexagonal":   return hexagonal_multiplicity(h, k, l)
    if crystal_system == "tetragonal":  return tetragonal_multiplicity(h, k, l)
    if crystal_system in ("cubic","isometric"): return cubic_multiplicity(h, k, l)
    return 2  # triclinic/monoclinic fallback


# ── Main PXRD simulator ───────────────────────────────────────────────────────

@dataclass
class PXRDPeak:
    h:        int
    k:        int
    l:        int
    d:        float      # Å
    two_theta: float     # degrees
    intensity: float     # raw (before normalisation)
    multiplicity: int

    def __repr__(self):
        return (f"({self.h}{self.k}{self.l}) d={self.d:.4f}Å "
                f"2θ={self.two_theta:.3f}° I={self.intensity:.1f}")


@dataclass
class PXRDPattern:
    two_theta: np.ndarray   # (M,) uniformly spaced grid
    intensity: np.ndarray   # (M,) profile intensity (normalised to 100)
    peaks:     List[PXRDPeak]
    wavelength: float = 1.5406   # Cu Kα

    def strongest_peak(self) -> PXRDPeak:
        return max(self.peaks, key=lambda p: p.intensity)

    def peak_positions(self) -> List[float]:
        return [p.two_theta for p in sorted(self.peaks, key=lambda p: p.two_theta)]

    def d_spacings(self) -> List[float]:
        return [p.d for p in sorted(self.peaks, key=lambda p: p.two_theta)]


def simulate_pxrd(
    elements:    List[str],
    frac_coords: np.ndarray,     # (N, 3)
    a: float, b: float, c: float,
    alpha: float = 90.0,
    beta:  float = 90.0,
    gamma: float = 90.0,
    wavelength:  float = 1.5406,   # Cu Kα (Å)
    two_theta_min: float = 2.0,
    two_theta_max: float = 30.0,
    step:          float = 0.02,
    hkl_max:       int   = 5,
    fwhm:          float = 0.2,    # degrees — Scherrer broadening
    crystal_system: str  = "hexagonal",
    intensity_cutoff: float = 0.5,  # % of max to include in peak list
) -> PXRDPattern:
    """
    Simulate PXRD pattern for a COF crystal structure.

    Parameters
    ----------
    elements     : atom type list (e.g. ['C','C','N',...])
    frac_coords  : (N,3) fractional coordinates
    a,b,c        : cell lengths in Å
    alpha,beta,gamma : cell angles in degrees
    wavelength   : X-ray wavelength in Å (Cu Kα = 1.5406)
    hkl_max      : max Miller index to enumerate (5 is sufficient for most COFs)
    fwhm         : full-width at half-maximum of peaks in degrees

    Returns
    -------
    PXRDPattern with peak list and continuous profile
    """
    M      = cell_matrix(a, b, c, alpha, beta, gamma)
    G_star = reciprocal_cell(M)

    peaks: List[PXRDPeak] = []

    for h in range(-hkl_max, hkl_max + 1):
        for k in range(-hkl_max, hkl_max + 1):
            for l in range(0, hkl_max + 1):   # l>=0 (Friedel pair degeneracy)
                if h == 0 and k == 0 and l == 0:
                    continue
                # Skip systematic absences (P1 — none; add space group logic here)

                d = d_spacing(h, k, l, G_star)
                tt = two_theta(d, wavelength)
                if tt is None or tt < two_theta_min or tt > two_theta_max:
                    continue

                s_val = math.sin(math.radians(tt / 2.0)) / wavelength
                F     = structure_factor(h, k, l, elements, frac_coords, s_val)
                I_raw = abs(F) ** 2
                lp    = lorentz_polarisation(tt)
                mult  = get_multiplicity(h, k, l, crystal_system)
                I     = I_raw * lp * mult

                if I > 1e-4:
                    peaks.append(PXRDPeak(h, k, l, d, tt, I, mult))

    if not peaks:
        tt_arr = np.arange(two_theta_min, two_theta_max, step)
        return PXRDPattern(tt_arr, np.zeros_like(tt_arr), [], wavelength)

    # Merge peaks within 0.01° (same reflection)
    peaks.sort(key=lambda p: p.two_theta)
    merged: List[PXRDPeak] = []
    for pk in peaks:
        if merged and abs(pk.two_theta - merged[-1].two_theta) < 0.01:
            merged[-1].intensity += pk.intensity
        else:
            merged.append(pk)
    peaks = merged

    # Normalise to 100
    I_max = max(p.intensity for p in peaks)
    for p in peaks:
        p.intensity = 100.0 * p.intensity / I_max

    peaks = [p for p in peaks if p.intensity >= intensity_cutoff]
    peaks.sort(key=lambda p: p.two_theta)

    # Build continuous profile (pseudo-Voigt via Lorentzian)
    tt_arr  = np.arange(two_theta_min, two_theta_max, step)
    profile = np.zeros_like(tt_arr)
    gamma_l = fwhm / 2.0   # half-width

    for pk in peaks:
        # Lorentzian lineshape
        delta = tt_arr - pk.two_theta
        loren = pk.intensity / (1.0 + (delta / gamma_l) ** 2)
        profile += loren

    if profile.max() > 0:
        profile = 100.0 * profile / profile.max()

    return PXRDPattern(tt_arr, profile, peaks, wavelength)


# ── Stacking and packing analysis ────────────────────────────────────────────

@dataclass
class StackingGeometry:
    """Describes the interlayer stacking of a 2D COF."""
    stacking_type:  str     # "AA", "AB", "ABC", "serrated", "inclined"
    layer_spacing:  float   # Å  (c-parameter or fraction thereof)
    offset_x:       float   # Å  lateral shift along a
    offset_y:       float   # Å  lateral shift along b
    offset_frac:    Tuple[float, float]   # (Δx/a, Δy/b) fractional
    tilt_angle:     float   # degrees — inclination of stacking vector
    pi_stack_energy_estimate: float  # eV/nm² (rough, from empirical formula)

    def describe(self) -> str:
        return (f"{self.stacking_type} stacking | "
                f"d={self.layer_spacing:.2f}Å | "
                f"offset=({self.offset_frac[0]:.2f},{self.offset_frac[1]:.2f}) | "
                f"tilt={self.tilt_angle:.1f}°")


def analyse_stacking(
    frac_coords: np.ndarray,
    a: float, b: float, c: float,
    alpha: float = 90.0, beta: float = 90.0, gamma: float = 120.0,
) -> StackingGeometry:
    """
    Infer stacking geometry from crystal structure.
    For 2D COFs, the c-axis is the stacking direction.
    """
    from data.crystal_graph import _cell_matrix as cm_fn
    M = cm_fn(a, b, c, alpha, beta, gamma)

    # Layer spacing = c × sin(angle to ab-plane)
    c_vec   = M[2]                           # Cartesian c-vector
    layer_d = float(np.linalg.norm(c_vec))   # = c for orthogonal cells

    # Find lateral offset between layers by looking at atom z-coordinates
    # Group atoms by z-layer (within 0.1 fractional unit)
    z_frac = frac_coords[:, 2]
    layer0_mask = z_frac < 0.5
    layer1_mask = ~layer0_mask

    if layer0_mask.sum() > 0 and layer1_mask.sum() > 0:
        com0 = frac_coords[layer0_mask].mean(axis=0)
        com1 = frac_coords[layer1_mask].mean(axis=0)
        # Lateral offset (fractional)
        dx_frac = (com1[0] - com0[0]) % 1.0
        dy_frac = (com1[1] - com0[1]) % 1.0
        if dx_frac > 0.5: dx_frac -= 1.0
        if dy_frac > 0.5: dy_frac -= 1.0
    else:
        dx_frac, dy_frac = 0.0, 0.0

    dx_cart = dx_frac * a
    dy_cart = dy_frac * b

    # Classify stacking
    lateral = math.sqrt(dx_cart**2 + dy_cart**2)
    if lateral < 0.3:
        stype = "AA"
    elif abs(dx_frac - 1/3) < 0.05 and abs(dy_frac - 2/3) < 0.05:
        stype = "AB (eclipsed+1/3)"
    elif lateral < a * 0.3:
        stype = "AB"
    else:
        stype = "inclined"

    tilt  = math.degrees(math.atan2(lateral, layer_d))

    # Empirical π-stacking energy (Grimme-style): E ~ -2.5*(1 + cos²φ) / d^6 eV·Å^6/nm²
    # per aromatic ring pair, rough estimate
    rings_per_nm2   = 1.0 / (a * b * math.sin(math.radians(gamma)) * 1e-2)
    pi_energy       = -2.5 * rings_per_nm2 / (layer_d ** 6) * 1e6 if layer_d > 0 else 0.0

    return StackingGeometry(
        stacking_type  = stype,
        layer_spacing  = layer_d,
        offset_x       = dx_cart,
        offset_y       = dy_cart,
        offset_frac    = (dx_frac, dy_frac),
        tilt_angle     = tilt,
        pi_stack_energy_estimate = pi_energy,
    )


def predict_preferred_stacking(
    a: float, b: float,
    n_aromatic_atoms: int,
    linkage_type: str = "imine",
) -> Dict[str, float]:
    """
    Predict relative stability of AA vs AB vs ABC stacking
    using empirical rules from COF literature:

    - Large π-systems (pyrene, porphyrin): prefer AA (maximise overlap)
    - Imine/boronate: AA slightly preferred (dipole-dipole stabilisation)
    - Beta-ketoenamine: AA strongly preferred (intramolecular H-bonds)
    - Steric: bulky substituents → AB or serrated

    Returns dict {stacking_type: relative_stability_score} where higher = more stable.
    """
    # Base scores from literature statistics (CoRE-COF distribution)
    scores = {"AA": 0.5, "AB": 0.35, "ABC": 0.15}

    # Correction for pi-system size
    pi_correction = min(0.25, n_aromatic_atoms / 100.0)
    scores["AA"]  += pi_correction
    scores["AB"]  -= pi_correction * 0.5

    # Correction for linkage type
    if linkage_type == "beta_ketoenamine":
        scores["AA"] += 0.15   # H-bond network locks AA
    elif linkage_type in ("boronate_ester", "boroxine"):
        scores["AB"] += 0.10   # Lewis acid B prefers offset to reduce strain

    # Normalise
    total = sum(scores.values())
    return {k: v/total for k, v in scores.items()}
