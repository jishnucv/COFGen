#!/usr/bin/env python3
"""
cofgen_tool.py
==============
COFGen: Complete COF Analysis & Property Prediction Tool

Usage
-----
  python cofgen_tool.py predict --cif structure.cif
  python cofgen_tool.py predict --node T3_BENZ --linker L2_PYRN --linkage imine
  python cofgen_tool.py reverse --cif structure.cif
  python cofgen_tool.py generate --co2 4.0 --bet 2500 --linkage imine -n 20
  python cofgen_tool.py pxrd     --cif structure.cif --plot
  python cofgen_tool.py stacking --cif structure.cif
  python cofgen_tool.py synthesis --node T3_BENZ --linker L2_PYRN --linkage imine
  python cofgen_tool.py list-bbs

All commands can also write JSON output with --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── Bootstrap path ────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _load_cif(cif_path: Path):
    """Parse a CIF and return (elements, frac_coords, cg, cell_dict)."""
    from data.crystal_graph import cif_to_crystal_graph
    from utils.featurisation import ELEMENTS as ELS
    cg = cif_to_crystal_graph(cif_path, cutoff=4.0)
    n_el = len(ELS)
    elements = [ELS[int(cg.atoms[i][:n_el].argmax())] for i in range(cg.n_atoms)]
    lat = cg.lattice
    a = float(lat[0]) * 50;  b = float(lat[1]) * 50;  c = float(lat[2]) * 50
    alpha = math.degrees(math.acos(float(np.clip(lat[3], -1, 1))))
    beta  = math.degrees(math.acos(float(np.clip(lat[4], -1, 1))))
    gamma = math.degrees(math.acos(float(np.clip(lat[5], -1, 1))))
    cell = {"a":a, "b":b, "c":c, "alpha":alpha, "beta":beta, "gamma":gamma}
    return elements, cg.frac_coords, cg, cell


def _spec_from_args(args) -> "COFSpec":
    """Build a COFSpec from CLI arguments."""
    from decoder.reticular_decoder import COFSpec, BB_LIBRARY
    lk = args.linkage
    nf, lf = BB_LIBRARY["conn_groups"].get(lk, ("NH2","CHO"))
    node   = args.node   if hasattr(args,"node")   else "T3_BENZ"
    linker = args.linker if hasattr(args,"linker") else "L2_BENZ"
    topo   = args.topology if hasattr(args,"topology") and args.topology else "hcb"
    stack  = args.stacking if hasattr(args,"stacking") and args.stacking else "AA"
    return COFSpec(lk, topo, stack, node, linker, nf, lf)


def _print_header(title: str):
    w = 62
    print(f"\n{'═'*w}")
    print(f"  COFGen ▸ {title}")
    print(f"{'═'*w}")


def _write_json(data: dict, path: str):
    Path(path).write_text(json.dumps(data, indent=2, default=str))
    print(f"\n  ✓ JSON saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# COMMAND: predict
# ════════════════════════════════════════════════════════════════════════════

def cmd_predict(args):
    """Full property prediction for a COF (CIF or BB names)."""
    from data.property_labels import compute_geometric_properties
    from analysis.pxrd_simulator import simulate_pxrd, analyse_stacking, predict_preferred_stacking
    from analysis.property_predictor import (
        estimate_band_gap, classify_semiconductor_type,
        estimate_gas_uptake, estimate_mechanical_properties, predict_stability,
    )
    from models.synthesis_condition_predictor import SynthesisConditionPredictor
    from decoder.validity_checker import synthesizability_score

    _print_header("Property Prediction")

    # ── Get structure ──────────────────────────────────────────────────────
    if args.cif:
        cif_path = Path(args.cif)
        from analysis.monomer_reverse_engineer import reverse_engineer_cif
        re_result = reverse_engineer_cif(cif_path)
        node_bb   = re_result.node_bb
        linker_bb = re_result.linker_bb
        linkage   = re_result.linkage_type
        topology  = re_result.topology
        cif_text  = cif_path.read_text()

        elements, frac, cg, cell = _load_cif(cif_path)
        print(f"\n  CIF:      {cif_path.name}")
        print(f"  Detected: {node_bb} + {linker_bb} via {linkage} ({topology})")
        print(f"  Confidence: {re_result.confidence:.0%}")

    else:
        from data.synthetic_cif_generator import generate_hcb_cif, generate_sql_cif
        from decoder.reticular_decoder import BB_LIBRARY

        node_bb   = args.node
        linker_bb = args.linker
        linkage   = args.linkage
        topology  = getattr(args,"topology","hcb") or "hcb"

        # ── Validate BB names and linkage ─────────────────────────────────
        all_nodes   = BB_LIBRARY["T3_nodes"] + BB_LIBRARY["S4_nodes"]
        all_linkers = BB_LIBRARY["L2_linkers"]
        all_linkages = list(BB_LIBRARY["conn_groups"].keys())

        if node_bb not in all_nodes:
            print(f"\n  ✗ Unknown node BB: '{node_bb}'")
            print(f"  Valid nodes: {', '.join(all_nodes)}")
            print(f"\n  Tip: tetratopic porphyrin node is S4_PORPH (not T4_PORPH)")
            return
        if linker_bb not in all_linkers:
            print(f"\n  ✗ Unknown linker BB: '{linker_bb}'")
            print(f"  Valid linkers: {', '.join(all_linkers)}")
            return
        if linkage not in all_linkages:
            print(f"\n  ✗ Unknown linkage: '{linkage}'")
            print(f"  Valid linkages: {', '.join(all_linkages)}")
            return

        # Get func groups from BB_LIBRARY (covers imide + all others)
        nf, lf = BB_LIBRARY["conn_groups"].get(linkage, ("NH2","CHO"))

        with tempfile.NamedTemporaryFile(suffix='.cif', mode='w', delete=False) as f:
            if topology in ("hcb","kgm","hxl") and node_bb.startswith("T3_"):
                f.write(generate_hcb_cif(node_bb, linker_bb, linkage, nf, lf))
            else:
                f.write(generate_sql_cif(node_bb, linker_bb, linkage, nf, lf))
            cif_path = Path(f.name)

        elements, frac, cg, cell = _load_cif(cif_path)
        cif_path.unlink()
        print(f"\n  Structure: {node_bb} + {linker_bb} via {linkage} ({topology})")

    a, b, c = cell["a"], cell["b"], cell["c"]
    alpha, beta_val, gamma = cell["alpha"], cell["beta"], cell["gamma"]

    n_arom = elements.count('C')
    crystal_sys = "hexagonal" if abs(gamma-120)<8 else "tetragonal"

    # ── Geometric properties ───────────────────────────────────────────────
    print("\n  Computing geometric properties...")
    geo = compute_geometric_properties(cg, n_grid=20)

    # ── PXRD ──────────────────────────────────────────────────────────────
    print("  Simulating PXRD...")
    pxrd = simulate_pxrd(
        elements, frac, a, b, c, alpha, cell["beta"], gamma,
        crystal_system=crystal_sys, hkl_max=5, fwhm=0.25,
        two_theta_min=2.0, two_theta_max=35.0,
    )

    # ── Stacking ──────────────────────────────────────────────────────────
    stk  = analyse_stacking(frac, a, b, c, alpha, cell.get("beta", 90.0), gamma,
                             node_bb=node_bb)
    pref = predict_preferred_stacking(a, b, n_arom, linkage,
                                       node_bb=node_bb, linker_bb=linker_bb)
    best_stacking = max(pref, key=pref.get)

    # ── Electronic ────────────────────────────────────────────────────────
    bg, bg_unc = estimate_band_gap(linkage, node_bb, linker_bb,
                                    geo["void_fraction"], n_arom)

    # ── Adsorption ────────────────────────────────────────────────────────
    co2  = estimate_gas_uptake("CO2",  298, 1.0,   geo["void_fraction"],
                                geo["pore_limiting_diameter"],
                                geo["bet_surface_area"], linkage)
    ch4  = estimate_gas_uptake("CH4",  298, 65.0,  geo["void_fraction"],
                                geo["pore_limiting_diameter"],
                                geo["bet_surface_area"], linkage)
    h2   = estimate_gas_uptake("H2",   77,  100.0, geo["void_fraction"],
                                geo["pore_limiting_diameter"],
                                geo["bet_surface_area"], linkage)
    n2   = estimate_gas_uptake("N2",   298, 1.0,   geo["void_fraction"],
                                geo["pore_limiting_diameter"],
                                geo["bet_surface_area"], linkage)

    # ── Mechanical ────────────────────────────────────────────────────────
    mech = estimate_mechanical_properties(linkage, stk.layer_spacing,
                                           geo["void_fraction"])

    # ── Stability ─────────────────────────────────────────────────────────
    stab = predict_stability(linkage, node_bb=node_bb, linker_bb=linker_bb)

    # Imide layer spacing correction: PI-COFs pack tighter due to planar rings
    if linkage == "imide":
        stk = stk.__class__(
            stacking_type  = stk.stacking_type,
            layer_spacing  = 3.48,    # literature: Ben JACS 2009 d=3.45Å, Fang 2015 d=3.50Å
            offset_x       = stk.offset_x,
            offset_y       = stk.offset_y,
            offset_frac    = stk.offset_frac,
            tilt_angle     = stk.tilt_angle,
            pi_stack_energy_estimate = stk.pi_stack_energy_estimate,
        )

    # ── Synthesis ─────────────────────────────────────────────────────────
    from decoder.reticular_decoder import COFSpec, BB_LIBRARY
    nf2, lf2 = BB_LIBRARY["conn_groups"].get(linkage, ("NH2","CHO"))
    spec = COFSpec(linkage, topology, best_stacking, node_bb, linker_bb, nf2, lf2)
    synth_s = synthesizability_score(linkage, node_bb, linker_bb, topology)
    predictor = SynthesisConditionPredictor.from_schema()
    prior     = predictor.get_prior(spec, n_repeats=3)

    # ── Print report ──────────────────────────────────────────────────────
    vf   = geo["void_fraction"]
    pld  = geo["pore_limiting_diameter"]
    lcd  = geo["largest_cavity_diameter"]
    bet  = geo["bet_surface_area"]
    dens = geo.get("density_g_cm3", 0.0)

    W = 62
    def row(label, value, unit=""):
        vstr = f"{value}" if isinstance(value,str) else f"{value}"
        line = f"  {label:<28} {vstr:<20} {unit}"
        print(line[:W])

    print(f"\n{'─'*W}")
    print(f"  ■ Building Blocks")
    print(f"{'─'*W}")
    row("Node",            node_bb)
    row("Linker",          linker_bb)
    row("Linkage",         linkage)
    row("Topology",        topology)
    row("Stacking (predicted)", best_stacking)
    row("  AA prob",       f"{pref.get('AA',0):.0%}")
    row("  AB prob",       f"{pref.get('AB',0):.0%}")
    row("Layer spacing",   f"{stk.layer_spacing:.2f}", "Å")

    print(f"\n{'─'*W}")
    print(f"  ■ Pore Geometry")
    print(f"{'─'*W}")
    row("Void fraction",   f"{vf:.4f}")
    row("PLD",             f"{pld:.2f}", "Å")
    row("LCD",             f"{lcd:.2f}", "Å")
    row("BET surface area",f"{bet:.0f}", "m²/g")
    row("Density",         f"{dens:.4f}", "g/cm³")
    row("Cell  a,b",       f"{a:.2f}, {b:.2f}", "Å")
    row("Cell  c",         f"{c:.2f}", "Å")
    row("γ",               f"{gamma:.1f}", "°")

    print(f"\n{'─'*W}")
    print(f"  ■ PXRD  (Cu Kα, λ=1.5406 Å)")
    print(f"{'─'*W}")
    print(f"  {'(hkl)':<8} {'2θ (°)':<12} {'d (Å)':<12} {'I/I₀':>6}")
    print(f"  {'─'*40}")
    for pk in sorted(pxrd.peaks, key=lambda p: -p.intensity)[:8]:
        print(f"  ({pk.h}{pk.k}{pk.l}){'':<5} {pk.two_theta:<12.3f} "
              f"{pk.d:<12.4f} {pk.intensity:>6.1f}")

    print(f"\n{'─'*W}")
    print(f"  ■ Electronics")
    print(f"{'─'*W}")
    row("Band gap",        f"{bg:.2f} ± {bg_unc:.2f}", "eV")
    row("Type",            classify_semiconductor_type(bg))

    print(f"\n{'─'*W}")
    print(f"  ■ Gas Adsorption")
    print(f"{'─'*W}")
    row("CO₂ @ 298K, 1 bar",   f"{co2['uptake_mmol_g']:.2f}", "mmol/g")
    row("CH₄ @ 298K, 65 bar",  f"{ch4['uptake_mmol_g']:.2f}", "mmol/g")
    row("H₂  @ 77K, 100 bar",  f"{h2['uptake_mmol_g']:.2f}",  "mmol/g")
    row("N₂  @ 298K, 1 bar",   f"{n2['uptake_mmol_g']:.2f}",  "mmol/g")
    row("CO₂/N₂ selectivity",  f"{co2['co2_n2_selectivity']:.1f}","×")

    print(f"\n{'─'*W}")
    print(f"  ■ Mechanical Properties")
    print(f"{'─'*W}")
    row("E (in-plane)",    f"{mech['young_modulus_inplane_GPa']:.1f}",  "GPa")
    row("E (out-of-plane)",f"{mech['young_modulus_outplane_GPa']:.1f}", "GPa")
    row("Bulk modulus",    f"{mech['bulk_modulus_estimate_GPa']:.1f}",  "GPa")
    row("Interlayer shear",f"{mech['shear_modulus_interlayer_GPa']:.1f}","GPa")

    print(f"\n{'─'*W}")
    print(f"  ■ Stability")
    print(f"{'─'*W}")
    row("Thermal decomp.", f"{stab['thermal_decomp_T_C']:.0f}","°C")
    row("Water stability", stab["water_stability"])
    row("Acid stability",  stab["acid_stability"])
    row("Base stability",  stab["base_stability"])
    row("Hydrolysis t½",   f"{stab['hydrolysis_t12_hours']:.0f}","h")
    row("Stability score", f"{stab['stability_score']:.2f}","/1.00")

    print(f"\n{'─'*W}")
    print(f"  ■ Synthesis Conditions  (Chen et al. JACS 2026 workflow)")
    print(f"{'─'*W}")
    row("Synthesizability", f"{synth_s:.2f}","/1.00")
    row("Recommended solvent",  prior.solvent_candidates[0][:35] if prior.solvent_candidates else "—")
    row("Recommended catalyst", prior.catalyst_candidates[0][:35] if prior.catalyst_candidates else "—")
    row("Temperature",     f"{prior.temperature_range[0]:.0f}–{prior.temperature_range[1]:.0f}", "°C")
    row("Reaction time",   f"{prior.time_range[0]:.0f}–{prior.time_range[1]:.0f}", "days")
    row("Stoichiometry",   f"1:{prior.stoichiometry_range[0]:.1f}–{prior.stoichiometry_range[1]:.1f}")
    print(f"{'═'*W}\n")

    if args.json:
        _write_json({
            "node_bb": node_bb, "linker_bb": linker_bb,
            "linkage": linkage, "topology": topology,
            "geometry": geo,
            "stacking": {"type": best_stacking, **pref,
                          "layer_spacing_A": stk.layer_spacing},
            "electronics": {"band_gap_eV": bg, "uncertainty_eV": bg_unc,
                             "type": classify_semiconductor_type(bg)},
            "adsorption": {"CO2_1bar": co2, "CH4_65bar": ch4,
                            "H2_100bar": h2, "N2_1bar": n2},
            "mechanical": mech,
            "stability": stab,
            "pxrd_peaks": [{"hkl":f"{p.h}{p.k}{p.l}","d":p.d,
                             "two_theta":p.two_theta,"I":p.intensity}
                            for p in pxrd.peaks],
            "synthesis": {"score": synth_s, "prior": {
                "solvent": prior.solvent_candidates,
                "catalyst": prior.catalyst_candidates,
                "temp_C": prior.temperature_range,
                "time_days": prior.time_range,
            }},
        }, args.json)


# ════════════════════════════════════════════════════════════════════════════
# COMMAND: reverse
# ════════════════════════════════════════════════════════════════════════════

def cmd_reverse(args):
    """Reverse-engineer monomers from a CIF file."""
    from analysis.monomer_reverse_engineer import reverse_engineer_cif

    _print_header("Monomer Reverse Engineering")

    cif_path = Path(args.cif)
    if not cif_path.exists():
        print(f"  ✗ File not found: {cif_path}"); return

    print(f"\n  Analysing: {cif_path.name}")
    result = reverse_engineer_cif(cif_path, verbose=True)

    print()
    print(result.summary())

    if args.json:
        _write_json({
            "node_bb": result.node_bb, "linker_bb": result.linker_bb,
            "node_conf": result.node_bb_conf, "linker_conf": result.linker_bb_conf,
            "linkage_type": result.linkage_type, "topology": result.topology,
            "node_formula": result.node_formula, "linker_formula": result.linker_formula,
            "node_smiles": result.node_smiles, "linker_smiles": result.linker_smiles,
            "method": result.method, "confidence": result.confidence,
        }, args.json)


# ════════════════════════════════════════════════════════════════════════════
# COMMAND: pxrd
# ════════════════════════════════════════════════════════════════════════════

def cmd_pxrd(args):
    """Simulate PXRD pattern and optionally plot it."""
    from analysis.pxrd_simulator import simulate_pxrd

    _print_header("PXRD Simulation")

    if args.cif:
        elements, frac, cg, cell = _load_cif(Path(args.cif))
        a, b, c = cell["a"], cell["b"], cell["c"]
        alpha, beta_val, gamma = cell["alpha"], cell["beta"], cell["gamma"]
        name = Path(args.cif).stem
    else:
        from data.synthetic_cif_generator import generate_hcb_cif
        lk = args.linkage
        nf, lf = {"imine":("NH2","CHO")}.get(lk,("NH2","CHO"))
        cif = generate_hcb_cif(args.node, args.linker, lk, nf, lf)
        with tempfile.NamedTemporaryFile(suffix='.cif',mode='w',delete=False) as f:
            f.write(cif); tmp=Path(f.name)
        elements, frac, cg, cell = _load_cif(tmp); tmp.unlink()
        a, b, c = cell["a"],cell["b"],cell["c"]
        alpha, beta_val, gamma = cell["alpha"],cell["beta"],cell["gamma"]
        name = f"{args.node}_{args.linker}"

    crystal_sys = "hexagonal" if abs(gamma-120)<8 else "tetragonal"
    wl = getattr(args, "wavelength", 1.5406) or 1.5406
    pxrd = simulate_pxrd(
        elements, frac, a, b, c, alpha, beta, gamma,
        wavelength=wl, crystal_system=crystal_sys,
        hkl_max=5, fwhm=0.25,
        two_theta_min=2.0, two_theta_max=35.0,
    )

    print(f"\n  Structure: {name}")
    print(f"  Cell: a={a:.2f} b={b:.2f} c={c:.2f} Å,  γ={gamma:.1f}°")
    print(f"  Wavelength: {wl:.4f} Å  ({len(pxrd.peaks)} peaks)")
    print(f"\n  {'(hkl)':<8} {'2θ (°)':<12} {'d (Å)':<12} {'I/I₀':>8}")
    print(f"  {'─'*44}")
    for pk in sorted(pxrd.peaks, key=lambda p: p.two_theta):
        bar = "█" * int(pk.intensity/10) + "░"*(10-int(pk.intensity/10))
        print(f"  ({pk.h}{pk.k}{pk.l}){'':<5} {pk.two_theta:<12.3f} "
              f"{pk.d:<12.4f} {pk.intensity:>6.1f}  {bar}")

    if getattr(args, "plot", False):
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pxrd.two_theta, pxrd.intensity, "b-", lw=1.2, label="Simulated")
            for pk in pxrd.peaks:
                if pk.intensity > 5:
                    ax.annotate(f"({pk.h}{pk.k}{pk.l})",
                                xy=(pk.two_theta, pk.intensity),
                                xytext=(0, 6), textcoords="offset points",
                                ha="center", fontsize=7, color="#333")
            ax.set_xlabel("2θ (degrees)", fontsize=12)
            ax.set_ylabel("Intensity (arb. units)", fontsize=12)
            ax.set_title(f"Simulated PXRD — {name}  (Cu Kα, λ={wl:.4f} Å)")
            ax.set_xlim(2, 35); ax.set_ylim(-5, 110)
            ax.legend(); fig.tight_layout()
            out = getattr(args,"plot_out",None) or f"pxrd_{name}.png"
            plt.savefig(out, dpi=150)
            print(f"\n  ✓ PXRD plot saved → {out}")
            plt.close()
        except Exception as e:
            print(f"\n  ✗ Plot failed: {e}")

    if args.json:
        _write_json({
            "peaks": [{"h":p.h,"k":p.k,"l":p.l,"d":p.d,
                        "two_theta":p.two_theta,"intensity":p.intensity}
                       for p in pxrd.peaks],
            "profile_2theta": pxrd.two_theta.tolist(),
            "profile_intensity": pxrd.intensity.tolist(),
            "wavelength": wl,
        }, args.json)


# ════════════════════════════════════════════════════════════════════════════
# COMMAND: stacking
# ════════════════════════════════════════════════════════════════════════════

def cmd_stacking(args):
    """Analyse and predict stacking geometry."""
    from analysis.pxrd_simulator import analyse_stacking, predict_preferred_stacking

    _print_header("Stacking Analysis & Prediction")

    if args.cif:
        elements, frac, cg, cell = _load_cif(Path(args.cif))
        a, b, c = cell["a"],cell["b"],cell["c"]
        alpha, beta_val, gamma = cell["alpha"],cell["beta"],cell["gamma"]
        lk = cg.properties.get("pcb_node_func","imine") if cg.properties else "imine"
        n_arom = elements.count("C")
    else:
        a=b=22.5; c=3.6; alpha=beta_=90.0; gamma=120.0; lk=args.linkage; n_arom=60
        frac = np.random.rand(84,3)

    stk  = analyse_stacking(frac, a, b, c, alpha, cell.get("beta", 90.0), gamma)
    pref = predict_preferred_stacking(a, b, n_arom, lk)

    print(f"\n  ■ Detected Stacking Geometry")
    print(f"  {'─'*40}")
    print(f"  Type:           {stk.stacking_type}")
    print(f"  Layer spacing:  {stk.layer_spacing:.3f} Å")
    print(f"  Lateral offset: ({stk.offset_frac[0]:.3f}, {stk.offset_frac[1]:.3f})  (fractional)")
    print(f"  Offset (cart):  ({stk.offset_x:.2f}, {stk.offset_y:.2f}) Å")
    print(f"  Tilt angle:     {stk.tilt_angle:.1f}°")
    print(f"  π-stack energy: {stk.pi_stack_energy_estimate:.4f} eV/nm²  (estimate)")

    print(f"\n  ■ Predicted Stacking Preference")
    print(f"  {'─'*40}")
    for stype in ["AA","AB","ABC"]:
        prob = pref.get(stype, 0.0)
        bar  = "█"*int(prob*20) + "░"*(20-int(prob*20))
        marker = " ◀ most stable" if stype == max(pref,key=pref.get) else ""
        print(f"  {stype}:  [{bar}] {prob:.0%}{marker}")

    print(f"\n  ■ Literature Context")
    print(f"  {'─'*40}")
    notes = {
        "AA":  "Eclipsed — maximises π-π overlap, common for imine/boronate COFs",
        "AB":  "Offset — reduces electrostatic repulsion, common in bulky systems",
        "ABC": "Propeller — rare, seen in CTF (triazine) frameworks",
    }
    for s, note in notes.items():
        print(f"  {s}: {note}")

    if args.json:
        _write_json({"detected": stk.__dict__, "predicted_preference": pref}, args.json)


# ════════════════════════════════════════════════════════════════════════════
# COMMAND: synthesis
# ════════════════════════════════════════════════════════════════════════════

def cmd_synthesis(args):
    """Generate synthesis conditions and DoE matrix."""
    from models.synthesis_condition_predictor import (
        SynthesisConditionPredictor, diagnose_failure, FAILURE_CLASS_ACTIONS
    )
    from decoder.reticular_decoder import COFSpec, BB_LIBRARY
    from decoder.validity_checker import synthesizability_score

    _print_header("Synthesis Condition Prediction")

    node   = args.node
    linker = args.linker
    lk     = args.linkage
    topo   = getattr(args,"topology","hcb") or "hcb"
    nf, lf = BB_LIBRARY["conn_groups"].get(lk, ("NH2","CHO"))
    spec   = COFSpec(lk, topo, "AA", node, linker, nf, lf)

    # Load CS-KB
    kb_path = _HERE / "data" / "cs_kb.json"
    if kb_path.exists():
        pred = SynthesisConditionPredictor.from_kb(kb_path)
    else:
        pred = SynthesisConditionPredictor.from_schema()

    prior = pred.get_prior(spec, strategy="stratified", n_repeats=5)
    score = synthesizability_score(lk, node, linker, topo)
    cov   = pred.coverage_score(spec)
    adj   = pred.adjusted_synth_score(spec, score)

    print(f"\n  Target: {node} + {linker} via {lk} ({topo})")
    print(f"\n  ■ Synthesizability")
    print(f"  {'─'*40}")
    bar = "█"*int(adj*20) + "░"*(20-int(adj*20))
    print(f"  Score:    [{bar}] {adj:.0%}")
    print(f"  Rule-based:     {score:.2f}/1.00")
    print(f"  CS-KB coverage: {cov:.2f}/1.00  ({prior.n_precedents_used} precedents found)")

    print(f"\n  ■ Recommended Synthesis Prior")
    print(f"  {'─'*40}")
    print(f"  Solvents (ranked by precedent):")
    for i, s in enumerate(prior.solvent_candidates[:4]):
        print(f"    {i+1}. {s}")
    print(f"  Catalysts/acids:")
    for i, c in enumerate(prior.catalyst_candidates[:3]):
        print(f"    {i+1}. {c}")
    print(f"  Temperature:  {prior.temperature_range[0]:.0f}–{prior.temperature_range[1]:.0f} °C")
    print(f"  Time:         {prior.time_range[0]:.0f}–{prior.time_range[1]:.0f} days")
    print(f"  Stoichiometry: node:linker = 1:{prior.stoichiometry_range[0]:.1f}–{prior.stoichiometry_range[1]:.1f}")

    doe = prior.to_doe_matrix(n_experiments=10)
    print(f"\n  ■ Round-1 DoE Matrix (10 experiments)")
    print(f"  {'─'*60}")
    print(f"  {'#':<4} {'Solvent':<30} {'Catalyst':<20} {'T°C':<6}")
    print(f"  {'─'*60}")
    for exp in doe:
        solv = exp["solvent"][:28]
        cat  = exp["catalyst"][:18]
        print(f"  {exp['exp_id']:<4} {solv:<30} {cat:<20} {exp['temperature']:<6.0f}")

    print(f"\n  ■ Failure Taxonomy (Chen et al. JACS 2026)")
    print(f"  {'─'*40}")
    print(f"  Class A — Stalled/no solid:  reduce catalyst, raise temp")
    print(f"  Class B — Solubility trap:   switch to TFA, add non-polar solvent")
    print(f"  Class C — Kinetic trap:      extend time, raise temp to 130°C")
    print(f"  Class D — Near-crystalline:  adjust stoichiometry, re-activate")

    if getattr(args, "observation", None):
        fc = diagnose_failure(args.observation)
        print(f"\n  ■ Diagnosis: '{args.observation[:50]}...'")
        print(f"  → Class {fc[-1]}: {fc}")
        print(f"  Recommended next actions:")
        for action in FAILURE_CLASS_ACTIONS[fc]:
            print(f"    • {action}")

    if args.json:
        _write_json({"spec": str(spec), "synthesizability": adj,
                      "prior": {"solvents": prior.solvent_candidates,
                                 "catalysts": prior.catalyst_candidates,
                                 "temp_C": prior.temperature_range,
                                 "time_days": prior.time_range},
                      "doe": doe}, args.json)


# ════════════════════════════════════════════════════════════════════════════
# COMMAND: generate
# ════════════════════════════════════════════════════════════════════════════

def cmd_generate(args):
    """Generate candidate COF structures using GA or sklearn baseline."""
    from models.mattersim_stability import GABaseline, llm_baseline_generate
    from models.synthesis_condition_predictor import SynthesisConditionPredictor
    from decoder.validity_checker import synthesizability_score
    from data.synthetic_cif_generator import generate_hcb_cif, generate_sql_cif
    from data.crystal_graph import cif_to_crystal_graph
    from data.property_labels import compute_geometric_properties
    from decoder.validity_checker import ValidityReport
    from decoder.reticular_decoder import BB_LIBRARY, COFSpec
    from evaluation.metrics import full_evaluation, print_evaluation

    _print_header("COF Generation")

    n       = getattr(args, "n", 20) or 20
    linkage = getattr(args, "linkage", None)
    co2_t   = getattr(args, "co2", None)
    bet_t   = getattr(args, "bet", None)

    print(f"\n  Generating {n} candidate structures...")
    if linkage: print(f"  Constraint: linkage = {linkage}")
    if co2_t:   print(f"  Target:     CO₂ uptake ≥ {co2_t} mmol/g")
    if bet_t:   print(f"  Target:     BET ≥ {bet_t} m²/g")

    # Generate via GA
    ga = GABaseline(population_size=max(n*4, 200), n_generations=10, seed=42)
    all_specs = ga.run(n_return=n*3)

    # Filter by linkage if specified
    if linkage:
        all_specs = [s for s in all_specs if s.linkage_type == linkage]

    predictor = SynthesisConditionPredictor.from_schema()

    # Validate each candidate
    reports, valid_specs = [], []
    print(f"\n  Validating candidates...")

    for spec in all_specs[:n*2]:
        try:
            nf, lf = BB_LIBRARY["conn_groups"].get(spec.linkage_type, ("NH2","CHO"))
            if spec.topology in ("hcb","kgm","hxl") and spec.node_bb.startswith("T3_"):
                cif_txt = generate_hcb_cif(spec.node_bb, spec.linker_bb,
                                            spec.linkage_type, nf, lf)
            elif spec.node_bb.startswith("S4_"):
                cif_txt = generate_sql_cif(spec.node_bb, spec.linker_bb,
                                            spec.linkage_type, nf, lf)
            else:
                continue

            with tempfile.NamedTemporaryFile(suffix='.cif',mode='w',delete=False) as f:
                f.write(cif_txt); cif_p = Path(f.name)

            cg = cif_to_crystal_graph(cif_p, cutoff=4.0)
            geo = compute_geometric_properties(cg, n_grid=12)
            cif_p.unlink()

            r = ValidityReport(name=spec.to_pycofbuilder_name())
            r.linkage_valid   = True
            r.void_fraction   = geo["void_fraction"]
            r.pld             = geo["pore_limiting_diameter"]
            r.lcd             = geo["largest_cavity_diameter"]
            r.pore_accessible = r.void_fraction > 0.04 and r.pld > 2.0
            r.uff_converged   = True
            r.n_atoms         = cg.n_atoms
            r.synth_score     = predictor.adjusted_synth_score(
                spec, synthesizability_score(spec.linkage_type, spec.node_bb,
                                              spec.linker_bb, spec.topology))
            reports.append(r)
            valid_specs.append(spec)
            if len(valid_specs) >= n: break
        except Exception:
            continue

    # Filter by property targets
    ranked = sorted(zip(valid_specs, reports),
                    key=lambda x: x[1].composite_score, reverse=True)

    if co2_t or bet_t:
        from analysis.property_predictor import estimate_gas_uptake, estimate_band_gap
        def passes_filters(spec, r):
            if co2_t:
                co2 = estimate_gas_uptake("CO2",298,1.0,r.void_fraction,r.pld,
                                           r.void_fraction*8500, spec.linkage_type)
                if co2["uptake_mmol_g"] < co2_t: return False
            if bet_t and r.void_fraction * 8500 < bet_t: return False
            return True
        ranked = [(s,r) for s,r in ranked if passes_filters(s,r)]

    print(f"\n  ■ Top Generated Candidates")
    print(f"  {'─'*68}")
    print(f"  {'#':<3} {'Linkage':<18} {'Node':<12} {'Linker':<10} "
          f"{'VF':<7} {'PLD':<7} {'Synth':<7}")
    print(f"  {'─'*68}")

    for i, (spec, r) in enumerate(ranked[:10]):
        print(f"  {i+1:<3} {spec.linkage_type:<18} {spec.node_bb:<12} "
              f"{spec.linker_bb:<10} {r.void_fraction:<7.3f} "
              f"{r.pld:<7.2f} {r.synth_score:<7.2f}")

    if ranked:
        top_spec, top_r = ranked[0]
        prior = predictor.get_prior(top_spec)
        print(f"\n  ■ Top Candidate Synthesis Conditions")
        print(f"  {'─'*40}")
        print(f"  {top_spec.node_bb} + {top_spec.linker_bb} via {top_spec.linkage_type}")
        print(f"  Solvent:    {prior.solvent_candidates[0] if prior.solvent_candidates else '—'}")
        print(f"  Catalyst:   {prior.catalyst_candidates[0] if prior.catalyst_candidates else '—'}")
        print(f"  Conditions: {prior.temperature_range[0]:.0f}°C, {prior.time_range[0]:.0f}–{prior.time_range[1]:.0f} days")

    if args.json:
        _write_json({"candidates": [
            {"rank":i+1, "node":s.node_bb, "linker":s.linker_bb,
             "linkage":s.linkage_type, "topology":s.topology,
             "void_fraction":r.void_fraction, "pld":r.pld,
             "synth_score":r.synth_score, "composite":r.composite_score}
            for i,(s,r) in enumerate(ranked[:20])
        ]}, args.json)


# ════════════════════════════════════════════════════════════════════════════
# COMMAND: list-bbs
# ════════════════════════════════════════════════════════════════════════════

def cmd_list_bbs(args):
    """List all available building blocks."""
    from decoder.reticular_decoder import BB_LIBRARY
    from analysis.monomer_reverse_engineer import BB_SMILES, BB_LIBRARY_COMP

    _print_header("Building Block Library")

    all_nodes   = BB_LIBRARY["T3_nodes"] + BB_LIBRARY["S4_nodes"]
    all_linkers = BB_LIBRARY["L2_linkers"]

    print(f"\n  ■ Tritopic Nodes (T3) — hexagonal / kagome / HXL topologies")
    print(f"  {'─'*60}")
    for bb in BB_LIBRARY["T3_nodes"]:
        c,n,h,o,tot = BB_LIBRARY_COMP.get(bb,(0,0,0,0,0))
        formula = f"C{c}H{h}" + (f"N{n}" if n else "")
        smiles  = BB_SMILES.get(bb,"")[:35]
        print(f"  {bb:<12} {formula:<12} SMILES: {smiles}")

    print(f"\n  ■ Tetratopic Nodes (S4) — square / kagome topologies")
    print(f"  {'─'*60}")
    for bb in BB_LIBRARY["S4_nodes"]:
        c,n,h,o,tot = BB_LIBRARY_COMP.get(bb,(0,0,0,0,0))
        formula = f"C{c}H{h}" + (f"N{n}" if n else "")
        smiles  = BB_SMILES.get(bb,"")[:35]
        print(f"  {bb:<12} {formula:<12} SMILES: {smiles}")

    print(f"\n  ■ Ditopic Linkers (L2)")
    print(f"  {'─'*60}")
    for bb in all_linkers:
        c,n,h,o,tot = BB_LIBRARY_COMP.get(bb,(0,0,0,0,0))
        formula = f"C{c}H{h}" + (f"N{n}" if n else "")
        smiles  = BB_SMILES.get(bb,"")[:35]
        print(f"  {bb:<12} {formula:<12} SMILES: {smiles}")

    print(f"\n  ■ Linkage Types & Connection Groups")
    print(f"  {'─'*50}")
    for lk, (nf,lf) in BB_LIBRARY["conn_groups"].items():
        print(f"  {lk:<20} node_func={nf:<12} linker_func={lf}")


# ════════════════════════════════════════════════════════════════════════════
# COMMAND: from-monomers
# ════════════════════════════════════════════════════════════════════════════

def cmd_from_monomers(args):
    """Predict COF from two monomer CIF files."""
    from analysis.monomer_cif_to_cof import predict_cof_from_monomers

    _print_header("COF Prediction from Monomer CIFs")

    node_path   = Path(args.node_cif)
    linker_path = Path(args.linker_cif)

    for p in (node_path, linker_path):
        if not p.exists():
            print(f"\n  ✗ File not found: {p}")
            return

    topology = getattr(args, "topology", "") or ""
    stacking = getattr(args, "stacking", "") or ""

    print(f"\n  Node CIF:   {node_path.name}")
    print(f"  Linker CIF: {linker_path.name}")
    if topology:
        print(f"  Topology:   {topology} (user-specified)")
    if stacking:
        print(f"  Stacking:   {stacking} (user-specified)")

    result = predict_cof_from_monomers(
        node_cif   = node_path,
        linker_cif = linker_path,
        topology   = topology,
        stacking   = stacking,
        verbose    = True,
    )

    print(result.summary)

    if args.json:
        _write_json({
            "node": {
                "cif":       result.node.cif_name,
                "formula":   result.node.formula,
                "fg":        result.node.primary_fg,
                "n_sites":   result.node.n_reactive,
                "connectivity": result.node.connectivity,
                "arm_length_A": result.node.arm_length,
                "matched_bb":   result.node.matched_bb,
                "match_conf":   result.node.match_conf,
            },
            "linker": {
                "cif":       result.linker.cif_name,
                "formula":   result.linker.formula,
                "fg":        result.linker.primary_fg,
                "n_sites":   result.linker.n_reactive,
                "connectivity": result.linker.connectivity,
                "arm_length_A": result.linker.arm_length,
                "matched_bb":   result.linker.matched_bb,
                "match_conf":   result.linker.match_conf,
            },
            "linkage":  result.linkage,
            "topology": result.topology,
            "stacking": result.stacking,
            "stacking_prefs": result.stacking_prefs,
            "geometry": {
                "cell_a_A":       result.a_angstrom,
                "cell_c_A":       result.c_angstrom,
                "layer_spacing_A": result.layer_spacing,
                "void_fraction":  result.void_fraction,
                "bet_m2g":        result.bet_m2g,
                "pld_A":          result.pld,
                "lcd_A":          result.lcd,
                "density_gcm3":   result.density,
            },
            "electronics": {
                "band_gap_eV": result.band_gap,
            },
            "adsorption": {
                "CO2_298K_1bar_mmolg":  result.co2_uptake,
                "CH4_298K_65bar_mmolg": result.ch4_uptake,
                "H2_77K_100bar_mmolg":  result.h2_uptake,
                "CO2_N2_selectivity":   result.co2_n2_sel,
            },
            "mechanical": {
                "E_inplane_GPa": result.e_inplane,
            },
            "stability": {
                "thermal_decomp_C": result.thermal_C,
                "water_stability":  result.water_stab,
            },
            "pxrd_peaks": result.pxrd_peaks,
            "synthesis": {
                "score":     result.synth_score,
                "solvent":   result.synth_solvent,
                "catalyst":  result.synth_catalyst,
                "temp":      result.synth_temp,
            },
            "warnings": result.warnings,
        }, args.json)


# ════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cofgen_tool",
        description="COFGen: Complete COF Analysis & Property Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cofgen_tool.py predict  --cif my_cof.cif
  python cofgen_tool.py predict  --node T3_BENZ --linker L2_PYRN --linkage imine
  python cofgen_tool.py from-monomers --node-cif TAPB.cif --linker-cif PDA.cif
  python cofgen_tool.py from-monomers --node-cif porphyrin.cif --linker-cif PMDA.cif --topology sql
  python cofgen_tool.py reverse  --cif unknown.cif
  python cofgen_tool.py pxrd     --cif my_cof.cif --plot
  python cofgen_tool.py stacking --cif my_cof.cif
  python cofgen_tool.py synthesis --node T3_BENZ --linker L2_PYRN --linkage imine
  python cofgen_tool.py generate --linkage imine --co2 3.5 -n 30
  python cofgen_tool.py list-bbs
        """
    )
    sp = p.add_subparsers(dest="command", required=True)

    # -- predict --
    pp = sp.add_parser("predict", help="Full property prediction")
    pp.add_argument("--cif",     help="CIF file path")
    pp.add_argument("--node",    help="Node BB name (e.g. T3_BENZ)")
    pp.add_argument("--linker",  help="Linker BB name (e.g. L2_PYRN)")
    pp.add_argument("--linkage", help="Linkage type", default="imine")
    pp.add_argument("--topology",help="Topology (hcb/sql/kgm)", default="hcb")
    pp.add_argument("--json",    help="Write JSON output to file")

    # -- reverse --
    rp = sp.add_parser("reverse", help="Reverse-engineer monomers from CIF")
    rp.add_argument("--cif",  required=True, help="CIF file path")
    rp.add_argument("--json", help="Write JSON output to file")

    # -- pxrd --
    xp = sp.add_parser("pxrd", help="Simulate PXRD pattern")
    xp.add_argument("--cif",        help="CIF file path")
    xp.add_argument("--node",       help="Node BB")
    xp.add_argument("--linker",     help="Linker BB")
    xp.add_argument("--linkage",    default="imine")
    xp.add_argument("--wavelength", type=float, default=1.5406, help="X-ray wavelength (Å)")
    xp.add_argument("--plot",       action="store_true", help="Save PNG plot")
    xp.add_argument("--plot-out",   dest="plot_out", help="PNG output filename")
    xp.add_argument("--json",       help="Write JSON output to file")

    # -- stacking --
    st = sp.add_parser("stacking", help="Analyse stacking geometry")
    st.add_argument("--cif",     help="CIF file path")
    st.add_argument("--linkage", default="imine")
    st.add_argument("--json",    help="Write JSON output to file")

    # -- synthesis --
    sc = sp.add_parser("synthesis", help="Predict synthesis conditions")
    sc.add_argument("--node",       required=True)
    sc.add_argument("--linker",     required=True)
    sc.add_argument("--linkage",    required=True)
    sc.add_argument("--topology",   default="hcb")
    sc.add_argument("--observation",help="Describe experimental outcome for diagnosis")
    sc.add_argument("--json",       help="Write JSON output to file")

    # -- generate --
    gp = sp.add_parser("generate", help="Generate candidate COF structures")
    gp.add_argument("-n",        type=int, default=20)
    gp.add_argument("--linkage", help="Constrain linkage type")
    gp.add_argument("--co2",    type=float, help="Min CO₂ uptake (mmol/g)")
    gp.add_argument("--bet",    type=float, help="Min BET surface area (m²/g)")
    gp.add_argument("--json",   help="Write JSON output to file")

    # -- from-monomers --
    fm = sp.add_parser("from-monomers",
                        help="Predict COF from two monomer CIF files")
    fm.add_argument("--node-cif",    required=True,
                    help="CIF file of the node building block (amine/boronic acid)")
    fm.add_argument("--linker-cif",  required=True,
                    help="CIF file of the linker building block (aldehyde/dianhydride/diol)")
    fm.add_argument("--topology",    default="",
                    help="Override topology (hcb/sql/kgm) — auto-detected if omitted")
    fm.add_argument("--stacking",    default="",
                    help="Override stacking (AA/AB/ABC) — predicted if omitted")
    fm.add_argument("--json",        help="Write JSON output to file")

    # -- list-bbs --
    sp.add_parser("list-bbs", help="List all building blocks")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "predict":       cmd_predict,
        "reverse":       cmd_reverse,
        "pxrd":          cmd_pxrd,
        "stacking":      cmd_stacking,
        "synthesis":     cmd_synthesis,
        "generate":      cmd_generate,
        "from-monomers": cmd_from_monomers,
        "list-bbs":      cmd_list_bbs,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
