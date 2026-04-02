"""
benchmark.py
============
Compare COFGen against baseline methods:

  Baseline 1 — Random enumeration (pyCOFBuilder exhaustive)
  Baseline 2 — Screening from ReDD-COFFEE (no generation)
  Baseline 3 — Substitution (swap one BB in a known structure)

Metrics reported:
  - SUN rate (stable / unique / novel)
  - Property MAE vs target
  - Internal diversity
  - Synthesizability distribution

Usage:
    python evaluation/benchmark.py \
        --cofgen_results   outputs/co2_targeted/results.json \
        --reference_dir    data/processed/ \
        --out              outputs/benchmark.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from decoder.reticular_decoder import COFSpec, BB_LIBRARY, RetricularDecoder
from decoder.validity_checker  import ValidityChecker
from evaluation.metrics        import full_evaluation, print_evaluation


# ─────────────────────────────────────────────────────────────────────────────
# Baseline generators
# ─────────────────────────────────────────────────────────────────────────────

def random_enumeration_baseline(
    n:           int,
    linkage:     Optional[str] = None,
    topology:    Optional[str] = None,
    seed:        int = 0,
) -> List[COFSpec]:
    """
    Baseline 1: uniformly sample from the pyCOFBuilder design space.
    """
    rng      = random.Random(seed)
    nodes    = BB_LIBRARY["T3_nodes"] + BB_LIBRARY["S4_nodes"]
    linkers  = BB_LIBRARY["L2_linkers"]
    linkages = list(BB_LIBRARY["conn_groups"].keys())
    topols   = ["hcb", "sql", "kgm", "hxl", "dia"]

    from utils.featurisation import STACKING_PATTERNS

    specs = []
    for _ in range(n):
        lk  = linkage  or rng.choice(linkages)
        tp  = topology or rng.choice(topols)
        st  = rng.choice(STACKING_PATTERNS[:3])  # AA, AB, ABC
        nd  = rng.choice(nodes)
        ln  = rng.choice(linkers)
        nf, lf = BB_LIBRARY["conn_groups"].get(lk, ("NH2", "CHO"))
        specs.append(COFSpec(lk, tp, st, nd, ln, nf, lf))
    return specs


def substitution_baseline(
    reference_specs: List[COFSpec],
    n:               int,
    seed:            int = 1,
) -> List[COFSpec]:
    """
    Baseline 3: take known structures and swap one BB randomly.
    """
    rng     = random.Random(seed)
    nodes   = BB_LIBRARY["T3_nodes"] + BB_LIBRARY["S4_nodes"]
    linkers = BB_LIBRARY["L2_linkers"]

    specs = []
    for _ in range(n):
        base = rng.choice(reference_specs)
        if rng.random() < 0.5:
            new_node = rng.choice(nodes)
            specs.append(COFSpec(
                base.linkage_type, base.topology, base.stacking,
                new_node, base.linker_bb, base.node_func, base.linker_func,
            ))
        else:
            new_linker = rng.choice(linkers)
            specs.append(COFSpec(
                base.linkage_type, base.topology, base.stacking,
                base.node_bb, new_linker, base.node_func, base.linker_func,
            ))
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Run one method and evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_method(
    name:          str,
    specs:         List[COFSpec],
    output_dir:    Path,
    reference_fps: Optional[set] = None,
    target_props:  Optional[Dict] = None,
    skip_validity: bool = False,
) -> Dict:
    out = output_dir / name
    rd  = RetricularDecoder(out / "structures")
    paths = rd.assemble_batch(specs)

    checker = ValidityChecker(uff_relax_enable=not skip_validity)
    names   = [s.to_pycofbuilder_name() for s in specs]
    reports = checker.check_batch(names, paths, specs)

    result = full_evaluation(reports, specs, reference_fps, target_props)
    result["method"] = name
    result["n"]      = len(specs)

    # Save
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "eval.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Method: {name}")
    print_evaluation(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    cofgen_results_path: Path,
    reference_dir:       Path,
    output_dir:          Path,
    n_baseline:          int  = 200,
    target_props:        Optional[Dict] = None,
    skip_validity:       bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COFGen results
    with open(cofgen_results_path) as f:
        cofgen_data = json.load(f)

    cofgen_specs = []
    for s in cofgen_data.get("structures", []):
        spec_str = s.get("spec", "")
        # Reconstruct spec from string (best effort)
        cofgen_specs.append(COFSpec(
            linkage_type = s.get("linkage_type", "imine"),
            topology     = s.get("topology", "hcb"),
            stacking     = s.get("stacking", "AA"),
            node_bb      = "T3_BENZ",
            linker_bb    = "L2_BENZ",
            node_func    = "NH2",
            linker_func  = "CHO",
        ))

    # Build reference fingerprint set
    from evaluation.metrics import structure_fingerprint
    ref_fps = set()
    from data.crystal_graph import CrystalGraph
    for p in sorted(reference_dir.glob("*.json"))[:5000]:
        try:
            cg = CrystalGraph.load(p)
            spec = COFSpec(cg.linkage_type, cg.topology, cg.stacking,
                           "T3_BENZ", "L2_BENZ", "NH2", "CHO")
            ref_fps.add(structure_fingerprint(spec))
        except Exception:
            continue

    print(f"Reference fingerprints: {len(ref_fps):,}")

    results = []

    # COFGen
    if cofgen_specs:
        r = evaluate_method(
            "cofgen", cofgen_specs, output_dir,
            ref_fps, target_props, skip_validity,
        )
        results.append(r)

    # Baseline 1: random
    rand_specs = random_enumeration_baseline(n_baseline)
    r = evaluate_method(
        "random_enumeration", rand_specs, output_dir,
        ref_fps, target_props, skip_validity,
    )
    results.append(r)

    # Baseline 3: substitution (if we have reference specs)
    if cofgen_specs:
        sub_specs = substitution_baseline(cofgen_specs[:50], n_baseline)
        r = evaluate_method(
            "substitution", sub_specs, output_dir,
            ref_fps, target_props, skip_validity,
        )
        results.append(r)

    # Save comparison table
    comparison = {m["method"]: {
        "sun_rate":   m["sun"]["sun_rate"],
        "stable_rate": m["sun"]["stable_rate"],
        "unique_rate": m["sun"]["unique_rate"],
        "novel_rate":  m["sun"]["novel_rate"],
        "diversity":   m["diversity"]["internal_diversity"],
        "mean_synth":  m["validity"]["mean_synth"],
    } for m in results}

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*50)
    print("Benchmark comparison")
    print("="*50)
    header = f"{'Method':<25} {'SUN':>6} {'Stable':>7} {'Novel':>7} {'Diversity':>10}"
    print(header)
    print("-" * len(header))
    for method, vals in comparison.items():
        print(f"{method:<25} {vals['sun_rate']:>6.1%} {vals['stable_rate']:>7.1%} "
              f"{vals['novel_rate']:>7.1%} {vals['diversity']:>10.3f}")
    print("="*50)
    print(f"\nFull results saved to {output_dir}/comparison.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cofgen_results", type=str, required=True)
    p.add_argument("--reference_dir",  type=str, default="data/processed/")
    p.add_argument("--out",            type=str, default="outputs/benchmark/")
    p.add_argument("--n_baseline",     type=int, default=200)
    p.add_argument("--skip_validity",  action="store_true")
    args = p.parse_args()
    run_benchmark(
        Path(args.cofgen_results),
        Path(args.reference_dir),
        Path(args.out),
        args.n_baseline,
        skip_validity=args.skip_validity,
    )
