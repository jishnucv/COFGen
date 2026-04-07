"""
metrics.py
==========
Evaluation metrics for COFGen generated structures.

COF-adapted SUN (Stable, Unique, Novel) rate, plus:
  - Property MAE (target vs generated)
  - Topology diversity
  - Linkage diversity
  - Fingerprint novelty vs training set
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from decoder.validity_checker import ValidityReport


# ─────────────────────────────────────────────────────────────────────────────
# Structure fingerprint (without RDKit — formula-based)
# ─────────────────────────────────────────────────────────────────────────────

def structure_fingerprint(spec) -> str:
    """
    Simple string fingerprint: sorted (node_bb, linker_bb, linkage, topology, stacking).
    Two structures with identical fingerprints are considered duplicates.
    """
    parts = sorted([
        spec.node_bb, spec.linker_bb,
        spec.linkage_type, spec.topology, spec.stacking,
    ])
    return "|".join(parts)


def tanimoto_similarity(fp1: str, fp2: str) -> float:
    """Jaccard similarity over character n-grams (proxy for Tanimoto)."""
    def ngrams(s, n=3):
        return set(s[i:i+n] for i in range(len(s)-n+1))
    a, b = ngrams(fp1), ngrams(fp2)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ─────────────────────────────────────────────────────────────────────────────
# SUN rate
# ─────────────────────────────────────────────────────────────────────────────

def compute_sun_rate(
    reports:         List[ValidityReport],
    specs:           List,                      # List[COFSpec]
    reference_fps:   Optional[set] = None,      # fingerprints of training set
    novelty_thresh:  float = 0.9,               # Tanimoto threshold for novelty
) -> Dict[str, float]:
    """
    Returns SUN rate and components:
      stable_rate  : fraction of generated structures that pass validity + UFF
      unique_rate  : fraction that are not duplicates within the generated batch
      novel_rate   : fraction not found in the reference set
      sun_rate     : stable AND unique AND novel
    """
    n = len(reports)
    if n == 0:
        return {"sun_rate": 0.0, "stable_rate": 0.0, "unique_rate": 0.0, "novel_rate": 0.0}

    # Stable
    stable_mask = np.array([r.is_stable for r in reports], dtype=bool)

    # Unique (within batch)
    fps = [structure_fingerprint(s) for s in specs]
    seen: set = set()
    unique_mask = np.zeros(n, dtype=bool)
    for i, fp in enumerate(fps):
        if fp not in seen:
            unique_mask[i] = True
            seen.add(fp)

    # Novel (vs reference)
    if reference_fps is not None:
        novel_mask = np.array([fp not in reference_fps for fp in fps], dtype=bool)
    else:
        novel_mask = np.ones(n, dtype=bool)

    sun_mask = stable_mask & unique_mask & novel_mask

    return {
        "n_generated":  n,
        "stable_rate":  float(stable_mask.mean()),
        "unique_rate":  float(unique_mask.mean()),
        "novel_rate":   float(novel_mask.mean()),
        "sun_rate":     float(sun_mask.mean()),
        "n_sun":        int(sun_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Structural diversity metrics
# ─────────────────────────────────────────────────────────────────────────────

def topology_distribution(specs: List) -> Dict[str, float]:
    """Fraction of generated structures per topology."""
    counts = Counter(s.topology for s in specs)
    total  = max(len(specs), 1)
    return {k: v / total for k, v in counts.most_common()}


def linkage_distribution(specs: List) -> Dict[str, float]:
    counts = Counter(s.linkage_type for s in specs)
    total  = max(len(specs), 1)
    return {k: v / total for k, v in counts.most_common()}


def internal_diversity(specs: List, n_pairs: int = 1000) -> float:
    """
    Mean pairwise Tanimoto dissimilarity over random pairs.
    Higher = more diverse batch.
    """
    n = len(specs)
    if n < 2:
        return 0.0
    fps = [structure_fingerprint(s) for s in specs]
    rng = np.random.default_rng(42)
    pairs = rng.integers(0, n, size=(min(n_pairs, n*(n-1)//2), 2))
    # Filter i != j
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if len(pairs) == 0:
        return 0.0
    sims = [tanimoto_similarity(fps[i], fps[j]) for i, j in pairs[:n_pairs]]
    return float(1.0 - np.mean(sims))


# ─────────────────────────────────────────────────────────────────────────────
# Property accuracy
# ─────────────────────────────────────────────────────────────────────────────

def property_mae(
    reports:  List[ValidityReport],
    targets:  Dict[str, float],   # {prop_name: target_value}
) -> Dict[str, float]:
    """
    For property-conditioned generation: compute MAE between target and
    generated structure properties (from validity reports / GCMC labels).
    """
    results = {}
    for prop, target in targets.items():
        vals = [getattr(r, prop, None) for r in reports
                if getattr(r, prop, None) is not None
                and not np.isnan(getattr(r, prop, float("nan")))]
        if vals:
            results[f"mae_{prop}"] = float(np.mean(np.abs(np.array(vals) - target)))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(
    reports:       List[ValidityReport],
    specs:         List,
    reference_fps: Optional[set] = None,
    target_props:  Optional[Dict[str, float]] = None,
) -> Dict:
    sun = compute_sun_rate(reports, specs, reference_fps)
    div = {
        "internal_diversity": internal_diversity(specs),
        "topology_dist":      topology_distribution(specs),
        "linkage_dist":       linkage_distribution(specs),
    }
    validity = ValidityReport.__class__  # just for type hint clarity
    validity_summary = {
        "valid_rate":     float(np.mean([r.is_valid for r in reports])),
        "stable_rate":    float(np.mean([r.is_stable for r in reports])),
        "mean_vf":        float(np.mean([r.void_fraction for r in reports])),
        "mean_pld":       float(np.mean([r.pld for r in reports])),
        "mean_lcd":       float(np.mean([r.lcd for r in reports])),
        "mean_synth":     float(np.mean([r.synth_score for r in reports])),
        "mean_composite": float(np.mean([r.composite_score for r in reports])),
    }
    prop_acc = property_mae(reports, target_props or {})

    return {
        "sun":           sun,
        "diversity":     div,
        "validity":      validity_summary,
        "property_mae":  prop_acc,
    }


def print_evaluation(eval_dict: Dict) -> None:
    print("\n" + "="*60)
    print("COFGen Evaluation Report")
    print("="*60)
    sun = eval_dict["sun"]
    print(f"\n[SUN]")
    print(f"  Generated:    {sun['n_generated']}")
    print(f"  Stable:       {sun['stable_rate']:.1%}")
    print(f"  Unique:       {sun['unique_rate']:.1%}")
    print(f"  Novel:        {sun['novel_rate']:.1%}")
    print(f"  SUN rate:     {sun['sun_rate']:.1%}  ({sun['n_sun']} structures)")

    v = eval_dict["validity"]
    print(f"\n[Validity]")
    print(f"  Valid rate:   {v['valid_rate']:.1%}")
    print(f"  Mean VF:      {v['mean_vf']:.3f}")
    print(f"  Mean PLD:     {v['mean_pld']:.2f} Å")
    print(f"  Mean synth:   {v['mean_synth']:.3f}")

    d = eval_dict["diversity"]
    print(f"\n[Diversity]")
    print(f"  Internal:     {d['internal_diversity']:.3f}")
    top_topo = list(d["topology_dist"].items())[:3]
    print(f"  Top topos:    {top_topo}")
    top_lk   = list(d["linkage_dist"].items())[:3]
    print(f"  Top linkages: {top_lk}")

    if eval_dict["property_mae"]:
        print(f"\n[Property MAE]")
        for k, v_ in eval_dict["property_mae"].items():
            print(f"  {k}: {v_:.4f}")
    print("="*60 + "\n")
