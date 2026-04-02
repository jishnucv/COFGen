"""
generate.py
===========
Generate novel COF structures using a trained COFGen model.

Usage:
    # Unconditional generation
    python scripts/generate.py --checkpoint checkpoints/base.pt --n 500

    # Property-targeted
    python scripts/generate.py \
        --checkpoint checkpoints/base.pt \
        --adapter_co2 checkpoints/adapter_co2.pt \
        --co2_target 4.0 \
        --bet_target 2500 \
        --linkage imine \
        --topology hcb \
        --n 200 \
        --out outputs/co2_targeted/ \
        --guidance_scale 2.0 \
        --n_steps 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from decoder.reticular_decoder import SpecDecoderMLP, RetricularDecoder, latents_to_structures
from decoder.validity_checker  import ValidityChecker
from evaluation.metrics        import full_evaluation, print_evaluation
from utils.featurisation       import (
    LINKAGE_TO_IDX, TOPOLOGY_TO_IDX, normalise_property,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--n",            type=int, default=100)
    p.add_argument("--out",          type=str, default="outputs/generated/")
    p.add_argument("--n_steps",      type=int, default=50)
    p.add_argument("--solver",       type=str, default="euler", choices=["euler", "rk4"])
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--device",       type=str, default="cpu")

    # Property targets
    p.add_argument("--co2_target",   type=float, default=None,  help="CO2 uptake mmol/g")
    p.add_argument("--bet_target",   type=float, default=None,  help="BET surface area m²/g")
    p.add_argument("--bg_target",    type=float, default=None,  help="Band gap eV")
    p.add_argument("--pld_target",   type=float, default=None,  help="Pore limiting diameter Å")
    p.add_argument("--linkage",      type=str,   default=None)
    p.add_argument("--topology",     type=str,   default=None)

    # Adapter checkpoints
    p.add_argument("--adapter_co2",  type=str, default=None)
    p.add_argument("--adapter_bet",  type=str, default=None)
    p.add_argument("--adapter_bg",   type=str, default=None)
    p.add_argument("--adapter_link", type=str, default=None)
    p.add_argument("--adapter_topo", type=str, default=None)

    p.add_argument("--skip_validity", action="store_true")
    p.add_argument("--reference_fps", type=str, default=None,
                   help="JSON file with set of training fingerprints for novelty check")
    return p.parse_args()


def main():
    args = parse_args()

    if not HAS_TORCH:
        print("[ERROR] torch is required for generation.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    from models.encoder      import COFEncoder
    from models.flow_matching import FlowMatchingNetwork, sample_ode, sample_cfg
    from models.adapters     import build_adapter, MultiAdapter

    latent_dim  = ckpt["args"].get("latent_dim", 256)
    hidden_dim  = ckpt["args"].get("hidden_dim", 256)

    encoder = COFEncoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
    encoder.load_state_dict(ckpt.get("encoder", {}))
    encoder.to(device).eval()

    flow_net = FlowMatchingNetwork(latent_dim=latent_dim)
    if "flow_net" in ckpt:
        flow_net.load_state_dict(ckpt["flow_net"])
    flow_net.to(device).eval()

    spec_decoder = SpecDecoderMLP(latent_dim=latent_dim)
    if "spec_decoder" in ckpt:
        spec_decoder.load_state_dict(ckpt["spec_decoder"])
    spec_decoder.to(device).eval()

    # ── Load adapters ─────────────────────────────────────────────────────────
    adapter_config = {"scalar_properties": [], "linkage": False, "topology": False}
    if args.adapter_co2:  adapter_config["scalar_properties"].append("co2_uptake_298k_1bar")
    if args.adapter_bet:  adapter_config["scalar_properties"].append("bet_surface_area")
    if args.adapter_bg:   adapter_config["scalar_properties"].append("band_gap")
    if args.adapter_link: adapter_config["linkage"] = True
    if args.adapter_topo: adapter_config["topology"] = True

    adapter = build_adapter(adapter_config)
    # Load adapter checkpoints
    _adapter_ckpts = {
        "scalar_co2_uptake_298k_1bar": args.adapter_co2,
        "scalar_bet_surface_area":     args.adapter_bet,
        "scalar_band_gap":             args.adapter_bg,
        "linkage":                     args.adapter_link,
        "topology":                    args.adapter_topo,
    }
    for adapter_name, ckpt_path in _adapter_ckpts.items():
        if ckpt_path and adapter_name in adapter.adapters:
            sub_ckpt = torch.load(ckpt_path, map_location=device)
            adapter.adapters[adapter_name].load_state_dict(sub_ckpt["adapter"])
            print(f"  Loaded adapter: {adapter_name} from {ckpt_path}")
    adapter.to(device).eval()

    # ── Build property conditioning ───────────────────────────────────────────
    # Property conditioning is built fresh each batch (handles partial last batch)

    # ── Generate in batches ───────────────────────────────────────────────────
    reticular_decoder = RetricularDecoder(out_dir / "structures")
    checker           = ValidityChecker(uff_relax_enable=not args.skip_validity)

    all_specs   = []
    all_reports = []
    remaining   = args.n

    print(f"Generating {args.n} structures...")

    while remaining > 0:
        batch_n = min(args.batch_size, remaining)

        # Rebuild property tensors at exact batch size
        batch_props = {}
        if args.co2_target is not None:
            batch_props["co2_uptake_298k_1bar"] = torch.tensor(
                [normalise_property("co2_uptake_298k_1bar", args.co2_target)] * batch_n,
                device=device)
        if args.bet_target is not None:
            batch_props["bet_surface_area"] = torch.tensor(
                [normalise_property("bet_surface_area", args.bet_target)] * batch_n,
                device=device)
        if args.bg_target is not None:
            batch_props["band_gap"] = torch.tensor(
                [normalise_property("band_gap", args.bg_target)] * batch_n,
                device=device)

        batch_lk = None
        batch_tp = None
        if args.linkage:
            lk_i = LINKAGE_TO_IDX.get(args.linkage, 0)
            batch_lk = torch.full((batch_n,), lk_i, dtype=torch.long, device=device)
        if args.topology:
            tp_i = TOPOLOGY_TO_IDX.get(args.topology, 0)
            batch_tp = torch.full((batch_n,), tp_i, dtype=torch.long, device=device)

        # Sample latents
        if args.guidance_scale > 1.0 and props:
            z = sample_cfg(
                flow_net, batch_n,
                n_steps=args.n_steps,
                props=batch_props,
                linkage_idx=batch_lk,
                topology_idx=batch_tp,
                guidance_scale=args.guidance_scale,
                device=str(device),
            )
        else:
            z = sample_ode(
                flow_net, batch_n,
                n_steps=args.n_steps,
                props=batch_props,
                linkage_idx=batch_lk,
                topology_idx=batch_tp,
                device=str(device),
                solver=args.solver,
            )

        # Decode
        specs, paths = latents_to_structures(
            z, spec_decoder, reticular_decoder,
            temperature=args.temperature,
        )
        all_specs.extend(specs)

        # Validate
        names = [s.to_pycofbuilder_name() for s in specs]
        reports = checker.check_batch(names, paths, specs)
        all_reports.extend(reports)

        valid_in_batch = sum(r.is_valid for r in reports)
        print(f"  Batch {len(all_specs)}/{args.n}: "
              f"{valid_in_batch}/{batch_n} valid")
        remaining -= batch_n

    # ── Evaluation ────────────────────────────────────────────────────────────
    reference_fps = None
    if args.reference_fps:
        with open(args.reference_fps) as f:
            reference_fps = set(json.load(f))

    target_props = {}
    if args.co2_target: target_props["co2_uptake_298k_1bar"] = args.co2_target
    if args.bet_target:  target_props["bet_surface_area"]    = args.bet_target

    eval_dict = full_evaluation(all_reports, all_specs, reference_fps, target_props)
    print_evaluation(eval_dict)

    # Save results
    results = {
        "args":       vars(args),
        "evaluation": eval_dict,
        "structures": [
            {**r.to_dict(), "spec": str(s)}
            for r, s in zip(all_reports, all_specs)
        ],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save ranked candidate list (by composite score)
    candidates = sorted(
        zip(all_specs, all_reports),
        key=lambda x: x[1].composite_score,
        reverse=True,
    )
    with open(out_dir / "candidates_ranked.json", "w") as f:
        json.dump([
            {
                "rank":        i + 1,
                "pcb_name":    s.to_pycofbuilder_name(),
                "linkage":     s.linkage_type,
                "topology":    s.topology,
                "stacking":    s.stacking,
                "node_bb":     s.node_bb,
                "linker_bb":   s.linker_bb,
                "valid":       r.is_valid,
                "stable":      r.is_stable,
                "void_frac":   r.void_fraction,
                "pld":         r.pld,
                "synth_score": r.synth_score,
                "composite":   r.composite_score,
            }
            for i, (s, r) in enumerate(candidates)
        ], f, indent=2)

    print(f"\nResults saved to {out_dir}")
    print(f"Top candidate: {candidates[0][0]}" if candidates else "No candidates")


if __name__ == "__main__":
    main()
