"""
train_flowmatch.py
==================
Train the base conditional flow matching model over the COF latent space.

Requires a pre-trained encoder checkpoint (from train_encoder.py).
The encoder is frozen; only the FlowMatchingNetwork is trained.

Usage:
    python scripts/train_flowmatch.py \
        --encoder_ckpt checkpoints/encoder/best.pt \
        --data data/processed/ \
        --epochs 200 \
        --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] torch not available — dry-run mode")

from data.cof_dataset  import COFDataset, collate_cof_graphs
from models.encoder    import COFEncoder
from models.flow_matching import FlowMatchingNetwork, cfm_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--data",         type=str, default="data/processed/")
    p.add_argument("--out",          type=str, default="checkpoints/flowmatch_base/")
    p.add_argument("--epochs",       type=int, default=200)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--latent_dim",   type=int, default=256)
    p.add_argument("--hidden_dim",   type=int, default=512)
    p.add_argument("--n_layers",     type=int, default=8)
    p.add_argument("--cfg_drop",     type=float, default=0.10,
                   help="Fraction of conditioning to drop for CFG training")
    p.add_argument("--properties",   nargs="+",
                   default=["bet_surface_area", "co2_uptake_298k_1bar",
                             "void_fraction", "pore_limiting_diameter"])
    p.add_argument("--device", type=str,
                   default="cuda" if HAS_TORCH and
                   __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def encode_batch(encoder, batch, device):
    """Run encoder in no-grad mode to get latent vectors."""
    with torch.no_grad():
        z, mu, _ = encoder(batch)
    return z, mu


def train_epoch(flow_net, encoder, loader, optimizer, device, cfg_drop, properties):
    flow_net.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

        # Get latent from frozen encoder (use mu for stable training)
        z_1, mu = encode_batch(encoder, batch, device)
        z_1 = mu  # use mean for flow matching target (less noisy)

        # Build property conditioning dict
        props = {}
        for p in properties:
            key = f"prop_{p}"
            if key in batch:
                val = batch[key].float()
                # CFG: randomly zero out some samples
                if cfg_drop > 0:
                    mask = torch.rand(len(val), device=device) < cfg_drop
                    val = val.masked_fill(mask, 0.0)
                props[p] = val

        linkage_idx  = batch.get("linkage_idx")
        topology_idx = batch.get("topology_idx")

        optimizer.zero_grad()
        loss = cfm_loss(flow_net, z_1, props, linkage_idx, topology_idx)
        loss.backward()
        nn.utils.clip_grad_norm_(flow_net.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return {"flow_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def val_epoch(flow_net, encoder, loader, device, properties):
    flow_net.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        _, mu = encode_batch(encoder, batch, device)
        props = {
            p: batch[f"prop_{p}"].float()
            for p in properties if f"prop_{p}" in batch
        }
        loss = cfm_loss(flow_net, mu, props,
                        batch.get("linkage_idx"), batch.get("topology_idx"))
        total_loss += loss.item()
        n_batches  += 1

    return {"val_flow_loss": total_loss / max(n_batches, 1)}


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_TORCH:
        print("[dry_run] torch unavailable. Architecture check complete.")
        return

    device = torch.device(args.device)

    # ── Load frozen encoder ───────────────────────────────────────────────────
    print(f"Loading encoder from {args.encoder_ckpt}")
    enc_ckpt   = torch.load(args.encoder_ckpt, map_location=device)
    enc_args   = enc_ckpt.get("args", {})
    encoder    = COFEncoder(
        latent_dim=enc_args.get("latent_dim", args.latent_dim),
        hidden_dim=enc_args.get("hidden_dim", 256),
    )
    encoder.load_state_dict(enc_ckpt["encoder"])
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ── Flow matching network ─────────────────────────────────────────────────
    flow_net = FlowMatchingNetwork(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )
    flow_net.to(device)
    n_params = sum(p.numel() for p in flow_net.parameters() if p.requires_grad)
    print(f"FlowMatchingNetwork: {n_params:,} trainable parameters")

    if args.dry_run:
        print("[dry_run] Exiting.")
        return

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = COFDataset(args.data, split="train", properties=args.properties)
    val_ds   = COFDataset(args.data, split="val",   properties=args.properties)

    if len(train_ds) == 0:
        print("[ERROR] No training data. Run build_dataset.py + compute_properties.py first.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_cof_graphs, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_cof_graphs, num_workers=2,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(flow_net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val  = float("inf")
    history   = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tm = train_epoch(flow_net, encoder, train_loader,
                         optimizer, device, args.cfg_drop, args.properties)
        vm = val_epoch(flow_net, encoder, val_loader, device, args.properties)
        scheduler.step()
        elapsed = time.time() - t0

        log = {"epoch": epoch, "elapsed": elapsed, **tm, **vm}
        history.append(log)
        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"flow={tm['flow_loss']:.5f}  val={vm['val_flow_loss']:.5f}  "
              f"t={elapsed:.1f}s")

        if vm["val_flow_loss"] < best_val:
            best_val = vm["val_flow_loss"]
            # Load spec_decoder from encoder checkpoint for bundling
            _enc_ckpt = torch.load(args.encoder_ckpt, map_location=device)
            torch.save({
                "epoch":        epoch,
                "flow_net":     flow_net.state_dict(),
                "encoder":      encoder.state_dict(),
                "spec_decoder": _enc_ckpt.get("spec_decoder", {}),
                "val_loss":     best_val,
                "args":         vars(args),
            }, out_dir / "best.pt")
            print(f"  ↑ New best ({best_val:.5f}) saved (bundled encoder+flow+spec_decoder).")

        if epoch % 20 == 0:
            torch.save({
                "epoch":     epoch,
                "flow_net":  flow_net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, out_dir / f"epoch_{epoch:04d}.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val flow loss: {best_val:.5f}")


if __name__ == "__main__":
    if HAS_TORCH:
        import torch
    main()
