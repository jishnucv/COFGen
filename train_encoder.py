"""
train_encoder.py
================
Train the COFEncoder VAE.

Usage:
    python scripts/train_encoder.py --data data/processed/ --epochs 100 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] torch not installed — running in dry-run / architecture check mode")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.cof_dataset import COFDataset, collate_cof_graphs
from models.encoder import COFEncoder, kl_divergence, model_summary
from decoder.reticular_decoder import SpecDecoderMLP, spec_decoder_loss
from utils.featurisation import LINKAGE_TO_IDX, TOPOLOGY_TO_IDX, STACKING_TO_IDX


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        type=str,   default="data/processed/")
    p.add_argument("--out",         type=str,   default="checkpoints/encoder/")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--latent_dim",  type=int,   default=256)
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--kl_weight",   type=float, default=1e-3,
                   help="KL annealing start weight (increases linearly to 1.0)")
    p.add_argument("--n_layers",    type=int,   default=6)
    p.add_argument("--device",      type=str,   default="cuda" if HAS_TORCH and
                   __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--dry_run",     action="store_true",
                   help="Print model summary and exit without training")
    return p.parse_args()


def train_epoch(
    encoder:       "COFEncoder",
    spec_decoder:  "SpecDecoderMLP",
    loader,
    optimizer,
    device,
    kl_weight: float,
    epoch: int,
) -> dict:
    encoder.train()
    spec_decoder.train()

    total_loss = total_recon = total_kl = 0.0
    n_batches  = 0

    for batch in loader:
        # Move to device
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

        optimizer.zero_grad()

        # Encode
        z, mu, log_var = encoder(batch)

        # KL divergence
        kl = kl_divergence(mu, log_var)

        # Decode: predict spec from z
        spec_logits = spec_decoder(z)
        spec_targets = {
            "linkage":  batch["linkage_idx"],
            "topology": batch["topology_idx"],
            "stacking": batch["stacking_idx"],
            "node":     batch.get("node_bb_idx",
                            torch.zeros_like(batch["linkage_idx"])),
            "linker":   batch.get("linker_bb_idx",
                            torch.zeros_like(batch["linkage_idx"])),
        }
        recon = spec_decoder_loss(spec_logits, spec_targets)

        loss = recon + kl_weight * kl
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(spec_decoder.parameters()), 1.0
        )
        optimizer.step()

        total_loss  += loss.item()
        total_recon += recon.item()
        total_kl    += kl.item()
        n_batches   += 1

    return {
        "loss":  total_loss  / max(n_batches, 1),
        "recon": total_recon / max(n_batches, 1),
        "kl":    total_kl    / max(n_batches, 1),
    }


@torch.no_grad()
def val_epoch(encoder, spec_decoder, loader, device, kl_weight) -> dict:
    encoder.eval()
    spec_decoder.eval()

    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        z, mu, log_var = encoder(batch)
        kl    = kl_divergence(mu, log_var)
        spec_logits = spec_decoder(z)
        spec_targets = {
            "linkage":  batch["linkage_idx"],
            "topology": batch["topology_idx"],
            "stacking": batch["stacking_idx"],
            "node":     batch.get("node_bb_idx",
                            torch.zeros_like(batch["linkage_idx"])),
            "linker":   batch.get("linker_bb_idx",
                            torch.zeros_like(batch["linkage_idx"])),
        }
        recon = spec_decoder_loss(spec_logits, spec_targets)
        total_loss += (recon + kl_weight * kl).item()
        n_batches  += 1

    return {"val_loss": total_loss / max(n_batches, 1)}


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build models ──────────────────────────────────────────────────────────
    encoder     = COFEncoder(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )
    spec_decoder = SpecDecoderMLP(latent_dim=args.latent_dim)

    print(model_summary(encoder))
    n_spec = sum(p.numel() for p in spec_decoder.parameters() if p.requires_grad)
    print(f"SpecDecoder: {n_spec:,} trainable parameters")

    if args.dry_run or not HAS_TORCH:
        print("[dry_run] Architecture check complete. Exiting.")
        return

    device = torch.device(args.device)
    encoder.to(device)
    spec_decoder.to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = COFDataset(args.data, split="train")
    val_ds   = COFDataset(args.data, split="val")

    if len(train_ds) == 0:
        print("[ERROR] No training data found. Run build_dataset.py first.")
        print("  Expected: data/processed/<structure_id>.json files")
        print("  Exiting.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_cof_graphs, num_workers=4, pin_memory=True,
    )
    val_loader   = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_cof_graphs, num_workers=2,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    params    = list(encoder.parameters()) + list(spec_decoder.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val = float("inf")
    history  = []

    for epoch in range(1, args.epochs + 1):
        # KL annealing: linearly increase weight from kl_weight to 1.0
        kl_w = min(1.0, args.kl_weight + (1.0 - args.kl_weight) * (epoch / args.epochs))

        t0    = time.time()
        train_m = train_epoch(encoder, spec_decoder, train_loader, optimizer, device, kl_w, epoch)
        val_m   = val_epoch(encoder, spec_decoder, val_loader, device, kl_w)
        scheduler.step()

        elapsed = time.time() - t0
        log = {
            "epoch": epoch,
            "elapsed": elapsed,
            "kl_weight": kl_w,
            **train_m, **val_m,
        }
        history.append(log)

        print(
            f"Epoch {epoch:4d}/{args.epochs}  "
            f"loss={train_m['loss']:.4f}  recon={train_m['recon']:.4f}  "
            f"kl={train_m['kl']:.4f}  val={val_m['val_loss']:.4f}  "
            f"kl_w={kl_w:.4f}  t={elapsed:.1f}s"
        )

        # Save best
        if val_m["val_loss"] < best_val:
            best_val = val_m["val_loss"]
            torch.save({
                "epoch":       epoch,
                "encoder":     encoder.state_dict(),
                "spec_decoder": spec_decoder.state_dict(),
                "val_loss":    best_val,
                "args":        vars(args),
            }, out_dir / "best.pt")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                "epoch":        epoch,
                "encoder":      encoder.state_dict(),
                "spec_decoder": spec_decoder.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "args":         vars(args),
            }, out_dir / f"epoch_{epoch:04d}.pt")

    # Save history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    if HAS_TORCH:
        import torch
    main()
