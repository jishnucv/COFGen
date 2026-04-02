"""
train_adapter.py
================
Fine-tune a property adapter on top of a frozen base flow matching model.

Usage:
    python scripts/train_adapter.py \
        --base_ckpt   checkpoints/flowmatch_base/best.pt \
        --encoder_ckpt checkpoints/encoder/best.pt \
        --property co2_uptake_298k_1bar \
        --out checkpoints/adapter_co2/ \
        --epochs 50

    # Multi-property adapter
    python scripts/train_adapter.py \
        --base_ckpt   checkpoints/flowmatch_base/best.pt \
        --encoder_ckpt checkpoints/encoder/best.pt \
        --property co2_uptake_298k_1bar bet_surface_area \
        --linkage --topology \
        --out checkpoints/adapter_multi/
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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from data.cof_dataset     import COFDataset, collate_cof_graphs
from models.encoder       import COFEncoder
from models.flow_matching import FlowMatchingNetwork
from models.adapters      import build_adapter, cfm_loss_with_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt",    type=str, required=True)
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--data",         type=str, default="data/processed/")
    p.add_argument("--out",          type=str, default="checkpoints/adapter/")
    p.add_argument("--property",     nargs="+", default=[],
                   help="Scalar property names to condition on")
    p.add_argument("--linkage",      action="store_true")
    p.add_argument("--topology",     action="store_true")
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--cfg_drop",     type=float, default=0.15)
    p.add_argument("--device", type=str,
                   default="cuda" if HAS_TORCH and
                   __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_TORCH:
        print("[dry_run] torch unavailable.")
        return

    device = torch.device(args.device)

    # ── Load and freeze encoder ───────────────────────────────────────────────
    enc_ckpt = torch.load(args.encoder_ckpt, map_location=device)
    enc_args = enc_ckpt.get("args", {})
    encoder  = COFEncoder(
        latent_dim=enc_args.get("latent_dim", 256),
        hidden_dim=enc_args.get("hidden_dim", 256),
    )
    encoder.load_state_dict(enc_ckpt["encoder"])
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ── Load and freeze base flow model ───────────────────────────────────────
    base_ckpt = torch.load(args.base_ckpt, map_location=device)
    base_args = base_ckpt.get("args", {})
    flow_net  = FlowMatchingNetwork(
        latent_dim=base_args.get("latent_dim", 256),
        hidden_dim=base_args.get("hidden_dim", 512),
        n_layers=base_args.get("n_layers", 8),
    )
    flow_net.load_state_dict(base_ckpt["flow_net"])
    flow_net.to(device).eval()
    for p in flow_net.parameters():
        p.requires_grad_(False)

    # ── Build adapter ─────────────────────────────────────────────────────────
    adapter_config = {
        "scalar_properties": args.property,
        "linkage":  args.linkage,
        "topology": args.topology,
        "hidden_dim": base_args.get("hidden_dim", 512),
    }
    adapter = build_adapter(adapter_config)
    adapter.to(device)
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Adapter: {n_params:,} trainable parameters")
    print(f"Conditioning on: {args.property}"
          + (" + linkage" if args.linkage else "")
          + (" + topology" if args.topology else ""))

    if args.dry_run:
        return

    # ── Data (only structures with all required properties) ───────────────────
    train_ds = COFDataset(args.data, split="train", properties=args.property)
    val_ds   = COFDataset(args.data, split="val",   properties=args.property)

    if len(train_ds) == 0:
        print(f"[ERROR] No structures found with properties: {args.property}")
        return

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_cof_graphs, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_cof_graphs, num_workers=2)

    optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val = float("inf")
    history  = []

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v
                     for k, v in batch.items()}

            with torch.no_grad():
                z_1, mu, _ = encoder(batch)
                z_1 = mu

            optimizer.zero_grad()
            loss = cfm_loss_with_adapter(
                flow_net, adapter, z_1, batch,
                cfg_drop_prob=args.cfg_drop,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        # Validation
        adapter.eval()
        val_loss = 0.0
        val_n    = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if hasattr(v, "to") else v
                         for k, v in batch.items()}
                z_1, mu, _ = encoder(batch)
                loss = cfm_loss_with_adapter(flow_net, adapter, mu, batch)
                val_loss += loss.item()
                val_n    += 1

        train_l = total_loss / max(n_batches, 1)
        val_l   = val_loss   / max(val_n, 1)
        log     = {"epoch": epoch, "train": train_l, "val": val_l}
        history.append(log)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_l:.5f}  val={val_l:.5f}")

        if val_l < best_val:
            best_val = val_l
            torch.save({
                "epoch":   epoch,
                "adapter": adapter.state_dict(),
                "config":  adapter_config,
                "val_loss": best_val,
            }, out_dir / "best.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nAdapter training done. Best val: {best_val:.5f}")


if __name__ == "__main__":
    if HAS_TORCH:
        import torch
    main()
