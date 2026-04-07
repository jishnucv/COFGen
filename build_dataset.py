"""
build_dataset.py
================
Convert a directory of raw CIF files into processed CrystalGraph JSON files.

Usage:
    python scripts/build_dataset.py \
        --raw_dir data/raw/ \
        --out_dir data/processed/ \
        --n_jobs 16 \
        --cutoff 5.0

Supports:
  - ReDD-COFFEE (pyCOFBuilder naming convention → metadata auto-extracted)
  - CoRE-COF    (experimental, linkage/topology from literature annotations)
  - CURATED-COFs
  - Any directory of .cif files

Outputs one <name>.json per structure. Skips structures that fail to parse.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crystal_graph import cif_to_crystal_graph, CrystalGraph


def process_one(
    cif_path: Path,
    out_dir: Path,
    cutoff: float,
    prop_file: Optional[Path],
) -> bool:
    """
    Process a single CIF and write its JSON to out_dir.
    Returns True on success.
    """
    out_path = out_dir / f"{cif_path.stem}.json"
    if out_path.exists():
        return True  # already done

    # Load optional property labels
    properties = {}
    if prop_file and prop_file.exists():
        with open(prop_file) as f:
            all_props = json.load(f)
        properties = all_props.get(cif_path.stem, {})

    try:
        graph = cif_to_crystal_graph(
            cif_path,
            cutoff=cutoff,
            properties=properties,
        )
        graph.save(out_path)
        return True
    except Exception as e:
        print(f"  [SKIP] {cif_path.name}: {e}")
        return False


def process_directory(
    raw_dir: Path,
    out_dir: Path,
    cutoff: float,
    n_jobs: int,
    prop_file: Optional[Path],
) -> None:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cif_paths = sorted(raw_dir.rglob("*.cif"))
    print(f"Found {len(cif_paths):,} CIF files in {raw_dir}")

    if n_jobs == 1:
        ok = 0
        t0 = time.time()
        for i, p in enumerate(cif_paths):
            success = process_one(p, out_dir, cutoff, prop_file)
            if success:
                ok += 1
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta  = (len(cif_paths) - i - 1) / rate
                print(f"  {i+1:,}/{len(cif_paths):,}  ok={ok:,}  "
                      f"{rate:.0f}/s  ETA {eta/60:.1f}min")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import functools
        fn = functools.partial(process_one, out_dir=out_dir, cutoff=cutoff, prop_file=prop_file)
        ok = 0
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(fn, p): p for p in cif_paths}
            for i, future in enumerate(as_completed(futures)):
                if future.result():
                    ok += 1
                if (i + 1) % 1000 == 0:
                    print(f"  {i+1:,}/{len(cif_paths):,}  ok={ok:,}")

    print(f"\nDone: {ok:,}/{len(cif_paths):,} structures processed → {out_dir}")

    # Write split file
    all_names = [p.stem for p in out_dir.glob("*.json")]
    import random
    rng = random.Random(42)
    rng.shuffle(all_names)
    n = len(all_names)
    n_train = int(n * 0.90)
    n_val   = int(n * 0.05)
    splits = {
        "train": all_names[:n_train],
        "val":   all_names[n_train:n_train + n_val],
        "test":  all_names[n_train + n_val:],
    }
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f)
    print(f"Split file written: train={len(splits['train']):,} "
          f"val={len(splits['val']):,} test={len(splits['test']):,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",   type=str, required=True)
    p.add_argument("--out_dir",   type=str, default="data/processed/")
    p.add_argument("--cutoff",    type=float, default=5.0)
    p.add_argument("--n_jobs",    type=int, default=1)
    p.add_argument("--prop_file", type=str, default=None,
                   help="JSON file mapping structure stem → property dict")
    args = p.parse_args()
    process_directory(
        Path(args.raw_dir), Path(args.out_dir),
        args.cutoff, args.n_jobs,
        Path(args.prop_file) if args.prop_file else None,
    )


if __name__ == "__main__":
    main()
