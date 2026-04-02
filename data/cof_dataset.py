"""
cof_dataset.py
==============
PyTorch Dataset wrapping a directory of pre-processed CrystalGraph JSON files.

Design choices:
  - Lazy loading: CrystalGraph JSON is read on first access, then cached in RAM
    (or via a memory-mapped index for large datasets like ReDD-COFFEE's 268k)
  - Supports filtering by linkage type, topology, or property range
  - Returns tensors ready for the graph transformer encoder

Usage
-----
    from data.cof_dataset import COFDataset, collate_cof_graphs

    ds = COFDataset("data/processed/", split="train")
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_cof_graphs)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Optional torch import ────────────────────────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Minimal shim so the module is importable without torch
    class Dataset:  # type: ignore
        pass

from data.crystal_graph import CrystalGraph
from utils.featurisation import (
    LINKAGE_TO_IDX, TOPOLOGY_TO_IDX, STACKING_TO_IDX,
    normalise_property, PROPERTY_STATS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Split utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_splits(
    all_names: List[str],
    train_frac: float = 0.90,
    val_frac:   float = 0.05,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Reproducible random train/val/test split."""
    rng = random.Random(seed)
    names = list(all_names)
    rng.shuffle(names)
    n = len(names)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return names[:n_train], names[n_train:n_train + n_val], names[n_train + n_val:]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class COFDataset(Dataset):
    """
    Parameters
    ----------
    processed_dir : Path to directory of CrystalGraph JSON files.
                    Each file is named <structure_id>.json.
    split         : "train", "val", "test", or "all"
    split_file    : Optional path to a JSON dict {"train": [...], "val": [...], "test": [...]}
                    If None, a random split is generated.
    properties    : List of property names to include as labels.
                    Structures without all requested properties are excluded.
    filter_fn     : Optional callable(CrystalGraph) → bool for custom filtering.
    max_atoms     : Drop structures with more than this many atoms (memory guard).
    cache_graphs  : Keep all loaded CrystalGraphs in RAM (fast, costs memory).
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "train",
        split_file: Optional[str | Path] = None,
        properties: Optional[List[str]] = None,
        filter_fn: Optional[Callable[[CrystalGraph], bool]] = None,
        max_atoms: int = 500,
        cache_graphs: bool = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.properties    = properties or []
        self.filter_fn     = filter_fn
        self.max_atoms     = max_atoms
        self.cache_graphs  = cache_graphs
        self._cache: Dict[str, CrystalGraph] = {}

        # ── Gather all available structure ids ───────────────────────────────
        all_paths = sorted(self.processed_dir.glob("*.json"))
        all_names = [p.stem for p in all_paths]

        # ── Load or generate split ───────────────────────────────────────────
        if split_file is not None:
            with open(split_file) as f:
                splits = json.load(f)
            names_in_split = splits[split]
        else:
            train, val, test = make_splits(all_names)
            splits_map = {"train": train, "val": val, "test": test, "all": all_names}
            names_in_split = splits_map[split]

        # ── Filter to names that actually exist ──────────────────────────────
        existing = set(all_names)
        self.names = [n for n in names_in_split if n in existing]

        # ── Apply property filter (quick check via index if available) ───────
        if self.properties:
            self.names = self._filter_by_properties(self.names)

        print(f"[COFDataset] split={split} | {len(self.names):,} structures")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _path(self, name: str) -> Path:
        return self.processed_dir / f"{name}.json"

    def _load(self, name: str) -> CrystalGraph:
        if name in self._cache:
            return self._cache[name]
        graph = CrystalGraph.load(self._path(name))
        if self.cache_graphs:
            self._cache[name] = graph
        return graph

    def _filter_by_properties(self, names: List[str]) -> List[str]:
        """Keep only structures that have all requested property labels."""
        kept = []
        for name in names:
            try:
                graph = self._load(name)
            except Exception:
                continue
            if all(p in graph.properties for p in self.properties):
                if graph.n_atoms <= self.max_atoms:
                    if self.filter_fn is None or self.filter_fn(graph):
                        kept.append(name)
        return kept

    # ── PyTorch Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Dict:
        name  = self.names[idx]
        graph = self._load(name)

        item: Dict = {
            "name":         name,
            "atoms":        graph.atoms,              # (N, ATOM_FEAT_DIM)
            "frac_coords":  graph.frac_coords,        # (N, 3)
            "lattice":      graph.lattice,            # (6,)
            "edge_index":   graph.edge_index,         # (2, E)
            "edge_attr":    graph.edge_attr,          # (E, BOND_FEAT_DIM)
            "edge_shift":   graph.edge_shift,         # (E, 3)
            "bb_index":     graph.bb_index,           # (N,)
            "n_atoms":      graph.n_atoms,
            "n_bbs":        graph.n_building_blocks,
            # Discrete labels
            "linkage_idx":  LINKAGE_TO_IDX.get(graph.linkage_type, 0),
            "topology_idx": TOPOLOGY_TO_IDX.get(graph.topology, 0),
            "stacking_idx": STACKING_TO_IDX.get(graph.stacking, 0),
        }

        # BB identity indices (from pyCOFBuilder filename metadata)
        from decoder.reticular_decoder import BB_LIBRARY
        _all_nodes   = BB_LIBRARY["T3_nodes"] + BB_LIBRARY["S4_nodes"]
        _all_linkers = BB_LIBRARY["L2_linkers"]
        _node_to_idx   = {n: i for i, n in enumerate(_all_nodes)}
        _linker_to_idx = {l: i for i, l in enumerate(_all_linkers)}

        pcb_node   = graph.properties.get("pcb_node_bb",   "")
        pcb_linker = graph.properties.get("pcb_linker_bb", "")
        item["node_bb_idx"]   = _node_to_idx.get(pcb_node,   0)
        item["linker_bb_idx"] = _linker_to_idx.get(pcb_linker, 0)
        item["pcb_node_bb"]   = pcb_node
        item["pcb_linker_bb"] = pcb_linker

        # Continuous property labels — normalised to [0, 1]
        for prop in self.properties:
            raw_val = graph.properties.get(prop, float("nan"))
            item[f"prop_{prop}"] = normalise_property(prop, raw_val)

        if HAS_TORCH:
            item = _to_tensors(item)

        return item

    # ── Convenience ──────────────────────────────────────────────────────────

    def get_graph(self, idx: int) -> CrystalGraph:
        return self._load(self.names[idx])

    def property_stats(self, prop: str) -> Tuple[float, float, float, float]:
        """Returns (mean, std, min, max) over the split for a given property."""
        vals = []
        for name in self.names:
            g = self._load(name)
            if prop in g.properties:
                vals.append(g.properties[prop])
        vals_arr = np.array(vals, dtype=np.float32)
        return (
            float(vals_arr.mean()), float(vals_arr.std()),
            float(vals_arr.min()),  float(vals_arr.max()),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tensor conversion
# ─────────────────────────────────────────────────────────────────────────────

def _to_tensors(item: dict) -> dict:
    """Convert numpy arrays to torch tensors in-place."""
    import torch
    for k, v in item.items():
        if isinstance(v, np.ndarray):
            if v.dtype in (np.float32, np.float64):
                item[k] = torch.from_numpy(v.astype(np.float32))
            elif v.dtype in (np.int32, np.int64):
                item[k] = torch.from_numpy(v.astype(np.int64))
        elif isinstance(v, (int, float)):
            item[k] = torch.tensor(v)
    return item


# ─────────────────────────────────────────────────────────────────────────────
# Batch collation
# ─────────────────────────────────────────────────────────────────────────────

def collate_cof_graphs(batch: List[Dict]) -> Dict:
    """
    Custom collate for variable-size crystal graphs.
    Concatenates node/edge tensors and builds a batch vector.
    Compatible with PyTorch Geometric conventions.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required for collate_cof_graphs")

    import torch

    out: Dict = {}
    keys = batch[0].keys()

    # These get concatenated along the atom/edge dimension
    concat_keys = {"atoms", "frac_coords", "bb_index", "edge_attr", "edge_shift"}
    # These need offset adjustment (edge_index) or are per-graph scalars
    passthrough_keys = {"lattice", "linkage_idx", "topology_idx", "stacking_idx"}

    atom_offset = 0
    batch_vec   = []
    edge_indices = []
    per_graph_lattice = []
    per_graph_scalars: Dict[str, List] = {}

    # Initialise concat buffers
    concat_buffers: Dict[str, List] = {k: [] for k in concat_keys if k in keys}

    for graph_idx, item in enumerate(batch):
        n = item["n_atoms"].item() if HAS_TORCH else int(item["n_atoms"])

        batch_vec.append(torch.full((n,), graph_idx, dtype=torch.long))

        for k in concat_keys:
            if k in item:
                concat_buffers[k].append(item[k])

        # Shift edge indices by atom offset
        ei = item["edge_index"] + atom_offset
        edge_indices.append(ei)

        per_graph_lattice.append(item["lattice"])

        for k in passthrough_keys:
            if k in item and k != "lattice":
                per_graph_scalars.setdefault(k, []).append(item[k])

        # Property labels
        for k, v in item.items():
            if k.startswith("prop_"):
                per_graph_scalars.setdefault(k, []).append(v)

        # name
        per_graph_scalars.setdefault("name", []).append(item.get("name", ""))

        atom_offset += n

    out["batch"]      = torch.cat(batch_vec, dim=0)
    out["edge_index"] = torch.cat(edge_indices, dim=1)
    out["lattice"]    = torch.stack(per_graph_lattice, dim=0)

    for k, bufs in concat_buffers.items():
        out[k] = torch.cat(bufs, dim=0)

    for k, vals in per_graph_scalars.items():
        if k == "name":
            out[k] = vals
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals

    out["n_atoms_per_graph"] = torch.tensor(
        [item["n_atoms"].item() if hasattr(item["n_atoms"], "item") else item["n_atoms"]
         for item in batch], dtype=torch.long
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test (no torch needed)
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test_dataset(processed_dir: str) -> None:
    ds = COFDataset(processed_dir, split="all", cache_graphs=True)
    if len(ds) == 0:
        print("[smoke] No structures found — run build_dataset.py first")
        return
    item = ds[0]
    print(f"[smoke] First item keys: {list(item.keys())}")
    print(f"[smoke] atoms shape: {item['atoms'].shape}")
    print(f"[smoke] edge_index shape: {item['edge_index'].shape}")
    print(f"[smoke] n_bbs: {item['n_bbs']}")


if __name__ == "__main__":
    import sys
    _smoke_test_dataset(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
