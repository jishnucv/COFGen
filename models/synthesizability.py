"""
synthesizability.py
===================
Binary classifier: P(experimental | COF spec + structure)

Trained on:
  positive  — experimental CoRE-COF / CURATED structures
  negative  — hypothetical ReDD-COFFEE structures not yet synthesised

Used at inference time to rank generated candidates by synthesizability.

Features used (no torch required for inference — sklearn compatible):
  - Linkage type (one-hot)
  - Topology (one-hot)
  - Stacking pattern (one-hot)
  - Void fraction
  - Pore limiting diameter
  - BET surface area
  - n_atoms (proxy for building block complexity)
  - n_building_blocks
  - Node BB identity (one-hot over known BBs)
  - Linker BB identity (one-hot over known BBs)

The full model (COFSynthClassifier) is a gradient-boosted tree or MLP
depending on what's installed. Falls back to rule-based scoring.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.featurisation import (
    N_LINKAGE_TYPES, N_TOPOLOGIES, N_STACKING,
    LINKAGE_TO_IDX, TOPOLOGY_TO_IDX, STACKING_TO_IDX,
    LINKAGE_TYPES, ALL_TOPOLOGIES, STACKING_PATTERNS,
)
from decoder.reticular_decoder import COFSpec, BB_LIBRARY

# ── Feature engineering ───────────────────────────────────────────────────────

ALL_NODE_BBs   = BB_LIBRARY["T3_nodes"] + BB_LIBRARY["S4_nodes"]
ALL_LINKER_BBs = BB_LIBRARY["L2_linkers"]
NODE_TO_IDX    = {n: i for i, n in enumerate(ALL_NODE_BBs)}
LINKER_TO_IDX  = {l: i for i, l in enumerate(ALL_LINKER_BBs)}

FEATURE_DIM = (
    N_LINKAGE_TYPES           # linkage one-hot
    + N_TOPOLOGIES            # topology one-hot
    + N_STACKING              # stacking one-hot
    + 5                       # void_frac, pld, lcd, bet, n_atoms (normalised)
    + len(ALL_NODE_BBs)       # node BB one-hot
    + len(ALL_LINKER_BBs)     # linker BB one-hot
)


def spec_to_features(
    spec:       COFSpec,
    properties: Optional[Dict[str, float]] = None,
    n_atoms:    int = 80,
) -> np.ndarray:
    """
    Encode a COFSpec + property dict into a fixed-length feature vector.
    """
    properties = properties or {}
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    idx  = 0

    # Linkage
    lk = LINKAGE_TO_IDX.get(spec.linkage_type, 0)
    feat[idx + lk] = 1.0
    idx += N_LINKAGE_TYPES

    # Topology
    tp = TOPOLOGY_TO_IDX.get(spec.topology, 0)
    feat[idx + tp] = 1.0
    idx += N_TOPOLOGIES

    # Stacking
    st = STACKING_TO_IDX.get(spec.stacking, 0)
    feat[idx + st] = 1.0
    idx += N_STACKING

    # Scalar properties (normalised to [0, 1] approximately)
    feat[idx + 0] = float(np.clip(properties.get("void_fraction", 0.4), 0, 1))
    feat[idx + 1] = float(np.clip(properties.get("pore_limiting_diameter", 10) / 30, 0, 1))
    feat[idx + 2] = float(np.clip(properties.get("largest_cavity_diameter", 15) / 50, 0, 1))
    feat[idx + 3] = float(np.clip(properties.get("bet_surface_area", 1000) / 8000, 0, 1))
    feat[idx + 4] = float(np.clip(n_atoms / 200, 0, 1))
    idx += 5

    # Node BB
    nb = NODE_TO_IDX.get(spec.node_bb, 0)
    feat[idx + nb] = 1.0
    idx += len(ALL_NODE_BBs)

    # Linker BB
    lb = LINKER_TO_IDX.get(spec.linker_bb, 0)
    feat[idx + lb] = 1.0
    idx += len(ALL_LINKER_BBs)

    assert idx == FEATURE_DIM
    return feat


def specs_to_feature_matrix(
    specs:      List[COFSpec],
    properties: Optional[List[Dict]] = None,
    n_atoms:    Optional[List[int]]  = None,
) -> np.ndarray:
    """Build (N, FEATURE_DIM) matrix for a list of COFSpecs."""
    props   = properties or [{}] * len(specs)
    atoms_l = n_atoms    or [80] * len(specs)
    return np.stack([
        spec_to_features(s, p, n)
        for s, p, n in zip(specs, props, atoms_l)
    ], axis=0)


# ── Rule-based baseline ───────────────────────────────────────────────────────

def rule_based_synth_score(spec: COFSpec) -> float:
    """
    Fast rule-based synthesizability — used when no trained classifier available.
    Mirrors validity_checker.synthesizability_score but on a COFSpec object.
    """
    from decoder.validity_checker import synthesizability_score
    return synthesizability_score(
        spec.linkage_type, spec.node_bb, spec.linker_bb, spec.topology
    )


# ── Sklearn-based classifier ──────────────────────────────────────────────────

class COFSynthClassifier:
    """
    Wrapper around a sklearn GradientBoostingClassifier or MLP.
    Predicts P(synthesisable | features).

    Training
    --------
    Labels:
      1 = experimentally synthesised (from CoRE-COF / CURATED)
      0 = hypothetical only (from ReDD-COFFEE, not in experimental DB)

    Usage
    -----
        clf = COFSynthClassifier()
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)  # shape (N,)
        clf.save("checkpoints/synth_clf.pkl")
    """

    def __init__(self, model_type: str = "gbdt"):
        self.model_type = model_type
        self.model      = None
        self._fitted    = False

    def _build_model(self):
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
        except ImportError:
            pass
        try:
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu", max_iter=500, random_state=42,
            )
        except ImportError:
            return None

    def fit(
        self,
        X:         np.ndarray,   # (N, FEATURE_DIM)
        y:         np.ndarray,   # (N,) binary {0, 1}
        eval_set:  Optional[Tuple] = None,
    ) -> "COFSynthClassifier":
        self.model = self._build_model()
        if self.model is None:
            raise RuntimeError("sklearn not installed — cannot train classifier")
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(synthesisable) for each sample."""
        if not self._fitted or self.model is None:
            # Fall back to zeros (caller should use rule_based_synth_score)
            return np.zeros(len(X), dtype=np.float32)
        proba = self.model.predict_proba(X)
        return proba[:, 1].astype(np.float32)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self._fitted or self.model is None:
            return 0.0
        return float(self.model.score(X, y))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "type": self.model_type}, f)

    @classmethod
    def load(cls, path: Path) -> "COFSynthClassifier":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(model_type=state.get("type", "gbdt"))
        obj.model    = state["model"]
        obj._fitted  = True
        return obj


# ── Training data builder ────────────────────────────────────────────────────

def build_training_data(
    experimental_dir:  Path,   # CoRE-COF / CURATED processed JSONs
    hypothetical_dir:  Path,   # ReDD-COFFEE processed JSONs
    max_hypothetical:  int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) training arrays for the synthesizability classifier.

    Returns (X: (N, FEATURE_DIM), y: (N,))
    """
    from data.crystal_graph import CrystalGraph
    from decoder.reticular_decoder import COFSpec

    rows_X = []
    rows_y = []

    # Positive examples
    for path in sorted(experimental_dir.glob("*.json")):
        try:
            cg = CrystalGraph.load(path)
            spec = COFSpec(
                linkage_type = cg.linkage_type,
                topology     = cg.topology,
                stacking     = cg.stacking,
                node_bb      = cg.bb_smiles[0] if cg.bb_smiles else "T3_BENZ",
                linker_bb    = cg.bb_smiles[1] if len(cg.bb_smiles) > 1 else "L2_BENZ",
                node_func    = "NH2",
                linker_func  = "CHO",
            )
            feat = spec_to_features(spec, cg.properties, cg.n_atoms)
            rows_X.append(feat)
            rows_y.append(1)
        except Exception:
            continue

    n_pos = len(rows_y)
    print(f"Positive (experimental): {n_pos}")

    # Negative examples (balanced)
    hyp_paths = sorted(hypothetical_dir.glob("*.json"))
    import random
    random.shuffle(hyp_paths)
    n_neg_target = min(max_hypothetical, max(n_pos * 10, 10000))

    for path in hyp_paths[:n_neg_target]:
        try:
            cg = CrystalGraph.load(path)
            spec = COFSpec(
                linkage_type = cg.linkage_type,
                topology     = cg.topology,
                stacking     = cg.stacking,
                node_bb      = cg.bb_smiles[0] if cg.bb_smiles else "T3_BENZ",
                linker_bb    = cg.bb_smiles[1] if len(cg.bb_smiles) > 1 else "L2_BENZ",
                node_func    = "NH2",
                linker_func  = "CHO",
            )
            feat = spec_to_features(spec, cg.properties, cg.n_atoms)
            rows_X.append(feat)
            rows_y.append(0)
        except Exception:
            continue

    print(f"Negative (hypothetical): {len(rows_y) - n_pos}")

    X = np.stack(rows_X, axis=0)
    y = np.array(rows_y, dtype=np.int64)
    return X, y


def train_classifier(
    experimental_dir: Path,
    hypothetical_dir: Path,
    out_path:         Path,
    test_frac:        float = 0.1,
) -> float:
    """Full training pipeline. Returns test accuracy."""
    X, y = build_training_data(experimental_dir, hypothetical_dir)

    # Train/test split
    n = len(y)
    idx = np.random.permutation(n)
    n_test = max(1, int(n * test_frac))
    X_test, y_test   = X[idx[:n_test]], y[idx[:n_test]]
    X_train, y_train = X[idx[n_test:]], y[idx[n_test:]]

    clf = COFSynthClassifier()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Synthesizability classifier test accuracy: {acc:.3f}")
    clf.save(out_path)
    return acc
