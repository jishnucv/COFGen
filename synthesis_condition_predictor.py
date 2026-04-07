"""
synthesis_condition_predictor.py
=================================
Predicts synthesis conditions for a generated COFSpec by retrieving
analogous precedents from a COF synthesis knowledge base (CS-KB).

Inspired by Chen et al. JACS 2026 "Chemist-Guided Human-AI Workflow
for Covalent Organic Framework Synthesis".

Their key findings integrated here:
  - Embedding-based retrieval (not keyword) is essential for good recall
  - Stratified sampling across similarity strata outperforms head-only Top-K
  - Solvent + catalyst jointly determine crystallisation outcome
  - Self-consensus over multiple LLM runs improves hit rate to ~0.83

This module implements:
  1. CS-KB schema for storing synthesis protocols (matches Chen et al. JSON)
  2. Embedding-based similarity search (TF-IDF fallback, sentence-transformers preferred)
  3. Range-type synthesis prior generation (solvent system, catalyst, T, t, stoichiometry)
  4. Failure taxonomy (Class A-D) for diagnosis-driven iteration
  5. COFGen integration: synthesizability score adjusted by CS-KB coverage

Usage
-----
    predictor = SynthesisConditionPredictor.from_kb("data/cs_kb.json")

    spec = COFSpec("imine", "hcb", "AA", "T3_BENZ", "L2_PYRN", "NH2", "CHO")
    prior = predictor.get_prior(spec, strategy="stratified", n_repeats=3)
    print(prior)

    # Adjust synthesizability score by CS-KB coverage
    score = predictor.coverage_score(spec)  # 0-1, 1 = well-covered in literature
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CS-KB schema (matches Chen et al. JSON extraction schema)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SynthesisProtocol:
    """
    One COF synthesis protocol record, matching Chen et al. CS-KB schema.
    Missing fields are None (not imputed).
    """
    # Identifiers
    cof_name:       str
    node_bb:        str                    # e.g. "TAPPy", "TAPB"
    linker_bb:      str                    # e.g. "4F-CHO", "PDA"
    linkage_type:   str                    # e.g. "imine", "boronate_ester"
    topology:       str = "hcb"

    # Synthesis parameters
    solvent_system:     Optional[str]   = None   # e.g. "o-DCB/n-BuOH (1:1)"
    catalyst:           Optional[str]   = None   # e.g. "6M AcOH (0.1 mL)"
    temperature_c:      Optional[float] = None   # °C
    time_days:          Optional[float] = None   # days
    stoichiometry:      Optional[str]   = None   # e.g. "1:2 (node:linker)"
    method:             Optional[str]   = None   # e.g. "solvothermal"

    # Characterisation
    pxrd_peaks_2theta:  Optional[List[float]] = None
    crystallinity:      Optional[str]   = None   # "high", "medium", "low", "amorphous"
    bet_surface_m2g:    Optional[float] = None
    pore_size_nm:       Optional[float] = None

    # Provenance
    doi:            Optional[str] = None
    year:           Optional[int] = None

    def to_text(self) -> str:
        """Compact textual representation for embedding."""
        parts = [
            f"COF: {self.cof_name}",
            f"Node: {self.node_bb}",
            f"Linker: {self.linker_bb}",
            f"Linkage: {self.linkage_type}",
            f"Topology: {self.topology}",
        ]
        if self.solvent_system:  parts.append(f"Solvent: {self.solvent_system}")
        if self.catalyst:        parts.append(f"Catalyst: {self.catalyst}")
        if self.temperature_c:   parts.append(f"Temp: {self.temperature_c}°C")
        if self.time_days:       parts.append(f"Time: {self.time_days}d")
        if self.stoichiometry:   parts.append(f"Stoich: {self.stoichiometry}")
        if self.crystallinity:   parts.append(f"Crystallinity: {self.crystallinity}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @staticmethod
    def from_dict(d: dict) -> "SynthesisProtocol":
        return SynthesisProtocol(**{k: v for k, v in d.items()
                                    if k in SynthesisProtocol.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────────────
# Failure taxonomy (Chen et al. Section 5.3)
# ─────────────────────────────────────────────────────────────────────────────

class FailureClass:
    """
    Chen et al. failure taxonomy for COF synthesis outcomes.
    Used to diagnose failed experiments and guide next-round planning.
    """
    # Class A: Stalled — no solid formed, clear solution
    # Interpretation: solubility issue (linker too soluble, reaction too fast)
    A = "A_stalled"

    # Class B: Solubility trap — gel/oil/amorphous precipitate
    # Interpretation: too rapid condensation, need milder catalyst or different solvent
    B = "B_solubility_trap"

    # Class C: Kinetic trap — microcrystalline but wrong phase or poor crystallinity
    # Interpretation: need error correction (more reversible conditions), longer time
    C = "C_kinetic_trap"

    # Class D: Near-crystalline — weak PXRD peaks, correct d-spacings
    # Interpretation: close to window, refine catalyst loading / temperature
    D = "D_near_crystalline"


FAILURE_CLASS_ACTIONS = {
    FailureClass.A: [
        "Switch to lower-polarity solvent (add mesitylene or o-DCB)",
        "Reduce catalyst concentration by 2×",
        "Try 120°C instead of 100°C (increase driving force)",
    ],
    FailureClass.B: [
        "Switch to TFA or reduce AcOH concentration",
        "Add mesitylene as non-polar co-solvent",
        "Use room-temperature pre-mixing step before heating",
    ],
    FailureClass.C: [
        "Increase reaction time to 5-7 days",
        "Raise temperature to 130°C",
        "Try high-boiling solvent (DMF, NMP) for better error correction",
    ],
    FailureClass.D: [
        "Adjust stoichiometry to 1:1.8 or 1:2.2",
        "Extend time by 2 additional days at same conditions",
        "Re-activate with fresh solvent wash at 60°C",
    ],
}


def diagnose_failure(
    observations: str,
    pxrd_has_peaks: bool = False,
    pxrd_matches_sim: bool = False,
) -> str:
    """
    Rule-based failure diagnosis (heuristic version of Chen et al. LLM diagnosis).
    Returns a FailureClass string.
    """
    obs_lower = observations.lower()

    if "clear" in obs_lower or "solution" in obs_lower or "no solid" in obs_lower:
        return FailureClass.A
    if "gel" in obs_lower or "oil" in obs_lower or "amorphous" in obs_lower:
        return FailureClass.B
    if pxrd_has_peaks and not pxrd_matches_sim:
        return FailureClass.C
    if pxrd_has_peaks and pxrd_matches_sim:
        return FailureClass.D
    # Default: kinetic trap
    return FailureClass.C


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF embedding (fallback when sentence-transformers unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFEmbedder:
    """
    Minimal TF-IDF vectoriser for protocol text similarity.
    Used when sentence-transformers is not installed.
    """
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf:   np.ndarray = np.array([])
        self._fitted = False

    def fit(self, texts: List[str]) -> "TFIDFEmbedder":
        # Build vocabulary
        from collections import Counter
        doc_freq: Counter = Counter()
        tokenised = [t.lower().split() for t in texts]
        for tokens in tokenised:
            for tok in set(tokens):
                doc_freq[tok] += 1
        vocab = {tok: i for i, (tok, _) in enumerate(doc_freq.most_common(2000))}
        self.vocab = vocab
        N = len(texts)
        self.idf = np.array([
            math.log((N + 1) / (doc_freq[tok] + 1)) + 1
            for tok in sorted(vocab, key=vocab.get)
        ], dtype=np.float32)
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        V = len(self.vocab)
        out = np.zeros((len(texts), V), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            from collections import Counter
            tf = Counter(tokens)
            for tok, cnt in tf.items():
                if tok in self.vocab:
                    j = self.vocab[tok]
                    out[i, j] = cnt / len(tokens) * self.idf[j]
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


# ─────────────────────────────────────────────────────────────────────────────
# Range-type synthesis prior
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SynthesisPrior:
    """
    Range-type synthesis prior over crystallisation conditions.
    Matches Chen et al. Module 2 output format.
    """
    solvent_candidates:  List[str]             # ordered by confidence
    catalyst_candidates: List[str]
    temperature_range:   Tuple[float, float]   # (min_C, max_C)
    time_range:          Tuple[float, float]   # (min_days, max_days)
    stoichiometry_range: Tuple[float, float]   # (min ratio, max ratio)
    n_precedents_used:   int  = 0
    confidence:          float = 0.0           # 0-1

    def top_conditions(self) -> Dict:
        """Return the single most likely set of conditions."""
        return {
            "solvent":       self.solvent_candidates[0] if self.solvent_candidates else "o-DCB/n-BuOH (1:1)",
            "catalyst":      self.catalyst_candidates[0] if self.catalyst_candidates else "6M AcOH (0.1 mL)",
            "temperature_c": (self.temperature_range[0] + self.temperature_range[1]) / 2,
            "time_days":     (self.time_range[0] + self.time_range[1]) / 2,
            "stoichiometry": f"1:{(self.stoichiometry_range[0]+self.stoichiometry_range[1])/2:.1f}",
        }

    def to_doe_matrix(self, n_experiments: int = 10) -> List[Dict]:
        """
        Convert prior to a design-of-experiments matrix.
        Round 1: fix temperature/time, vary solvent × catalyst.
        Mirrors Chen et al. Module 3 DoE strategy.
        """
        rng = random.Random(42)
        solvents  = self.solvent_candidates[:4]
        catalysts = self.catalyst_candidates[:3]
        t_mid     = (self.temperature_range[0] + self.temperature_range[1]) / 2
        time_mid  = (self.time_range[0] + self.time_range[1]) / 2

        experiments = []
        for i in range(n_experiments):
            solvent  = rng.choice(solvents)  if solvents  else "o-DCB/n-BuOH (1:1)"
            catalyst = rng.choice(catalysts) if catalysts else "6M AcOH"
            experiments.append({
                "exp_id":      i + 1,
                "solvent":     solvent,
                "catalyst":    catalyst,
                "temperature": t_mid,
                "time_days":   time_mid,
                "stoichiometry": f"1:{rng.uniform(*self.stoichiometry_range):.1f}",
            })
        return experiments


# ─────────────────────────────────────────────────────────────────────────────
# Main predictor
# ─────────────────────────────────────────────────────────────────────────────

class SynthesisConditionPredictor:
    """
    Retrieves synthesis conditions for a COFSpec from a knowledge base.

    Without an LLM (offline mode):
      - Retrieves Top-K similar protocols by embedding similarity
      - Aggregates solvent/catalyst frequencies into a ranked list
      - Returns a range-type SynthesisPrior

    With an LLM (online mode via Anthropic API in artifacts):
      - Uses the retrieved precedents as RAG context
      - Prompts the LLM to generate a consolidated prior (Chen et al. Module 2)
    """

    # Well-known solvent systems from COF literature, ranked by literature frequency
    DEFAULT_SOLVENTS = [
        "o-DCB/n-BuOH (1:1, v/v)",
        "mesitylene/1,4-dioxane (1:1, v/v)",
        "DMF/AcOH",
        "1,4-dioxane/mesitylene (1:1, v/v)",
        "DMSO/AcOH",
        "DMF",
        "acetonitrile/water (9:1)",
    ]
    DEFAULT_CATALYSTS = [
        "6M AcOH (aq), 0.1 mL",
        "3M AcOH (aq), 0.1 mL",
        "1M TFA, 0.2 mL",
        "9M AcOH (aq), 0.1 mL",
        "p-TsOH (10 mol%)",
        "Sc(OTf)3 (5 mol%)",
    ]

    # Linkage-specific condition priors (from COF literature)
    LINKAGE_PRIORS = {
        "imine": {
            "solvents":    ["o-DCB/n-BuOH (1:1, v/v)", "mesitylene/1,4-dioxane (1:1, v/v)"],
            "catalysts":   ["6M AcOH (aq), 0.1 mL", "3M AcOH (aq), 0.1 mL"],
            "temp_range":  (100.0, 130.0),
            "time_range":  (2.0, 5.0),
            "stoich_range": (1.8, 2.2),
        },
        "boronate_ester": {
            "solvents":    ["mesitylene/1,4-dioxane (1:1, v/v)", "DMF/AcOH"],
            "catalysts":   ["no catalyst", "molecular sieves 4Å"],
            "temp_range":  (80.0, 120.0),
            "time_range":  (3.0, 7.0),
            "stoich_range": (0.95, 1.05),
        },
        "beta_ketoenamine": {
            "solvents":    ["o-DCB/n-BuOH (1:1, v/v)", "DMSO/AcOH"],
            "catalysts":   ["6M AcOH (aq), 0.1 mL", "p-TsOH (10 mol%)"],
            "temp_range":  (120.0, 150.0),
            "time_range":  (3.0, 7.0),
            "stoich_range": (2.8, 3.2),
        },
        "triazine": {
            "solvents":    ["ZnCl2 melt (ionothermal)", "P2O5/methanesulfonic acid"],
            "catalysts":   ["no catalyst (ionothermal)", "TfOH"],
            "temp_range":  (300.0, 400.0),
            "time_range":  (2.0, 5.0),
            "stoich_range": (1.0, 1.0),
        },
        "hydrazone": {
            "solvents":    ["DMF/AcOH", "1,4-dioxane/AcOH"],
            "catalysts":   ["AcOH (cat.)", "p-TsOH (5 mol%)"],
            "temp_range":  (100.0, 120.0),
            "time_range":  (3.0, 5.0),
            "stoich_range": (1.8, 2.2),
        },
        "imide": {
            # Imide COF synthesis (PI-COFs): Ben et al. JACS 2009; Fang et al. JACS 2015
            # High temperature solvothermal, no acid catalyst needed (irreversible reaction)
            # Solvent must dissolve both amine and dianhydride at >150°C
            "solvents":    ["DMF (neat)", "NMP (N-methyl-2-pyrrolidone)",
                            "DMAc/isoquinoline (9:1, v/v)", "DMF/isoquinoline (9:1, v/v)"],
            "catalysts":   ["no catalyst", "isoquinoline (1-2 drops, base catalyst)",
                            "acetic acid (glacial, 1 drop)"],
            "temp_range":  (180.0, 220.0),   # much higher than imine COFs
            "time_range":  (24.0, 72.0),     # longer reaction time
            "stoich_range": (0.95, 1.05),    # near-stoichiometric (irreversible)
        },
    }

    def __init__(
        self,
        protocols:  List[SynthesisProtocol] = None,
        embeddings: Optional[np.ndarray] = None,
    ):
        self.protocols  = protocols or []
        self.embeddings = embeddings
        self._embedder  = None

        if protocols and embeddings is None:
            self._build_index()

    def _build_index(self) -> None:
        texts = [p.to_text() for p in self.protocols]
        self._embedder  = TFIDFEmbedder()
        self.embeddings = self._embedder.fit_transform(texts)

    @classmethod
    def from_kb(cls, kb_path: Path) -> "SynthesisConditionPredictor":
        """Load from a JSON file of SynthesisProtocol records."""
        with open(kb_path) as f:
            records = json.load(f)
        protocols = [SynthesisProtocol.from_dict(r) for r in records]
        return cls(protocols=protocols)

    @classmethod
    def from_schema(cls) -> "SynthesisConditionPredictor":
        """
        Create a predictor seeded with known literature protocols for common COFs.
        Used when no CS-KB file is available.
        """
        known = [
            SynthesisProtocol(
                cof_name="TAPPy-4F", node_bb="TAPPy", linker_bb="4F-CHO",
                linkage_type="imine", topology="sql",
                solvent_system="o-DCB/n-BuOH (1:1, v/v)",
                catalyst="6M AcOH (aq), 0.1 mL",
                temperature_c=120.0, time_days=3.0, stoichiometry="1:2",
                crystallinity="high", doi="10.1021/jacs.5c20068", year=2026,
            ),
            SynthesisProtocol(
                cof_name="TAPPy-8F", node_bb="TAPPy", linker_bb="8F-CHO",
                linkage_type="imine", topology="sql",
                solvent_system="o-DCB/n-BuOH (1:1, v/v)",
                catalyst="1M TFA, 0.2 mL",
                temperature_c=120.0, time_days=3.0, stoichiometry="1:2",
                crystallinity="high", doi="10.1021/jacs.5c20068", year=2026,
            ),
            SynthesisProtocol(
                cof_name="COF-LZU1", node_bb="TAPB", linker_bb="PDA",
                linkage_type="imine", topology="hcb",
                solvent_system="o-DCB/n-BuOH (1:1, v/v)",
                catalyst="6M AcOH (aq), 0.1 mL",
                temperature_c=120.0, time_days=3.0, stoichiometry="1:1.5",
                crystallinity="high", year=2011,
            ),
            SynthesisProtocol(
                cof_name="TpPa-1", node_bb="Tp", linker_bb="Pa-1",
                linkage_type="beta_ketoenamine", topology="hcb",
                solvent_system="o-DCB/n-BuOH (1:1, v/v)",
                catalyst="6M AcOH (aq), 0.1 mL",
                temperature_c=120.0, time_days=3.0, stoichiometry="1:3",
                crystallinity="high", year=2012,
            ),
            SynthesisProtocol(
                cof_name="COF-1", node_bb="BDBA", linker_bb="BDBA",
                linkage_type="boroxine", topology="hxl",
                solvent_system="mesitylene (neat)",
                catalyst="no catalyst",
                temperature_c=120.0, time_days=3.0, stoichiometry="1:1",
                crystallinity="high", year=2005,
            ),
        ]
        return cls(protocols=known)

    def retrieve_top_k(
        self,
        spec: "COFSpec",
        k: int = 20,
        strategy: str = "stratified",
    ) -> List[Tuple[SynthesisProtocol, float]]:
        """
        Retrieve Top-K protocols similar to the given COFSpec.
        Returns (protocol, similarity_score) pairs.

        Strategies (Chen et al. strategies A-D):
          "head"       : Top-K by similarity (strategy A)
          "shuffled"   : Top-K shuffled (strategy B)
          "stratified" : Stratified sampling across similarity strata (strategy C — default)
          "tail"       : Tail-weighted stratified (strategy D)
        """
        if not self.protocols:
            return []

        # Build query text from COFSpec
        query_text = (f"Node: {spec.node_bb} Linker: {spec.linker_bb} "
                      f"Linkage: {spec.linkage_type} Topology: {spec.topology}")

        if self._embedder and self.embeddings is not None:
            query_emb = self._embedder.transform([query_text])
            sims      = (self.embeddings @ query_emb.T).flatten()
        else:
            # Keyword fallback
            query_tokens = set(query_text.lower().split())
            sims = np.array([
                len(query_tokens & set(p.to_text().lower().split())) / max(len(query_tokens), 1)
                for p in self.protocols
            ])

        ranked_idx = np.argsort(sims)[::-1]

        if strategy == "head":
            idx_chosen = ranked_idx[:k]
        elif strategy == "shuffled":
            idx_chosen = ranked_idx[:k]
            np.random.shuffle(idx_chosen)
        elif strategy == "stratified":
            # Split ranked list into 3 strata, sample from each
            n = len(ranked_idx)
            strata = [ranked_idx[:n//3], ranked_idx[n//3:2*n//3], ranked_idx[2*n//3:]]
            k_per = max(1, k // 3)
            idx_chosen = np.concatenate([
                s[np.random.choice(len(s), min(k_per, len(s)), replace=False)]
                for s in strata if len(s) > 0
            ])
        elif strategy == "tail":
            # Over-sample from mid/tail strata
            n = len(ranked_idx)
            head  = ranked_idx[:n//4]
            tail  = ranked_idx[n//4:]
            k_head = max(1, k // 4)
            k_tail = k - k_head
            idx_h  = head[np.random.choice(len(head), min(k_head, len(head)), replace=False)]
            idx_t  = tail[np.random.choice(len(tail), min(k_tail, len(tail)), replace=False)]
            idx_chosen = np.concatenate([idx_h, idx_t])
        else:
            idx_chosen = ranked_idx[:k]

        return [(self.protocols[i], float(sims[i])) for i in idx_chosen]

    def get_prior(
        self,
        spec: "COFSpec",
        strategy: str = "stratified",
        k: int = 20,
        n_repeats: int = 3,
    ) -> SynthesisPrior:
        """
        Generate a range-type synthesis prior for a COFSpec.
        Aggregates retrieved precedents + linkage-specific defaults.
        n_repeats > 1 implements self-consensus (Chen et al. Module 2).
        """
        from collections import Counter

        # Retrieve precedents
        precedents = self.retrieve_top_k(spec, k=k, strategy=strategy)

        # Also use linkage-specific literature knowledge
        linkage_defaults = self.LINKAGE_PRIORS.get(
            spec.linkage_type, self.LINKAGE_PRIORS.get("imine", {})
        )

        # Aggregate solvents and catalysts
        solvent_counts:  Counter = Counter()
        catalyst_counts: Counter = Counter()
        temps:  List[float] = []
        times:  List[float] = []
        stoichs: List[float] = []

        # From retrieved precedents
        for proto, sim in precedents:
            if proto.solvent_system:
                solvent_counts[proto.solvent_system] += sim
            if proto.catalyst:
                catalyst_counts[proto.catalyst] += sim
            if proto.temperature_c:
                temps.append(proto.temperature_c)
            if proto.time_days:
                times.append(proto.time_days)

        # Self-consensus: repeat and average (simulates multiple LLM runs)
        for _ in range(n_repeats - 1):
            precs2 = self.retrieve_top_k(spec, k=k, strategy=strategy)
            for proto, sim in precs2:
                if proto.solvent_system:
                    solvent_counts[proto.solvent_system] += sim * 0.5
                if proto.catalyst:
                    catalyst_counts[proto.catalyst] += sim * 0.5

        # Add linkage defaults as soft prior
        for s in linkage_defaults.get("solvents", []):
            solvent_counts[s] += 0.3
        for c in linkage_defaults.get("catalysts", []):
            catalyst_counts[c] += 0.3

        # Build ranked lists
        solvents  = [s for s, _ in solvent_counts.most_common(5)]
        catalysts = [c for c, _ in catalyst_counts.most_common(4)]

        # Fallback to defaults if no precedents
        if not solvents:
            solvents  = linkage_defaults.get("solvents", self.DEFAULT_SOLVENTS[:3])
        if not catalysts:
            catalysts = linkage_defaults.get("catalysts", self.DEFAULT_CATALYSTS[:2])

        # Temperature and time ranges
        t_default = linkage_defaults.get("temp_range", (100.0, 130.0))
        time_default = linkage_defaults.get("time_range", (2.0, 5.0))
        stoich_default = linkage_defaults.get("stoich_range", (1.8, 2.2))

        temp_range  = (min(temps + [t_default[0]]), max(temps + [t_default[1]])) if temps else t_default
        time_range  = (min(times + [time_default[0]]), max(times + [time_default[1]])) if times else time_default
        stoich_range = stoich_default

        confidence = min(1.0, len(precedents) / 10.0)

        return SynthesisPrior(
            solvent_candidates  = solvents,
            catalyst_candidates = catalysts,
            temperature_range   = temp_range,
            time_range          = time_range,
            stoichiometry_range = stoich_range,
            n_precedents_used   = len(precedents),
            confidence          = confidence,
        )

    def coverage_score(self, spec: "COFSpec") -> float:
        """
        Returns CS-KB coverage score for this linkage/topology/BB combination.
        Used to adjust synthesizability score in COFGen.
        High score = many precedents in literature = easier to synthesise.
        """
        if not self.protocols:
            # No KB: use linkage-based heuristic
            easy = {"imine", "boronate_ester", "beta_ketoenamine"}
            return 0.8 if spec.linkage_type in easy else 0.4

        precedents = self.retrieve_top_k(spec, k=10, strategy="head")
        if not precedents:
            return 0.2

        # Score based on mean similarity and number of crystalline precedents
        mean_sim     = float(np.mean([s for _, s in precedents]))
        n_crystalline = sum(1 for p, _ in precedents
                           if p.crystallinity in ("high", "medium"))
        return float(np.clip(mean_sim * 0.5 + n_crystalline / 10.0 * 0.5, 0, 1))

    def adjusted_synth_score(self, spec: "COFSpec", base_score: float) -> float:
        """
        Blend rule-based synthesizability score with CS-KB coverage.
        """
        coverage = self.coverage_score(spec)
        # 60% rule-based, 40% literature coverage
        return float(np.clip(0.6 * base_score + 0.4 * coverage, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Build a minimal CS-KB from existing processed dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_kb_from_processed(
    processed_dir: Path,
    out_path: Path,
) -> None:
    """
    Build a CS-KB JSON from processed CrystalGraph files.
    Extracts metadata from pyCOFBuilder filenames.
    """
    from data.crystal_graph import CrystalGraph

    records = []
    for p in sorted(processed_dir.glob("*.json")):
        try:
            cg   = CrystalGraph.load(p)
            prop = cg.properties
            rec  = SynthesisProtocol(
                cof_name     = cg.name,
                node_bb      = prop.get("pcb_node_bb",    ""),
                linker_bb    = prop.get("pcb_linker_bb",  ""),
                linkage_type = cg.linkage_type,
                topology     = cg.topology,
                bet_surface_m2g = prop.get("bet_surface_area"),
                pore_size_nm    = prop.get("pore_limiting_diameter"),
                crystallinity   = "synthetic",
            )
            records.append(rec.to_dict())
        except Exception:
            continue

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"CS-KB written: {len(records)} records → {out_path}")
