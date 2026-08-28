"""
RSWA (REM Sleep Without Atonia) per-epoch detector.

Phase 3 baseline first (logistic regression / SVM on EMG+EEG features),
same "classical baseline before deep model" discipline as
parkinsons-eeg-classifier. Only add a sequence model (and only then
consider the dt-aware block from mamba-plasticc-transients) if the baseline
demonstrably isn't enough -- see docs/technical_premise.md for why the
dt-aware block is NOT a default-on choice here.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class RSWABaselineDetector:
    """
    StandardScaler -> LogisticRegression on hand-crafted EMG/EEG features.
    This is the floor to beat, analogous to the SVM baseline in
    parkinsons-eeg-classifier -- don't skip straight to a deep model.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RSWABaselineDetector":
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)[:, 1]


# Phase 4 placeholder — NOT wired in by default (config.yaml:
# rswa_detector.use_dt_aware_block defaults to false). Only enable after
# running the same head-to-head protocol as mamba-plasticc-transients
# Phase 4: identical training recipe, multiple seeds, dt_obs clamped before
# training. If it doesn't beat the plain-Mamba / baseline head here either,
# report that honestly in this file's docstring and in the README, the way
# the other two repos report their negative/mixed results.
class RSWADtAwareSequenceModel:
    def __init__(self) -> None:
        raise NotImplementedError(
            "Phase 4 — do not implement before reading docs/technical_premise.md. "
            "The dt-aware block underperformed plain Mamba on mamba-plasticc-transients; "
            "this needs a fair, honestly-reported comparison, not an assumed win."
        )
