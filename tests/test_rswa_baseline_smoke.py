"""Synthetic-data test for the RSWA baseline classifier (Phase 3 floor)."""
import numpy as np

from src.models.rswa_detector import RSWABaselineDetector


def test_baseline_learns_separable_synthetic_classes():
    rng = np.random.default_rng(42)
    n = 200
    # class 0 (atonia maintained): low EMG RMS feature
    # class 1 (atonia lost): high EMG RMS feature
    X0 = rng.normal(loc=0.0, scale=0.5, size=(n // 2, 4))
    X1 = rng.normal(loc=3.0, scale=0.5, size=(n // 2, 4))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    detector = RSWABaselineDetector().fit(X, y)
    proba = detector.predict_proba(X)

    # On well-separated synthetic data the baseline should do clearly
    # better than chance -- this only tests plumbing, not real performance,
    # which is unknown until Phase 3 runs on actual CAP data.
    preds = (proba > 0.5).astype(int)
    accuracy = (preds == y).mean()
    assert accuracy > 0.9
