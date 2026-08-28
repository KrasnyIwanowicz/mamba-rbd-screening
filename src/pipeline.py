"""
End-to-end pipeline: full-night EEG (+EMG for training / +IMU for deployment)
-> sleep stages -> REM epochs -> RSWA score -> night-level risk score.

This file intentionally stays thin — it orchestrates the already-tested
pieces (Phases 2 and 3/5) rather than containing new logic itself, same
pattern as mamba-plasticc-transients/src/train.py orchestrating its models/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def run_full_night_pipeline(
    eeg: np.ndarray,
    fs: float,
    sleep_stager,  # loaded from external/sleep_staging submodule, Phase 2
    rswa_detector,  # from src.models.rswa_detector, Phase 3
) -> dict:
    """
    Returns a dict with:
      - hypnogram: per-epoch stage sequence
      - rem_epoch_indices: which epochs were REM
      - rswa_fraction: fraction of REM epochs flagged as atonia-lost
      - risk_score: night-level RBD risk score (currently == rswa_fraction;
        Phase 6 should validate whether a more informative aggregation,
        e.g. weighting by REM epoch duration or phasic/tonic distinction,
        actually improves subject-level classification before adding it)

    NOTE: risk_score is a proxy, not a diagnosis — see README framing.
    """
    raise NotImplementedError(
        "Wire this up once Phase 2 (sleep stager transfer) and Phase 3 "
        "(RSWA baseline) each independently work — don't build the "
        "end-to-end chain before its parts are individually validated."
    )
