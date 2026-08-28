from __future__ import annotations

from pathlib import Path

import numpy as np


def run_full_night_pipeline(
    eeg: np.ndarray,
    fs: float,
    sleep_stager,  # loaded from external/sleep_staging submodule, Phase 2
    rswa_detector,  # from src.models.rswa_detector, Phase 3
) -> dict:
    raise NotImplementedError(
        "Wire this up once Phase 2 (sleep stager transfer) and Phase 3 "
        "(RSWA baseline) each independently work — don't build the "
        "end-to-end chain before its parts are individually validated."
    )
