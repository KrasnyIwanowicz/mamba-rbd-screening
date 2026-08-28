
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EMG_CHANNEL_CANDIDATES = ["EMG1", "EMG1-EMG2", "Chin1-Chin2", "EMG Chin"]

@dataclass
class PSGRecording:
    subject_id: str
    group: str  # "rbd" or "n" (normal control), per config.yaml groups_include
    eeg: np.ndarray  # (n_samples,) or (n_channels, n_samples)
    emg: np.ndarray | None  # (n_samples,), None if no EMG channel found
    sample_rate_hz: float
    hypnogram: np.ndarray | None  # per-30s-epoch stage labels, None until Phase 2 runs the stager


def list_subjects(cap_root: Path, groups_include: list[str]) -> list[str]:
    raise NotImplementedError(
        "Implement after Phase 1 data audit — see docs/technical_premise.md "
        "and data/README.md for what needs verifying first."
    )


def load_recording(subject_id: str, cap_root: Path) -> PSGRecording:

    raise NotImplementedError("Implement after Phase 1 channel inventory.")
