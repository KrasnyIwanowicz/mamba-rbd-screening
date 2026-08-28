from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut


@dataclass
class RSWAEpochSample:
    features: np.ndarray  # e.g. [emg_rms_mean, emg_rms_std, eeg_theta_power, eeg_beta_power]
    label: int  # 1 = atonia lost (RSWA), 0 = atonia maintained
    subject_id: str
    group: str  # "rbd" or "n" -- kept for subject-level aggregation in evaluate.py


def leave_one_subject_out_splits(samples: list[RSWAEpochSample]):
    X = np.arange(len(samples))
    groups = [s.subject_id for s in samples]
    logo = LeaveOneGroupOut()
    yield from logo.split(X, groups=groups)
