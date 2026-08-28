"""
CAP Sleep Database loading.

Structure as actually shipped (verify against the archive itself, don't
assume -- see docs/technical_premise.md and data/README.md): CAP is a
multi-lab archive spanning many years, so unlike Sleep-EDF-20's single
protocol, montage and annotation format vary by subject. This loader must
be defensive about missing channels rather than assuming a fixed layout.

TODO (Phase 1): once the archive is downloaded, replace the placeholder
CHANNEL candidates below with a verified per-subject inventory (write that
inventory to docs/ as its own artifact, same as the PD project documented
ds002778's real structure).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Candidate submental EMG channel names seen across CAP recordings.
# MUST be verified per-recording before trusting any downstream RSWA label --
# do not assume the first match is correct without checking the actual signal
# (chin EMG has a distinctive high-frequency, low-amplitude signature; a
# mislabeled channel will silently poison every downstream result).
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
    """
    Return subject IDs present under cap_root matching the requested groups.

    TODO (Phase 1): implement against the actual downloaded archive layout.
    CAP subject folders are typically named like 'rbd1', 'rbd2', ..., 'n1', 'n2'
    -- confirm this naming convention holds for the full download before
    relying on string prefix matching.
    """
    raise NotImplementedError(
        "Implement after Phase 1 data audit — see docs/technical_premise.md "
        "and data/README.md for what needs verifying first."
    )


def load_recording(subject_id: str, cap_root: Path) -> PSGRecording:
    """
    Load one subject's EEG (+ EMG if present) via mne or pyedflib.

    Must NOT silently substitute a wrong channel if the expected EMG channel
    name isn't found -- return emg=None and let the caller decide whether
    that subject is usable for RSWA training, rather than guessing.
    """
    raise NotImplementedError("Implement after Phase 1 channel inventory.")
