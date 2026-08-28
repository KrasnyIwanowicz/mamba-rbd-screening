"""
Preprocessing: filtering, EMG envelope extraction, REM-epoch windowing.

Follows the same discipline as parkinsons-eeg-classifier/src/preprocessing.py
(band-pass + notch + fixed epoching) but adds an EMG-specific path, since
this project (unlike the two existing repos) needs a muscle-tone signal,
not just cortical EEG.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(signal: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass, same approach as the other two repos."""
    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, signal)


def emg_rms_envelope(emg: np.ndarray, fs: float, window_s: float = 1.0) -> np.ndarray:
    """
    RMS envelope of the EMG signal, the standard clinical proxy for muscle
    tone used in RSWA scoring (AASM scoring manual uses amplitude/tone
    thresholds on rectified/RMS EMG, not raw amplitude).

    This is the feature the EMG-trained RSWA baseline should be checked
    against first (Phase 3) -- if a learned classifier can't beat a simple
    RMS threshold, the learned model isn't earning its complexity.
    """
    window = int(window_s * fs)
    squared = emg.astype(np.float64) ** 2
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(squared, kernel, mode="same"))


def extract_rem_epochs(
    signal: np.ndarray,
    hypnogram: np.ndarray,
    fs: float,
    epoch_length_s: float = 30.0,
    rem_label: int = 4,
) -> list[np.ndarray]:
    """
    Slice a continuous signal into REM-only epochs, given a per-epoch
    hypnogram from the sleep-stager (Phase 2 output).

    rem_label follows the sleep-staging repo's class mapping -- confirm the
    exact integer/label convention against mamba-eeg-sleep-staging's
    src/data_loader.py class_mapping before wiring this together; don't
    assume REM==4 without checking.
    """
    epoch_len_samples = int(epoch_length_s * fs)
    rem_epochs = []
    for i, stage in enumerate(hypnogram):
        if stage == rem_label:
            start = i * epoch_len_samples
            end = start + epoch_len_samples
            if end <= len(signal):
                rem_epochs.append(signal[start:end])
    return rem_epochs
