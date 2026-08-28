from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(signal: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 4) -> np.ndarray:
    #Zero-phase Butterworth band-pass, same approach as the other two repos.
    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, signal)


def emg_rms_envelope(emg: np.ndarray, fs: float, window_s: float = 1.0) -> np.ndarray:
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
    epoch_len_samples = int(epoch_length_s * fs)
    rem_epochs = []
    for i, stage in enumerate(hypnogram):
        if stage == rem_label:
            start = i * epoch_len_samples
            end = start + epoch_len_samples
            if end <= len(signal):
                rem_epochs.append(signal[start:end])
    return rem_epochs
