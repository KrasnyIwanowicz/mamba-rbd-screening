from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 4) -> np.ndarray:
    # Zero-phase Butterworth band-pass, same approach as the other two repos.
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


# Standardowe pasmo EMG podbrodkowego w scoringu RSWA (AASM; Ferri i wsp. 2008
# licza REM Atonia Index na 10-100 Hz). Dolna granica 10 Hz usuwa dryf linii
# bazowej, artefakty ruchowe i wiekszosc energii przesluchu EKG -- bez tego
# RMS epoki mierzy glownie te artefakty, a nie napiecie miesnia.
EMG_BAND_HZ = (10.0, 100.0)
MAINS_HZ = 50.0  # CAP pochodzi z laboratoriow wloskich (siec 50 Hz)


def notch_filter(signal: np.ndarray, fs: float, freq_hz: float = MAINS_HZ, quality: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch; no-op gdy freq_hz jest ponad Nyquistem."""
    if freq_hz >= fs / 2.0:
        return np.asarray(signal, dtype=np.float64)
    b, a = iirnotch(freq_hz, quality, fs=fs)
    return filtfilt(b, a, signal)


def emg_bandpass(
    signal: np.ndarray,
    fs: float,
    low_hz: float = EMG_BAND_HZ[0],
    high_hz: float = EMG_BAND_HZ[1],
    mains_hz: float | None = MAINS_HZ,
    order: int = 4,
) -> np.ndarray:
    """Pasmo EMG do scoringu RSWA: high-pass low_hz, low-pass min(high_hz, 0.45*fs), notch sieci.

    Przy fs=200 Hz gorna granica 100 Hz rowna sie Nyquistowi, wiec jest
    przycinana do 90 Hz -- roznica wzgledem 10-100 Hz z literatury jest
    niewielka, ale istnieje; nie udajemy, ze to identyczne pasmo.
    """
    signal = np.asarray(signal, dtype=np.float64)
    high = min(high_hz, 0.45 * fs)
    if high <= low_hz:
        raise ValueError(f"fs={fs} Hz za niskie dla pasma EMG od {low_hz} Hz")
    filtered = bandpass_filter(signal, low_hz, high, fs, order=order)
    if mains_hz is not None:
        filtered = notch_filter(filtered, fs, freq_hz=mains_hz)
    return filtered
