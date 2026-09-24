"""Kontrola przesluchu EKG w EMG brody.

Zespol QRS ma energie do ~30-40 Hz, wiec filtr 10-100 Hz go nie usuwa. Jesli
elektrody brody "slysza" serce, kazde uderzenie wyglada jak krotki wybuch EMG
i zawyza metryki mini-epokowe -- szczegolnie gdy przesluch rozni sie miedzy
grupami (inne elektrody/wzmacniacze), co udaje roznice RBD vs kontrola.

Dwa narzedzia:
- ecg_contamination_ratio: srednia |EMG| w oknie +-50 ms wokol zalamkow R
  podzielona przez srednia |EMG| ogolem. ~1 = brak przesluchu; wyraznie > 1
  (heurystycznie > 1.5) = EMG podaza za rytmem serca.
- ecg_blanking_masks: maski probek do wykluczenia (+-50 ms wokol R), do
  policzenia metryk RSWA "z wycietym EKG" jako testu odpornosci.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from src.preprocessing import bandpass_filter

QRS_HALF_WINDOW_S = 0.05


def detect_r_peaks(ecg: np.ndarray, fs: float, min_rr_s: float = 0.33) -> np.ndarray:
    """Indeksy zalamkow R: pasmo 5-30 Hz, kwadrat (niezaleznie od polaryzacji odprowadzenia), find_peaks.

    Prosty detektor do diagnostyki, nie do analizy HRV. min_rr_s=0.33 s -> max ~180/min.
    """
    ecg = np.asarray(ecg, dtype=np.float64)
    if len(ecg) < int(fs):
        return np.empty(0, dtype=int)
    energy = bandpass_filter(ecg - ecg.mean(), 5.0, min(30.0, 0.45 * fs), fs) ** 2
    ref = np.percentile(energy, 99.5)
    if ref <= 0:
        return np.empty(0, dtype=int)
    peaks, _ = find_peaks(energy, height=0.2 * ref, distance=max(1, int(min_rr_s * fs)))
    return peaks


def _qrs_mask(n: int, peaks: np.ndarray, fs: float, half_window_s: float) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    half = int(round(half_window_s * fs))
    for p in peaks:
        mask[max(0, p - half):min(n, p + half + 1)] = True
    return mask


def ecg_blanking_masks(ecg_epochs: list[np.ndarray], fs: float, half_window_s: float = QRS_HALF_WINDOW_S) -> list[np.ndarray]:
    """Per epoka: True = probka w oknie QRS (do wykluczenia z EMG)."""
    return [_qrs_mask(len(e), detect_r_peaks(e, fs), fs, half_window_s) for e in ecg_epochs]


def ecg_contamination_ratio(
    emg_epochs: list[np.ndarray],
    ecg_epochs: list[np.ndarray],
    fs: float,
    half_window_s: float = QRS_HALF_WINDOW_S,
) -> float:
    """Srednia |EMG| w oknach QRS / srednia |EMG| ogolem (po wszystkich epokach). NaN bez zalamkow R."""
    near, total_sum, total_n = [], 0.0, 0
    for emg, ecg in zip(emg_epochs, ecg_epochs):
        amp = np.abs(np.asarray(emg, dtype=np.float64))
        mask = _qrs_mask(len(amp), detect_r_peaks(ecg, fs), fs, half_window_s)
        if mask.any():
            near.append(amp[mask])
        total_sum += float(amp.sum())
        total_n += len(amp)
    if not near or total_n == 0 or total_sum == 0:
        return float("nan")
    return float(np.concatenate(near).mean() / (total_sum / total_n))
