"""Kontrola przesluchu EKG na syntetycznych sygnalach o znanym przesluchu."""
import numpy as np
import pytest

from src.artifacts import detect_r_peaks, ecg_blanking_masks, ecg_contamination_ratio
from src.rswa_scoring import mini_epoch_rms, score_rswa_mini_epochs

FS = 200


def _ecg(seconds=30, hr_bpm=60, seed=0, invert=False):
    """Pociag waskich zalamkow R (~20 ms) + szum; zwraca (sygnal [V], indeksy R)."""
    rng = np.random.default_rng(seed)
    n = seconds * FS
    x = rng.normal(0, 20e-6, n)
    r_idx = np.arange(int(0.5 * FS), n - FS // 10, int(60 / hr_bpm * FS))
    kernel = 1e-3 * np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    for r in r_idx:
        x[r - 6:r + 7] += -kernel if invert else kernel
    return x, r_idx


@pytest.mark.parametrize("invert", [False, True])
def test_detect_r_peaks_finds_every_beat_regardless_of_polarity(invert):
    ecg, r_idx = _ecg(invert=invert)
    peaks = detect_r_peaks(ecg, FS)
    assert len(peaks) == len(r_idx)
    assert np.max(np.abs(peaks - r_idx)) <= 2


def test_contamination_ratio_near_one_without_leak_and_high_with_leak():
    rng = np.random.default_rng(1)
    ecgs, clean, leaky = [], [], []
    for i in range(5):
        ecg, _ = _ecg(seed=i)
        emg = rng.normal(0, 1e-6, len(ecg))
        ecgs.append(ecg)
        clean.append(emg)
        leaky.append(emg + 0.01 * ecg)  # 1% EKG w EMG: QRS ~10 uV na tle 1 uV
    assert ecg_contamination_ratio(clean, ecgs, FS) == pytest.approx(1.0, abs=0.1)
    assert ecg_contamination_ratio(leaky, ecgs, FS) > 1.5


def test_ecg_gating_removes_leak_driven_mini_epoch_activity():
    """EKG przeciekajace tylko do czesci epok udaje RSWA; po wycieciu QRS 'aktywnosc' znika."""
    rng = np.random.default_rng(2)
    emg, ecgs = [], []
    for i in range(10):
        ecg, _ = _ecg(seed=10 + i)
        signal = rng.normal(0, 1e-6, len(ecg))
        if i < 3:
            signal = signal + 0.03 * ecg  # QRS ~30 uV w EMG: RMS mini-epoki > 2x tla
        emg.append(signal)
        ecgs.append(ecg)
    raw = score_rswa_mini_epochs(emg, FS)
    gated = score_rswa_mini_epochs(emg, FS, exclude_masks=ecg_blanking_masks(ecgs, FS))
    assert raw.rswa_mini_index > 0.25
    assert gated.rswa_mini_index < 0.05


def test_ecg_gating_keeps_real_muscle_bursts():
    rng = np.random.default_rng(3)
    emg, ecgs = [], []
    for i in range(10):
        ecg, _ = _ecg(seed=20 + i)
        signal = rng.normal(0, 1e-6, len(ecg))
        if i < 3:
            signal[10 * FS:12 * FS] += rng.normal(0, 6e-6, 2 * FS)  # 2-s wybuch miesniowy
        emg.append(signal)
        ecgs.append(ecg)
    gated = score_rswa_mini_epochs(emg, FS, exclude_masks=ecg_blanking_masks(ecgs, FS))
    assert gated.rswa_mini_index == pytest.approx(0.03, abs=0.01)  # 3 z 100 mini-epok


def test_mini_epoch_rms_with_mask_ignores_excluded_samples_and_flags_mostly_excluded():
    x = np.ones(3 * FS * 2)
    x[:10] = 100.0
    exclude = np.zeros_like(x, dtype=bool)
    exclude[:10] = True
    exclude[3 * FS:3 * FS + int(0.8 * 3 * FS)] = True  # druga mini-epoka w 80% wycieta
    rms = mini_epoch_rms(x, FS, 3.0, exclude)
    assert rms[0] == pytest.approx(1.0)
    assert np.isnan(rms[1])


def test_ecg_contamination_ratio_nan_without_beats():
    flat = [np.zeros(30 * FS)]
    assert np.isnan(ecg_contamination_ratio([np.ones(30 * FS)], flat, FS))
