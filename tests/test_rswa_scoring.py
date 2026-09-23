"""Testy dla src/rswa_scoring.py na syntetycznym EMG."""
import numpy as np
import pytest

from src.rswa_scoring import compute_nrem_baseline, score_rswa


def _quiet_epoch(n=1000, amplitude=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, amplitude, size=n)


def _active_epoch(n=1000, amplitude=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, amplitude, size=n)


def test_compute_nrem_baseline_requires_at_least_one_epoch():
    with pytest.raises(ValueError):
        compute_nrem_baseline([])


def test_compute_nrem_baseline_is_median_rms():
    epochs = [_quiet_epoch(seed=i) for i in range(5)]
    baseline = compute_nrem_baseline(epochs)
    assert baseline > 0
    # powinno być zbliżone do amplitude=0.05 (RMS szumu gaussowskiego ~ sigma)
    assert 0.03 < baseline < 0.08


def test_score_rswa_flags_high_amplitude_rem_epochs_as_atonia_lost():
    nrem_epochs = [_quiet_epoch(seed=i) for i in range(10)]
    # 3 ciche epoki REM (atonia zachowana) + 2 głośne (atonia utracona)
    rem_epochs = [_quiet_epoch(seed=100 + i) for i in range(3)] + [
        _active_epoch(seed=200 + i) for i in range(2)
    ]

    result = score_rswa(rem_epochs, nrem_epochs, threshold_multiplier=2.0)

    assert result.atonia_lost.tolist() == [False, False, False, True, True]
    assert result.rswa_index == pytest.approx(2 / 5)


def test_score_rswa_returns_nan_index_with_zero_rem_epochs():
    nrem_epochs = [_quiet_epoch(seed=i) for i in range(5)]
    result = score_rswa([], nrem_epochs)
    assert np.isnan(result.rswa_index)
    assert len(result.atonia_lost) == 0


def test_score_rswa_baseline_scales_with_threshold_multiplier():
    nrem_epochs = [_quiet_epoch(seed=i) for i in range(10)]
    rem_epochs = [_active_epoch(amplitude=0.15, seed=i) for i in range(5)]

    lenient = score_rswa(rem_epochs, nrem_epochs, threshold_multiplier=1.5)
    strict = score_rswa(rem_epochs, nrem_epochs, threshold_multiplier=5.0)

    # Wyższy próg -> mniej (albo tyle samo) epok oznaczonych jako atonia_lost
    assert strict.rswa_index <= lenient.rswa_index


# --- metryki mini-epokowe i RAI (2026-09-23) --------------------------------
from src.rswa_scoring import mini_epoch_rms, rem_atonia_index, score_rswa_mini_epochs

FS = 200


def _epoch(amplitude=1e-6, seed=0, burst=None):
    """30-s epoka EMG [V]; burst=(start_s, dur_s, amplitude) dokleja wybuch fazowy."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, amplitude, 30 * FS)
    if burst is not None:
        start, dur, amp = burst
        x[int(start * FS):int((start + dur) * FS)] = rng.normal(0, amp, int(dur * FS))
    return x


def test_mini_epoch_rms_shape():
    assert mini_epoch_rms(np.ones(30 * FS), FS, 3.0).shape == (10,)


def test_phasic_bursts_invisible_to_30s_rule_but_caught_by_mini_epochs():
    """Dokladnie mechanizm rswa_index=0.0 dla rbd1: 2-s wybuchy w REM rozmywaja
    sie w RMS 30 s, a NREM ma wyzsze napiecie tla niz atoniczny REM."""
    nrem = [_epoch(3e-6, seed=i) for i in range(20)]
    rem = [_epoch(1e-6, seed=100 + i, burst=(10, 2, 5e-6)) for i in range(10)]

    legacy = score_rswa(rem, nrem, threshold_multiplier=2.0)
    mini = score_rswa_mini_epochs(rem, FS)

    assert legacy.rswa_index == 0.0
    assert mini.rswa_mini_index == pytest.approx(0.1, abs=0.02)  # 1 z 10 mini-epok
    assert mini.tonic_epoch_fraction == 0.0  # fazowe, nie toniczne


def test_tonic_activity_counts_as_tonic_epochs():
    rem = [_epoch(1e-6, seed=i) for i in range(8)] + [_epoch(6e-6, seed=50 + i) for i in range(2)]
    mini = score_rswa_mini_epochs(rem, FS)
    assert mini.tonic_epoch_fraction == pytest.approx(0.2)


def test_score_rswa_mini_epochs_empty_input_is_nan():
    assert np.isnan(score_rswa_mini_epochs([], FS).rswa_mini_index)


def test_rai_is_near_one_for_atonia_and_low_for_sustained_activity():
    atonic = [_epoch(0.3e-6, seed=i) for i in range(6)]
    active = [_epoch(0.3e-6, seed=i, burst=(0, 30, 6e-6)) if i % 2 else _epoch(0.3e-6, seed=i) for i in range(6)]
    rai_atonic = rem_atonia_index(atonic, FS)
    rai_active = rem_atonia_index(active, FS, epoch_starts_s=np.arange(6) * 30.0)
    assert rai_atonic > 0.95
    assert rai_active < rai_atonic - 0.3
    assert 0.0 <= rai_active <= 1.0


def test_rai_empty_is_nan():
    assert np.isnan(rem_atonia_index([], FS))
