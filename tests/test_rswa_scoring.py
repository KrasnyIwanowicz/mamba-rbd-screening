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
