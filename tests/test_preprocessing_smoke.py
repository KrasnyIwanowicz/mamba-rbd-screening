"""
Synthetic-data smoke tests — no dataset download needed, same pattern as
the 17 synthetic-data tests in parkinsons-eeg-classifier and the
per-phase tests in mamba-eeg-sleep-staging.
"""
import numpy as np

from src.preprocessing import bandpass_filter, emg_rms_envelope, extract_rem_epochs


def test_bandpass_filter_preserves_length():
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)
    filtered = bandpass_filter(signal, low_hz=1, high_hz=45, fs=fs)
    assert filtered.shape == signal.shape


def test_emg_rms_envelope_nonnegative():
    fs = 200.0
    rng = np.random.default_rng(0)
    emg = rng.normal(0, 1, size=int(fs * 5))
    env = emg_rms_envelope(emg, fs=fs)
    assert env.shape == emg.shape
    assert np.all(env >= 0)


def test_emg_rms_envelope_higher_for_higher_amplitude_signal():
    # Sanity check on the actual clinical logic: more muscle activity ->
    # higher RMS envelope. If this ever fails, the envelope function is
    # broken, not just "different," since this is a monotonic physical fact.
    fs = 200.0
    n = int(fs * 5)
    quiet = np.zeros(n)
    active = np.ones(n) * 3.0
    assert emg_rms_envelope(active, fs=fs).mean() > emg_rms_envelope(quiet, fs=fs).mean()


def test_extract_rem_epochs_only_returns_rem_labeled_windows():
    fs = 10.0
    epoch_length_s = 2.0
    epoch_len_samples = int(epoch_length_s * fs)
    n_epochs = 6
    signal = np.arange(n_epochs * epoch_len_samples, dtype=float)
    # stages: Wake, N1, N2, N3, REM, REM  (rem_label=4)
    hypnogram = np.array([0, 1, 2, 3, 4, 4])

    rem_epochs = extract_rem_epochs(signal, hypnogram, fs=fs, epoch_length_s=epoch_length_s, rem_label=4)

    assert len(rem_epochs) == 2
    expected_first = signal[4 * epoch_len_samples : 5 * epoch_len_samples]
    np.testing.assert_array_equal(rem_epochs[0], expected_first)
