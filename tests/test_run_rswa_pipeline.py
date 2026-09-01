"""
Testy orchestracji scripts/run_rswa_pipeline.py na fikcyjnym loaderze --
nie wymagaja pobranego CAP Sleep Database. Sprawdzaja logike (dzielenie
epok na REM/NREM, obsluge brakujacych danych), nie prawdziwe liczby EMG.
"""
import numpy as np
import pytest

from scripts.run_rswa_pipeline import compute_subject_rswa, run


class _FakeEpoch:
    def __init__(self, stage, emg_chin, is_rbd):
        self.stage = stage
        self.emg_chin = emg_chin
        self.is_rbd = is_rbd


class _FakeLoader:
    """Podstawia sie pod CAPSleepLoader.load_subject bez dotykania mne/EDF."""

    def __init__(self, subjects: dict):
        self._subjects = subjects  # subject_id -> list[_FakeEpoch] albo Exception

    def load_subject(self, subject_id, stages_filter=None):
        result = self._subjects.get(subject_id, [])
        if isinstance(result, Exception):
            raise result
        return result


def _quiet(seed=0, n=100):
    return np.random.default_rng(seed).normal(0, 0.05, size=n)


def _active(seed=0, n=100):
    return np.random.default_rng(seed).normal(0, 1.0, size=n)


def test_compute_subject_rswa_splits_rem_and_nrem_correctly():
    epochs = [
        _FakeEpoch("WAKE", _quiet(0), is_rbd=True),
        _FakeEpoch("N2", _quiet(1), is_rbd=True),
        _FakeEpoch("N3", _quiet(2), is_rbd=True),
        _FakeEpoch("REM", _quiet(3), is_rbd=True),
        _FakeEpoch("REM", _active(4), is_rbd=True),
    ]
    loader = _FakeLoader({"rbd1": epochs})

    result = compute_subject_rswa(loader, "rbd1")

    assert result["skipped_reason"] == ""
    assert result["group"] == "rbd"
    assert result["n_rem_epochs"] == 2
    assert result["n_nrem_epochs"] == 2  # N2 + N3, WAKE wykluczone
    assert 0.0 <= result["rswa_index"] <= 1.0


def test_compute_subject_rswa_skips_when_no_nrem_epochs():
    epochs = [_FakeEpoch("REM", _quiet(0), is_rbd=False)]
    loader = _FakeLoader({"n1": epochs})

    result = compute_subject_rswa(loader, "n1")

    assert "N2/N3" in result["skipped_reason"]


def test_compute_subject_rswa_skips_when_no_rem_epochs():
    epochs = [_FakeEpoch("N2", _quiet(0), is_rbd=False)]
    loader = _FakeLoader({"n1": epochs})

    result = compute_subject_rswa(loader, "n1")

    assert "REM" in result["skipped_reason"]


def test_compute_subject_rswa_handles_missing_file_gracefully():
    loader = _FakeLoader({"rbd99": FileNotFoundError("brak pliku")})

    result = compute_subject_rswa(loader, "rbd99")

    assert "brak pliku" in result["skipped_reason"]


def test_compute_subject_rswa_handles_missing_emg_channel_gracefully():
    # CAPSleepLoader.load_subject rzuca ValueError, nie zwraca [], gdy nie
    # znajdzie kanalu Chin EMG -- upewniamy sie, ze to tez lapiemy, nie
    # tylko FileNotFoundError.
    loader = _FakeLoader({"n4": ValueError("Pacjent n4 nie posiada kanalu Chin EMG")})

    result = compute_subject_rswa(loader, "n4")

    assert "Chin EMG" in result["skipped_reason"]


def test_compute_subject_rswa_handles_empty_epoch_list():
    loader = _FakeLoader({"n4": []})

    result = compute_subject_rswa(loader, "n4")

    assert "0 epok" in result["skipped_reason"]


def test_run_writes_csv_with_mixed_success_and_skips(tmp_path, monkeypatch):
    import scripts.run_rswa_pipeline as mod

    good_epochs = [
        _FakeEpoch("N2", _quiet(0), is_rbd=True),
        _FakeEpoch("REM", _quiet(1), is_rbd=True),
    ]
    fake_loader = _FakeLoader({"rbd1": good_epochs, "n1": []})
    monkeypatch.setattr(mod, "CAPSleepLoader", lambda data_dir: fake_loader)

    output_csv = tmp_path / "reports" / "rswa_scores.csv"
    rows = run(data_dir="unused", subject_ids=["rbd1", "n1"], output_csv=output_csv)

    assert output_csv.exists()
    assert len(rows) == 2
    content = output_csv.read_text(encoding="utf-8")
    assert "rbd1" in content
    assert "n1" in content
