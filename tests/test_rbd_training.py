"""Dataset (spojna liczba kanalow), wybor pacjentow i grupowa CV -- na danych syntetycznych."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.cap_loader import EpochData
from src.models.mamba_rbd import MambaRBDClassifier
from src.training.train_rbd import cross_validate, select_subjects


def _epoch(subject_id, has_leg, n=600):
    rng = np.random.default_rng(len(subject_id))
    return EpochData(
        subject_id=subject_id, epoch_idx=0, stage="REM", start_sec=0.0, duration_sec=30.0,
        emg_chin=rng.normal(size=n), emg_leg=rng.normal(size=n) if has_leg else None,
        eeg_central=None, sampling_rate=20, is_rbd=subject_id.startswith("rbd"),
    )


def test_dataset_pads_missing_leg_channel_so_batches_collate(monkeypatch):
    import src.data.rbd_dataset as mod

    class FakeLoader:
        def __init__(self, **kwargs):
            pass

        def load_subject(self, subject_id, stages_filter=None):
            return [_epoch(subject_id, has_leg=(subject_id == "rbd1"))] * 2

    monkeypatch.setattr(mod, "CAPSleepLoader", FakeLoader)
    ds = mod.RBDEpochDataset(["rbd1", "n1"])  # rbd1 ma noge, n1 nie
    batch = next(iter(DataLoader(ds, batch_size=4)))  # wczesniej: RuntimeError w collate
    assert batch["signal"].shape == (4, 2, 600)
    assert torch.all(batch["signal"][2:, 1] == 0)  # kanal nogi n1 wyzerowany
    assert ds.groups == ["rbd1", "rbd1", "n1", "n1"]


def test_select_subjects_excludes_other_cap_pathologies(tmp_path):
    for sid in ["rbd1", "rbd2", "n1", "nfle1", "narco1", "plm1"]:
        (tmp_path / f"{sid}.edf").touch()
        (tmp_path / f"{sid}.txt").touch()
    (tmp_path / "n2.edf").touch()  # brak .txt -> pominiety
    assert select_subjects(tmp_path) == ["n1", "rbd1", "rbd2"]


class _Synthetic(Dataset):
    """Pacjenci rbd: wyzsza amplituda EMG. 3 epoki na pacjenta, [2, 600] probek."""

    def __init__(self, n_per_group=3):
        rng = np.random.default_rng(0)
        self.items, self.groups, labels = [], [], []
        for group, amp, label in (("rbd", 3.0, 1), ("n", 1.0, 0)):
            for i in range(n_per_group):
                for _ in range(3):
                    self.items.append((torch.tensor(rng.normal(0, amp, (2, 600)), dtype=torch.float32), label))
                    self.groups.append(f"{group}{i}")
                    labels.append(label)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        return {"signal": x, "label": torch.tensor(y)}


def test_cross_validate_runs_all_folds_and_reports_seed_spread():
    ds = _Synthetic()
    result = cross_validate(
        ds, ds.groups, ds.labels, seeds=[0, 1], epochs=1, batch_size=4, n_splits=3,
        model_factory=lambda: MambaRBDClassifier(in_channels=2, d_model=8, d_state=4, n_layers=1),
    )
    assert [r["seed"] for r in result["per_seed"]] == [0, 1]
    # kazdy pacjent przewidziany dokladnie raz na seed (wszystkie foldy, nie tylko pierwszy)
    assert len(result["predictions"]) == 2 * 6
    mean, std = result["summary"]["auc"]
    assert 0.0 <= mean <= 1.0 and std >= 0.0


def test_cross_validate_requires_both_classes():
    ds = _Synthetic()
    with pytest.raises(ValueError):
        cross_validate(ds, ds.groups, np.ones(len(ds), dtype=int), seeds=[0], epochs=1)
