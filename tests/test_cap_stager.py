"""Faza 2: adapter stagera z submodulu (bez checkpointu -- testy ksztaltow i logiki)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.staging import cap_stager
from src.staging.cap_stager import (
    cap_stages_to_indices,
    load_stager,
    predict_stages,
    prepare_eeg_epochs,
    rem_windows,
)

needs_submodule = pytest.mark.skipif(
    not (cap_stager.SUBMODULE_MODELS_DIR / "sleep_stager.py").exists(),
    reason="submodul external/sleep_staging nie jest zainicjalizowany",
)


def test_prepare_eeg_epochs_resamples_200_to_100_hz_without_normalizing():
    x = np.random.default_rng(0).normal(0, 2e-5, (4, 6000))
    out = prepare_eeg_epochs(x, fs_in=200)
    assert out.shape == (4, 3000) and out.dtype == np.float32
    assert 1e-5 < out.std() < 4e-5  # nadal wolty, nie z-score


def test_prepare_eeg_epochs_is_identity_at_100_hz():
    x = np.random.default_rng(0).normal(0, 1e-5, (3, 3000))
    np.testing.assert_allclose(prepare_eeg_epochs(x, fs_in=100), x.astype(np.float32), rtol=1e-5)


class _StubModel(torch.nn.Module):
    """Przewiduje REM (4) dla epok o duzej amplitudzie, N2 (2) dla reszty."""

    def forward(self, x):
        rem = x.abs().mean(-1) > 1.0
        logits = torch.zeros(*x.shape[:2], 5)
        logits[..., 2] = 1.0
        logits[..., 4] = rem.float() * 2.0
        return logits


@pytest.mark.parametrize("n_epochs", [5, 20, 47])
def test_predict_stages_covers_every_epoch_including_partial_last_window(n_epochs):
    X = np.zeros((n_epochs, 3000), dtype=np.float32)
    X[-1] = 5.0  # ostatnia epoka to "REM"
    preds = predict_stages(_StubModel(), X, seq_len=20)
    assert preds.shape == (n_epochs,)
    assert np.all(preds >= 0)
    assert preds[-1] == 4 and np.all(preds[:-1] == 2)


def test_rem_windows_merges_contiguous_rem_epochs():
    stages = np.array([2, 4, 4, 2, 4])
    starts = np.array([0.0, 30.0, 60.0, 90.0, 150.0])  # luka 120 s
    assert rem_windows(stages, starts) == [(30.0, 90.0), (150.0, 180.0)]


def test_cap_stage_mapping_excludes_movement_and_unknown():
    assert cap_stages_to_indices(["WAKE", "N1", "N2", "N3", "REM", "MOVEMENT", "UNKNOWN"]).tolist() == [0, 1, 2, 3, 4, -1, -1]


def test_load_stager_fails_loudly_without_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_stager(tmp_path / "missing.pt")


@needs_submodule
def test_submodule_stager_loads_under_alias_and_runs_on_cap_shaped_input():
    model = cap_stager.build_stager("mamba").eval()
    X = prepare_eeg_epochs(np.random.default_rng(0).normal(0, 2e-5, (25, 6000)), fs_in=200)
    preds = predict_stages(model, X, seq_len=20)
    assert preds.shape == (25,) and set(preds.tolist()) <= {0, 1, 2, 3, 4}
    import src  # nasz pakiet src nie zostal nadpisany przez src submodulu

    assert Path(src.__path__[0]).resolve() == cap_stager.REPO_ROOT / "src"


@needs_submodule
def test_load_stager_roundtrip(tmp_path):
    path = tmp_path / "mamba_best.pt"
    torch.save(cap_stager.build_stager("mamba").state_dict(), path)
    assert not load_stager(path).training
