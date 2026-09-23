import numpy as np
import pytest

from src.data.cap_loader import EpochData
from src.pipeline.rbd_pipeline import RBDScreeningPipeline

FS = 200


def _epoch(i, stage, amp, burst=False):
    rng = np.random.default_rng(i)
    chin = rng.normal(0, amp, 30 * FS)
    if burst:
        chin[: 3 * FS] = rng.normal(0, 10 * amp, 3 * FS)
    return EpochData("rbd1", i, stage, 30.0 * i, 30.0, chin, None, None, FS, True)


def _night():
    return [_epoch(i, "N2", 3e-6) for i in range(10)] + [_epoch(10 + i, "REM", 1e-6, burst=i % 2 == 0) for i in range(10)]


def test_missing_model_checkpoint_is_an_error_not_random_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        RBDScreeningPipeline(rbd_model_path=tmp_path / "typo.pt")


def test_auto_stager_requires_checkpoint():
    with pytest.raises(ValueError):
        RBDScreeningPipeline(use_auto_stager=True)


def test_screen_epochs_without_threshold_is_uncalibrated_not_a_diagnosis():
    result = RBDScreeningPipeline().screen_epochs("rbd1", _night())
    assert result.predicted_label == "UNCALIBRATED"
    assert result.rem_epochs_detected == 10
    assert result.rswa_mini_index == pytest.approx(0.05, abs=0.01)
    assert result.model_rbd_score is None


def test_screen_epochs_applies_calibrated_threshold():
    assert RBDScreeningPipeline(rswa_mini_threshold=0.02).screen_epochs("rbd1", _night()).predicted_label == "ELEVATED_RSWA"
    assert RBDScreeningPipeline(rswa_mini_threshold=0.5).screen_epochs("rbd1", _night()).predicted_label == "NORMAL_RSWA"


def test_screen_epochs_without_rem():
    result = RBDScreeningPipeline().screen_epochs("n1", [_epoch(0, "N2", 1e-6)])
    assert result.predicted_label == "NO_REM_DETECTED"
