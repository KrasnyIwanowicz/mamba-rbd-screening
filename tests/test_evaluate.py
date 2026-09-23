import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate_subject_metric, loso_threshold_predictions, youden_threshold


def test_youden_threshold_separates_perfectly_separable_scores():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert youden_threshold(scores, labels) == pytest.approx(0.5)  # srodek luki 0.3-0.7


def test_loso_threshold_never_uses_the_held_out_subject():
    # Pojedynczy "odstajacy" pacjent: gdyby prog byl wybierany z nim, bylby
    # zawsze poprawnie sklasyfikowany. W LOSO musi byc bledny.
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.15])
    labels = np.array([0, 0, 0, 1, 1, 1])
    preds = loso_threshold_predictions(scores, labels)
    assert preds[5] == 0


def test_evaluate_subject_metric_respects_prespecified_direction():
    rai = np.array([0.95, 0.9, 0.92, 0.5, 0.6, 0.4])  # nizszy RAI = RBD
    labels = np.array([0, 0, 0, 1, 1, 1])
    right = evaluate_subject_metric(rai, labels, "rem_atonia_index", higher_means_rbd=False)
    wrong = evaluate_subject_metric(rai, labels, "rem_atonia_index", higher_means_rbd=True)
    assert right.auc == 1.0 and right.loso_accuracy == 1.0
    assert wrong.auc == 0.0  # kierunek NIE jest odwracany po fakcie


def test_evaluate_subject_metric_drops_nan_and_needs_both_groups():
    r = evaluate_subject_metric(np.array([0.1, np.nan, 0.9]), np.array([0, 1, 1]), "m")
    assert r.n_positive == 1 and r.n_negative == 1
    with pytest.raises(ValueError):
        evaluate_subject_metric(np.array([0.1, 0.2]), np.array([1, 1]), "m")


def test_evaluate_rswa_script_writes_report(tmp_path):
    from scripts.evaluate_rswa import evaluate_csv

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "subject_id": [f"rbd{i}" for i in range(1, 9)] + [f"n{i}" for i in range(1, 9)] + ["nfle1"],
        "group": ["rbd"] * 8 + ["n"] * 8 + ["nfle"],
        "rswa_index": rng.uniform(0, 0.1, 17),
        "rswa_mini_index": np.r_[rng.uniform(0.2, 0.5, 8), rng.uniform(0.0, 0.1, 8), 0.9],
        "tonic_epoch_fraction": rng.uniform(0, 0.2, 17),
        "rem_atonia_index": np.r_[rng.uniform(0.4, 0.7, 8), rng.uniform(0.9, 1.0, 8), 0.1],
    })
    scores = tmp_path / "scores.csv"
    df.to_csv(scores, index=False)
    rows = evaluate_csv(scores, tmp_path / "eval.csv")
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["rswa_mini_index"]["auc"] == 1.0
    assert by_metric["rem_atonia_index"]["auc"] == 1.0
    assert by_metric["rswa_mini_index"]["n_rbd"] + by_metric["rswa_mini_index"]["n_control"] == 16  # nfle wykluczony


def test_rank_auc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score

    from src.evaluate import _rank_auc

    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, 40)
    scores = rng.integers(0, 5, 40).astype(float)  # duzo remisow
    assert _rank_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))
