"""Subject-level evaluation of a continuous RSWA metric: RBD vs. control.

Jedyny prawdziwy ground truth w CAP to diagnoza na poziomie PACJENTA (grupa
"rbd" vs "n") -- nie ma etykiet per epoka (patrz docs/technical_premise.md).
Dlatego ewaluacja jest na poziomie pacjenta:

- AUC (bez progu) + test Manna-Whitneya: czy metryka w ogole rozdziela grupy;
- czulosc/swoistosc przy progu wybranym w leave-one-subject-out: prog jest
  wybierany (maks. Youden J) na n-1 pacjentach i stosowany do odlozonego --
  wybranie progu na calym zbiorze i raportowanie accuracy na tym samym
  zbiorze to wlasnie "fake-looking result" z ROADMAP, przed ktorym ostrzega.

Przy n~38 przedzialy ufnosci sa szerokie -- raportujemy bootstrapowy 95% CI
dla AUC, zeby nie sprzedawac punktowej wartosci jako pewnej.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu, rankdata
from sklearn.metrics import roc_auc_score


@dataclass
class SubjectLevelResult:
    metric: str
    n_positive: int
    n_negative: int
    higher_means_rbd: bool
    auc: float
    auc_ci95: tuple[float, float]
    mannwhitney_p: float
    loso_sensitivity: float
    loso_specificity: float
    loso_accuracy: float


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Prog maksymalizujacy czulosc + swoistosc - 1 (predykcja: score >= prog -> 1).

    Kandydaci to srodki miedzy kolejnymi unikalnymi wartosciami (plus skrajne),
    a nie same obserwacje -- prog lezacy dokladnie NA najnizszym wyniku klasy
    dodatniej systematycznie myli w LOSO pacjenta tuz ponizej niego.
    """
    uniq = np.unique(scores)
    candidates = np.concatenate([[uniq[0]], (uniq[:-1] + uniq[1:]) / 2.0, [uniq[-1] + 1e-12]])
    best_t, best_j = candidates[0], -np.inf
    for t in candidates:
        pred = scores >= t
        sens = np.mean(pred[labels == 1]) if np.any(labels == 1) else 0.0
        spec = np.mean(~pred[labels == 0]) if np.any(labels == 0) else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_t, best_j = t, j
    return float(best_t)


def loso_threshold_predictions(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Predykcja dla kazdego pacjenta z progiem dobranym bez niego."""
    preds = np.zeros(len(scores), dtype=int)
    for i in range(len(scores)):
        mask = np.arange(len(scores)) != i
        t = youden_threshold(scores[mask], labels[mask])
        preds[i] = int(scores[i] >= t)
    return preds


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC = U / (n1 * n0) (Mann-Whitney), remisy jako 0.5 -- to samo co roc_auc_score, szybciej."""
    n1 = int(np.sum(labels == 1))
    n0 = len(labels) - n1
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(_rank_auc(labels[idx], scores[idx]))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def evaluate_subject_metric(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str,
    higher_means_rbd: bool = True,
) -> SubjectLevelResult:
    """labels: 1 = rbd, 0 = kontrola. NaN w scores -> pacjent pomijany.

    higher_means_rbd=False dla metryk typu RAI (nizszy = mniej atonii);
    wtedy score jest negowany przed AUC/progowaniem, zeby kierunek byl
    ustalony Z GORY, a nie dobrany po fakcie pod lepsze AUC.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    keep = ~np.isnan(scores)
    scores, labels = scores[keep], labels[keep]
    if len(np.unique(labels)) < 2:
        raise ValueError("Potrzeba pacjentow z obu grup (rbd i n) do ewaluacji.")
    oriented = scores if higher_means_rbd else -scores

    auc = float(roc_auc_score(labels, oriented))
    p = float(mannwhitneyu(oriented[labels == 1], oriented[labels == 0], alternative="greater").pvalue)
    preds = loso_threshold_predictions(oriented, labels)
    return SubjectLevelResult(
        metric=metric,
        n_positive=int(np.sum(labels == 1)),
        n_negative=int(np.sum(labels == 0)),
        higher_means_rbd=higher_means_rbd,
        auc=auc,
        auc_ci95=bootstrap_auc_ci(oriented, labels),
        mannwhitney_p=p,
        loso_sensitivity=float(np.mean(preds[labels == 1] == 1)),
        loso_specificity=float(np.mean(preds[labels == 0] == 0)),
        loso_accuracy=float(np.mean(preds == labels)),
    )
