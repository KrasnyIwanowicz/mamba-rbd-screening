"""
src/training/train_rbd.py
Trening MambaRBDClassifier z walidacja krzyzowa GRUPOWANA po pacjencie
(domyslnie leave-one-subject-out), kilka seedow, ewaluacja na poziomie pacjenta.

Co ten model faktycznie przewiduje: kazda epoka REM dostaje etykiete
diagnozy swojego pacjenta (rbd=1, n=0), bo CAP nie ma etykiet RSWA per
epoka. Wynik per pacjent = srednie prawdopodobienstwo po jego epokach REM.
Raportujemy AUC/accuracy/czulosc/swoistosc per seed ORAZ rozrzut miedzy
seedami (ROADMAP Faza 3: nie chowac sie za jednym szczesliwym runem).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (src/training/ -> src/ -> root)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.cap_loader import subject_group
from src.models.mamba_rbd import MambaRBDClassifier


def select_subjects(data_dir: str | Path, groups: tuple[str, ...] = ("rbd", "n")) -> list[str]:
    """Pacjenci z kompletem EDF+TXT, tylko z zadanych grup CAP.

    Poprzednio brane byly WSZYSTKIE *.edf, wiec np. nfle*/narco*/plm* (inne
    patologie) trafialy do klasy 0 jako "zdrowi".
    """
    data_dir = Path(data_dir)
    selected = []
    for edf in sorted(data_dir.glob("*.edf")):
        try:
            group = subject_group(edf.stem)
        except ValueError:
            continue
        if group in groups and edf.with_suffix(".txt").exists():
            selected.append(edf.stem)
    return selected


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip: float | None = 1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (out.argmax(dim=-1) == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        out = model(batch["signal"].to(device))
        probs.append(torch.softmax(out, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.empty(0)


def _class_weights(labels: np.ndarray, device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def subject_metrics(subject_scores: dict[str, float], subject_labels: dict[str, int], threshold: float = 0.5) -> dict:
    ids = sorted(subject_scores)
    s = np.array([subject_scores[i] for i in ids])
    y = np.array([subject_labels[i] for i in ids])
    pred = (s >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan"),
        "accuracy": float(np.mean(pred == y)),
        "sensitivity": float(np.mean(pred[y == 1] == 1)) if np.any(y == 1) else float("nan"),
        "specificity": float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else float("nan"),
    }


def cross_validate(
    dataset: Dataset,
    groups: list[str],
    labels: np.ndarray,
    seeds: list[int],
    epochs: int,
    batch_size: int = 16,
    lr: float = 1e-3,
    grad_clip: float | None = 1.0,
    n_splits: int | None = None,
    device: str | torch.device = "cpu",
    model_factory: Callable[[], nn.Module] | None = None,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Grupowa CV po pacjencie; zwraca metryki per seed, mean/std i predykcje per pacjent.

    n_splits=None -> leave-one-subject-out; liczba -> StratifiedGroupKFold
    (szybciej, zachowuje obie klasy w kazdym foldzie).
    """
    groups_arr = np.asarray(groups)
    labels = np.asarray(labels, dtype=int)
    subject_labels = {g: int(labels[groups_arr == g][0]) for g in np.unique(groups_arr)}
    if len(set(subject_labels.values())) < 2:
        raise ValueError("Potrzeba pacjentow z obu klas (rbd i n).")
    model_factory = model_factory or (lambda: MambaRBDClassifier(in_channels=2, d_model=64, n_layers=2))
    indices = np.arange(len(labels))

    per_seed: list[dict] = []
    predictions: list[dict] = []
    for seed in seeds:
        splitter = (
            LeaveOneGroupOut()
            if n_splits is None
            else StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        )
        subject_scores: dict[str, float] = {}
        for fold, (train_idx, val_idx) in enumerate(splitter.split(indices, labels, groups=groups_arr)):
            torch.manual_seed(seed * 1000 + fold)
            np.random.seed(seed * 1000 + fold)
            model = model_factory().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            criterion = nn.CrossEntropyLoss(weight=_class_weights(labels[train_idx], device))
            generator = torch.Generator().manual_seed(seed * 1000 + fold)
            train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True, generator=generator)
            val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=batch_size, shuffle=False)

            for _ in range(epochs):
                train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip)

            probs = predict_proba(model, val_loader, device)
            for g in np.unique(groups_arr[val_idx]):
                subject_scores[str(g)] = float(np.mean(probs[groups_arr[val_idx] == g]))
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / f"mamba_rbd_seed{seed}_fold{fold + 1}.pt")

        metrics = subject_metrics(subject_scores, subject_labels)
        per_seed.append({"seed": seed, **metrics})
        predictions.extend(
            {"seed": seed, "subject_id": sid, "label": subject_labels[sid], "score": score}
            for sid, score in sorted(subject_scores.items())
        )
        print(f"[seed {seed}] " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    summary = {
        key: (float(np.nanmean([r[key] for r in per_seed])), float(np.nanstd([r[key] for r in per_seed])))
        for key in ("auc", "accuracy", "sensitivity", "specificity")
    }
    return {"per_seed": per_seed, "summary": summary, "predictions": predictions}


def run_training(
    data_dir: str = "data/raw/capslpdb",
    epochs: int = 10,
    batch_size: int = 16,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_splits: int | None = None,
    output_csv: str = "reports/mamba_rbd_cv.csv",
    checkpoint_dir: str | None = None,
):
    from src.data.rbd_dataset import RBDEpochDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Rozpoczynam trening na: {device}")

    subjects = select_subjects(data_dir)
    groups_present = {subject_group(s) for s in subjects}
    if groups_present != {"rbd", "n"}:
        print(f"[!] Potrzeba pacjentow z obu grup rbd i n z plikami EDF+TXT w {data_dir}; mam: {subjects}")
        return None

    dataset = RBDEpochDataset(subject_ids=subjects, data_dir=data_dir)
    labels = np.array([s["label"] for s in dataset.samples])
    result = cross_validate(
        dataset, dataset.groups, labels, list(seeds), epochs,
        batch_size=batch_size, n_splits=n_splits, device=device,
        model_factory=lambda: MambaRBDClassifier(in_channels=dataset.n_channels, d_model=64, n_layers=2),
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
    )
    print("\n=== Poziom pacjenta, mean ± std po seedach ===")
    for key, (mean, std) in result["summary"].items():
        print(f"{key:12s} {mean:.3f} ± {std:.3f}")

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "subject_id", "label", "score"])
        writer.writeheader()
        writer.writerows(result["predictions"])
    print(f"Predykcje per pacjent: {out}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/capslpdb")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-splits", type=int, default=None, help="brak = leave-one-subject-out")
    parser.add_argument("--output", default="reports/mamba_rbd_cv.csv")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()
    run_training(args.data_dir, args.epochs, args.batch_size, tuple(args.seeds), args.n_splits, args.output, args.checkpoint_dir)
