"""Subject-level RBD vs. control evaluation of every metric in reports/rswa_scores.csv."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.evaluate import evaluate_subject_metric

# Kierunek ustalony z gory (fizjologia), nie dobierany pod wynik.
METRICS = {
    "rswa_index": True,
    "rswa_mini_index": True,
    "tonic_epoch_fraction": True,
    "rem_atonia_index": False,  # RAI: nizszy = mniej atonii = bardziej RBD
}


def evaluate_csv(scores_csv: str | Path, output_csv: str | Path) -> list[dict]:
    df = pd.read_csv(scores_csv)
    df = df[df["group"].isin(["rbd", "n"])]
    labels = (df["group"] == "rbd").astype(int).to_numpy()
    rows: list[dict] = []
    n_total, ci_missing = 0, True
    for metric, higher_means_rbd in METRICS.items():
        if metric not in df.columns:
            print(f"[-] {metric}: brak kolumny w {scores_csv} -- uruchom ponownie scripts/run_rswa_pipeline.py")
            continue
        scores = pd.to_numeric(df[metric], errors="coerce").to_numpy()
        try:
            r = evaluate_subject_metric(scores, labels, metric, higher_means_rbd=higher_means_rbd)
        except ValueError as e:
            print(f"[!] {metric}: {e}")
            continue
        n_total, ci_missing = r.n_positive + r.n_negative, bool(np.isnan(r.auc_ci95[0]))
        rows.append({
            "metric": r.metric,
            "n_rbd": r.n_positive,
            "n_control": r.n_negative,
            "auc": round(r.auc, 3),
            "auc_ci95_low": round(r.auc_ci95[0], 3),
            "auc_ci95_high": round(r.auc_ci95[1], 3),
            "mannwhitney_p_one_sided": float(f"{r.mannwhitney_p:.3g}"),
            "loso_sensitivity": round(r.loso_sensitivity, 3),
            "loso_specificity": round(r.loso_specificity, 3),
            "loso_accuracy": round(r.loso_accuracy, 3),
        })
    if rows:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"\nZapisano do {output_csv}")
        if n_total < 20 or ci_missing:
            print("UWAGA: bardzo mala proba -- wyniki czysto orientacyjne.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="reports/rswa_scores.csv")
    parser.add_argument("--output", default="reports/rswa_evaluation.csv")
    args = parser.parse_args()
    return 0 if evaluate_csv(args.scores, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
