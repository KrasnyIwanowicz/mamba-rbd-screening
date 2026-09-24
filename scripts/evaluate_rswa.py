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
    "rswa_mini_index_ecg_gated": True,
}

# Kontrole negatywne: wielkosci, ktore NIE mierza RSWA. Jesli ktoras z nich
# rozdziela rbd od n rownie dobrze jak metryka RSWA, ta metryka moze mierzyc
# sprzet/wzmocnienie/przesluch EKG, a nie atonie. Tu liczy sie "separowalnosc"
# max(AUC, 1-AUC) -- kierunek nie ma znaczenia, kazda roznica jest podejrzana.
NEGATIVE_CONTROLS = {
    "nrem_baseline_rms": True,        # ogolny poziom EMG brody w NREM [V] -- zalezy od wzmocnienia
    "rem_background_rms": True,       # atoniczne tlo REM [V] -- j.w.
    "ecg_contamination_ratio": True,  # przesluch EKG do EMG brody
}
# Metryki zalezne od BEZWZGLEDNEJ amplitudy -- porownywane z kontrolami amplitudy.
AMPLITUDE_DEPENDENT = {"rem_atonia_index"}
CONFOUND_WARNING_SEPARABILITY = 0.8


def evaluate_csv(scores_csv: str | Path, output_csv: str | Path) -> list[dict]:
    df = pd.read_csv(scores_csv)
    df = df[df["group"].isin(["rbd", "n"])]
    labels = (df["group"] == "rbd").astype(int).to_numpy()
    rows: list[dict] = []
    n_total, ci_missing = 0, True
    for metric, higher_means_rbd in {**METRICS, **NEGATIVE_CONTROLS}.items():
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
            "role": "negative_control" if metric in NEGATIVE_CONTROLS else "rswa_metric",
            "separability": round(max(r.auc, 1.0 - r.auc), 3),
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
        for warning in confound_warnings(rows):
            print(warning)
    return rows


def confound_warnings(rows: list[dict]) -> list[str]:
    """Ostrzezenia, gdy kontrola negatywna rozdziela grupy tak dobrze jak metryki RSWA."""
    controls = {r["metric"]: r["separability"] for r in rows if r["role"] == "negative_control"}
    warnings = []
    for name, sep in controls.items():
        if sep < CONFOUND_WARNING_SEPARABILITY:
            continue
        warnings.append(
            f"UWAGA: kontrola negatywna {name} rozdziela rbd/n z separowalnoscia {sep:.2f} -- "
            "grupy roznia sie czyms, co nie jest RSWA."
        )
        if name in ("nrem_baseline_rms", "rem_background_rms"):
            for r in rows:
                if r["metric"] in AMPLITUDE_DEPENDENT and r["separability"] <= sep + 0.05:
                    warnings.append(
                        f"  -> {r['metric']} zalezy od bezwzglednej amplitudy (progi w uV) i nie rozdziela "
                        f"lepiej ({r['separability']:.2f}) niz sam poziom sygnalu: nie interpretowac jako RSWA."
                    )
        if name == "ecg_contamination_ratio":
            warnings.append("  -> porownaj rswa_mini_index z rswa_mini_index_ecg_gated (EKG wyciete).")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="reports/rswa_scores.csv")
    parser.add_argument("--output", default="reports/rswa_evaluation.csv")
    args = parser.parse_args()
    return 0 if evaluate_csv(args.scores, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
