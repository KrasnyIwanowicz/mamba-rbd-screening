"""Rule-based RSWA metrics per CAP subject -> reports/rswa_scores.csv.

Trzy metryki na pacjenta (wszystkie z EMG brody po emg_bandpass, 10-90 Hz @ 200 Hz):
- rswa_index:          epoka 30 s REM vs 2x mediana NREM (pierwotny baseline; zachowany do porownania)
- rswa_mini_index:     odsetek 3-s mini-epok REM > 2x atoniczne tlo REM (SINBAR-podobne, "any")
- rem_atonia_index:    RAI (Ferri 2008/2010), 1 = pelna atonia, nizszy = wiecej aktywnosci
Kontrole artefaktu EKG (src/artifacts.py):
- ecg_contamination_ratio:     |EMG| przy zalamkach R / |EMG| ogolem w REM (~1 = brak przesluchu)
- rswa_mini_index_ecg_gated:   rswa_mini_index po wycieciu +-50 ms wokol kazdego R
Ewaluacja rbd vs n: scripts/evaluate_rswa.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import csv
import warnings

import numpy as np

from src.artifacts import ecg_blanking_masks, ecg_contamination_ratio
from src.data.cap_loader import CAPSleepLoader, subject_group
from src.rswa_scoring import rem_atonia_index, score_rswa, score_rswa_mini_epochs

NREM_STAGES = {"N2", "N3"}
DEFAULT_GROUP_COUNTS = {"rbd": 22, "n": 16}

FIELDNAMES = [
    "subject_id", "group", "n_rem_epochs", "n_nrem_epochs",
    "nrem_baseline_rms", "rswa_index",
    "rem_background_rms", "rswa_mini_index", "tonic_epoch_fraction", "rem_atonia_index",
    "ecg_contamination_ratio", "rswa_mini_index_ecg_gated",
    "skipped_reason",
]


def compute_subject_rswa(loader, subject_id: str) -> dict:
    try:
        epochs = loader.load_subject(subject_id, stages_filter=None)
    except (FileNotFoundError, ValueError) as e:
        return {"subject_id": subject_id, "skipped_reason": str(e)}

    if not epochs:
        return {"subject_id": subject_id, "skipped_reason": "load_subject zwrocilo 0 epok"}

    rem = [e for e in epochs if e.stage == "REM"]
    nrem_signals = [e.emg_chin for e in epochs if e.stage in NREM_STAGES]

    if not nrem_signals:
        return {"subject_id": subject_id, "skipped_reason": "brak epok N2/N3 -- nie da sie policzyc linii bazowej"}
    if not rem:
        return {"subject_id": subject_id, "skipped_reason": "brak epok REM"}

    rem_signals = [e.emg_chin for e in rem]
    fs = float(rem[0].sampling_rate)
    result = score_rswa(rem_signals, nrem_signals)
    mini = score_rswa_mini_epochs(rem_signals, fs)
    rai = rem_atonia_index(rem_signals, fs, epoch_starts_s=np.array([e.start_sec for e in rem]))

    ecg = [x for x in (getattr(e, "ecg", None) for e in rem) if x is not None]
    ecg_ratio, mini_gated = float("nan"), float("nan")
    if len(ecg) == len(rem):
        ecg_ratio = ecg_contamination_ratio(rem_signals, ecg, fs)
        mini_gated = score_rswa_mini_epochs(rem_signals, fs, exclude_masks=ecg_blanking_masks(ecg, fs)).rswa_mini_index

    return {
        "subject_id": subject_id,
        "group": subject_group(subject_id),
        "n_rem_epochs": len(rem_signals),
        "n_nrem_epochs": len(nrem_signals),
        "nrem_baseline_rms": result.nrem_baseline_rms,
        "rswa_index": result.rswa_index,
        "rem_background_rms": mini.background_rms,
        "rswa_mini_index": mini.rswa_mini_index,
        "tonic_epoch_fraction": mini.tonic_epoch_fraction,
        "rem_atonia_index": rai,
        "ecg_contamination_ratio": ecg_ratio,
        "rswa_mini_index_ecg_gated": mini_gated,
        "skipped_reason": "",
    }


def run(data_dir: str | Path, subject_ids: list[str], output_csv: str | Path) -> list[dict]:
    loader = CAPSleepLoader(data_dir=data_dir, preprocess_emg=True)
    rows = []
    for subject_id in subject_ids:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            row = compute_subject_rswa(loader, subject_id)
            for w in caught:
                print(f"[{subject_id}] {w.message}")
        rows.append(row)
        if row.get("skipped_reason"):
            status = row["skipped_reason"]
        else:
            status = (
                f"rswa_index={row['rswa_index']:.3f} mini={row['rswa_mini_index']:.3f} "
                f"RAI={row['rem_atonia_index']:.3f} ECG_ratio={row['ecg_contamination_ratio']:.2f} "
                f"mini_gated={row['rswa_mini_index_ecg_gated']:.3f} (n_rem={row['n_rem_epochs']}, n_nrem={row['n_nrem_epochs']})"
            )
        print(f"{subject_id}: {status}")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})

    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/capslpdb", help="Folder z pobranymi .edf/.txt z CAP Sleep Database")
    parser.add_argument("--groups", nargs="+", default=["rbd", "n"], help="Ktore grupy przetworzyc")
    parser.add_argument("--output", default="reports/rswa_scores.csv")
    parser.add_argument("--subjects", nargs="+", default=None, help="np. rbd1 rbd2 n1 (nadpisuje --groups)")
    args = parser.parse_args()

    subject_ids = args.subjects or [
        f"{g}{i}" for g in args.groups if g in DEFAULT_GROUP_COUNTS for i in range(1, DEFAULT_GROUP_COUNTS[g] + 1)
    ]

    run(args.data_dir, subject_ids, args.output)
    print(f"\nZapisano wyniki do {args.output}")
