from __future__ import annotations
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import csv
import warnings

from src.data.cap_loader import CAPSleepLoader
from src.rswa_scoring import score_rswa

NREM_STAGES = {"N2", "N3"}


def compute_subject_rswa(loader: CAPSleepLoader, subject_id: str) -> dict:
    try:
        epochs = loader.load_subject(subject_id, stages_filter=None)
    except (FileNotFoundError, ValueError) as e:
        return {"subject_id": subject_id, "skipped_reason": str(e)}

    if not epochs:
        return {"subject_id": subject_id, "skipped_reason": "load_subject zwrocilo 0 epok"}

    rem_signals = [e.emg_chin for e in epochs if e.stage == "REM"]
    nrem_signals = [e.emg_chin for e in epochs if e.stage in NREM_STAGES]

    if not nrem_signals:
        return {"subject_id": subject_id, "skipped_reason": "brak epok N2/N3 -- nie da sie policzyc linii bazowej"}
    if not rem_signals:
        return {"subject_id": subject_id, "skipped_reason": "brak epok REM"}

    result = score_rswa(rem_signals, nrem_signals)
    is_rbd = epochs[0].is_rbd

    return {
        "subject_id": subject_id,
        "group": "rbd" if is_rbd else "n",
        "n_rem_epochs": len(rem_signals),
        "n_nrem_epochs": len(nrem_signals),
        "nrem_baseline_rms": result.nrem_baseline_rms,
        "rswa_index": result.rswa_index,
        "skipped_reason": "",
    }


def run(data_dir: str | Path, subject_ids: list[str], output_csv: str | Path) -> list[dict]:
    loader = CAPSleepLoader(data_dir=data_dir)
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
            status = f"rswa_index={row['rswa_index']:.3f} (n_rem={row['n_rem_epochs']}, n_nrem={row['n_nrem_epochs']})"
        print(f"{subject_id}: {status}")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id", "group", "n_rem_epochs", "n_nrem_epochs",
        "nrem_baseline_rms", "rswa_index", "skipped_reason",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/capslpdb", help="Folder z pobranymi .edf/.txt z CAP Sleep Database")
    parser.add_argument("--groups", nargs="+", default=["rbd", "n"], help="Ktore grupy przetworzyc")
    parser.add_argument("--output", default="reports/rswa_scores.csv")
    args = parser.parse_args()

    group_counts = {"rbd": 22, "n": 16}
    subject_ids = [
        f"{g}{i}" for g in args.groups if g in group_counts for i in range(1, group_counts[g] + 1)
    ]

    run(args.data_dir, subject_ids, args.output)
    print(f"\nZapisano wyniki do {args.output}")
