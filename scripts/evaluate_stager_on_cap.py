"""Faza 2: zmierz spadek jakosci stagera (Sleep-EDF-20) na nagraniach CAP.

Porownuje predykcje SleepStager z hipnogramem CAP (R&K scorowany przez
ekspertow) per pacjent: accuracy, macro-F1, kappa, F1 per klasa (REM F1 jest
tu kluczowe -- od niego zalezy, ktore epoki trafia do detektora RSWA).
Punkt odniesienia z submodulu (Sleep-EDF-20, in-domain): Mamba 81.2% acc,
macro-F1 0.777. Rownolegle zapisuje okna REM (przewidziane i referencyjne).

Przyklad:
  python scripts/evaluate_stager_on_cap.py --checkpoint external/sleep_staging/results/mamba_best.pt \
      --eeg-channel "(?i)c4[-_]a1" --subjects rbd1 n1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from src.data.cap_loader import CAPSleepLoader, subject_group
from src.staging.cap_stager import (
    STAGER_CLASSES,
    STAGER_FS,
    cap_stages_to_indices,
    load_stager,
    predict_stages,
    prepare_eeg_epochs,
    rem_windows,
)

LABELS = list(range(len(STAGER_CLASSES)))


def stage_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    out = {
        "n_epochs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS)),
    }
    out.update({f"f1_{name}": float(v) for name, v in zip(STAGER_CLASSES, per_class)})
    return out


def evaluate_subject(loader: CAPSleepLoader, model, subject_id: str, scale_to_std: float | None = None) -> tuple[dict, dict]:
    epochs = loader.load_subject(subject_id, stages_filter=None)
    eeg = [e.eeg_central for e in epochs if e.eeg_central is not None]
    if not epochs or len(eeg) != len(epochs):
        raise ValueError(f"{subject_id}: brak epok albo brak kanalu EEG pasujacego do wzorca")
    X = prepare_eeg_epochs(eeg, fs_in=epochs[0].sampling_rate, scale_to_std=scale_to_std)
    y_pred = predict_stages(model, X)
    y_true = cap_stages_to_indices([e.stage for e in epochs])
    starts = np.array([e.start_sec for e in epochs])
    scored = y_true >= 0
    metrics = {"subject_id": subject_id, "group": subject_group(subject_id), **stage_metrics(y_true[scored], y_pred[scored])}
    windows = {
        "predicted_rem": rem_windows(y_pred, starts),
        "reference_rem": rem_windows(y_true, starts),
    }
    return metrics, {"y_true": y_true[scored], "y_pred": y_pred[scored], "windows": windows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="external/sleep_staging/results/mamba_best.pt")
    parser.add_argument("--sequence-encoder", default="mamba", choices=["mamba", "bilstm", "transformer"])
    parser.add_argument("--data-dir", default="data/raw/capslpdb")
    parser.add_argument("--subjects", nargs="+", default=[f"rbd{i}" for i in range(1, 23)] + [f"n{i}" for i in range(1, 17)])
    parser.add_argument("--eeg-channel", default=None, help="regex kanalu EEG (domyslnie wzorce loadera: C4-A1, C3-A2, ...)")
    parser.add_argument("--scale-to-std", type=float, default=None, help="opcjonalna adaptacja amplitudy [V], np. 2e-5")
    parser.add_argument("--output", default="reports/stager_transfer_cap.csv")
    parser.add_argument("--rem-windows", default="reports/stager_rem_windows.json")
    args = parser.parse_args()

    model = load_stager(args.checkpoint, args.sequence_encoder)
    loader = CAPSleepLoader(
        args.data_dir, target_fs=STAGER_FS,
        eeg_patterns=[args.eeg_channel] if args.eeg_channel else None,
    )
    rows, all_true, all_pred, windows = [], [], [], {}
    for sid in args.subjects:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                metrics, extra = evaluate_subject(loader, model, sid, args.scale_to_std)
            for w in caught:
                print(f"[{sid}] {w.message}")
        except (FileNotFoundError, ValueError) as e:
            print(f"[-] {sid}: {e}")
            continue
        rows.append(metrics)
        all_true.append(extra["y_true"])
        all_pred.append(extra["y_pred"])
        windows[sid] = extra["windows"]
        print(f"{sid}: acc={metrics['accuracy']:.3f} macroF1={metrics['macro_f1']:.3f} "
              f"kappa={metrics['kappa']:.3f} F1_REM={metrics['f1_REM']:.3f}")

    if not rows:
        print("[!] Zaden pacjent nie zostal oceniony.")
        return 1

    pooled = stage_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    rows.append({"subject_id": "POOLED", "group": "", **pooled})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.rem_windows).write_text(json.dumps(windows, indent=1), encoding="utf-8")

    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_pred), labels=LABELS)
    print("\nPOOLED: " + "  ".join(f"{k}={v:.3f}" for k, v in pooled.items() if k != "n_epochs"))
    print("Sleep-EDF-20 (in-domain, submodul): acc=0.812 macro_f1=0.777 -- roznica to zmierzony domain shift.")
    print("Macierz pomylek (wiersze = CAP, kolumny = stager):", STAGER_CLASSES)
    print(cm)
    print(f"\nZapisano {out} i {args.rem_windows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
