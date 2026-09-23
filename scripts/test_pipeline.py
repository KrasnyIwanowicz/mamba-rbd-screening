"""Demo: screening jednej nocy CAP (domyslnie rbd1) -- wynik ryzyka, nie diagnoza."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rbd_pipeline import RBDScreeningPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_id", nargs="?", default="rbd1")
    parser.add_argument("--data-dir", default="data/raw/capslpdb")
    parser.add_argument("--rbd-model", default=None, help="checkpoint z src/training/train_rbd.py (opcjonalny)")
    parser.add_argument("--stager-checkpoint", default=None, help="podaj, aby wybierac REM auto-stagerem zamiast hipnogramu CAP")
    parser.add_argument("--rswa-threshold", type=float, default=None, help="prog rswa_mini_index skalibrowany w LOSO")
    args = parser.parse_args()

    pipeline = RBDScreeningPipeline(
        rbd_model_path=args.rbd_model,
        use_auto_stager=args.stager_checkpoint is not None,
        stager_checkpoint=args.stager_checkpoint,
        rswa_mini_threshold=args.rswa_threshold,
    )
    result = pipeline.screen_subject(data_dir=args.data_dir, subject_id=args.subject_id)

    print("\n" + "=" * 50)
    print("RAPORT SCREENINGU RSWA (wynik ryzyka, NIE diagnoza)")
    print("=" * 50)
    print(f"ID Pacjenta:                 {result.subject_id}")
    print(f"Zrodlo epok REM:             {result.rem_source}")
    print(f"Liczba epok REM:             {result.rem_epochs_detected} ({result.rem_epochs_detected * 0.5:.1f} min)")
    print(f"RSWA mini-epoki (3 s):       {result.rswa_mini_index * 100:.2f}%")
    print(f"Epoki toniczne (>=50%):      {result.tonic_epoch_fraction * 100:.2f}%")
    print(f"REM Atonia Index (RAI):      {result.rem_atonia_index:.3f}")
    score = "brak checkpointu" if result.model_rbd_score is None else f"{result.model_rbd_score:.4f}"
    print(f"Wynik modelu Mamba:          {score}")
    print(f"Etykieta:                    [{result.predicted_label}]")
    print("=" * 50)


if __name__ == "__main__":
    main()
