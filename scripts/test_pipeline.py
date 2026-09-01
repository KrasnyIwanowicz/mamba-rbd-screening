import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rbd_pipeline import RBDScreeningPipeline


def main():
    pipeline = RBDScreeningPipeline(
        rbd_model_path=None,  # Uruchomienie na wagach zainicjalizowanych lub po treningu
        rswa_threshold=0.5,
        patient_diag_threshold=0.3
    )

    result = pipeline.screen_subject(data_dir="data/raw/capslpdb", subject_id="rbd1")

    print("\n" + "="*50)
    print("AUTOMATYCZNY RAPORT SCREENINGU RBD (MAMBA)")
    print("="*50)
    print(f"ID Pacjenta:                 {result.subject_id}")
    print(f"Liczba epok REM:             {result.rem_epochs_detected} ({result.rem_epochs_detected * 0.5:.1f} min)")
    print(f"Epoki z utratą atonii (RSWA): {result.rswa_positive_epochs}")
    print(f"Indeks RSWA (RSWA %):        {result.rswa_index * 100:.2f}%")
    print(f"Prawdopodobieństwo RBD:      {result.rbd_probability:.4f}")
    print(f"Końcowa klasyfikacja:        [{result.predicted_label}]")
    print("="*50)


if __name__ == "__main__":
    main()