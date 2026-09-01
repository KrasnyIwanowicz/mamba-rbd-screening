import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader
from src.data.rbd_dataset import RBDEpochDataset


def main():
    # Na start testujemy na rbd1 (docelowo lista wszystkich rbd* i n*)
    subjects = ["rbd1"]
    
    dataset = RBDEpochDataset(subject_ids=subjects, include_legs=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    print("\n[*] Test pobierania batcha z DataLoader...")
    for batch in loader:
        signals = batch["signal"]
        labels = batch["label"]
        print(f"Batch signals shape: {signals.shape} (Format: [Batch, Kanały EMG, Próbki])")
        print(f"Batch labels:        {labels.tolist()} (1=RBD, 0=Healthy)")
        break


if __name__ == "__main__":
    main()