import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mamba_rbd import MambaRBDClassifier
from src.data.rbd_dataset import RBDEpochDataset
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Urządzenie: {device}")

    # 1. Inicjalizacja modelu
    model = MambaRBDClassifier(in_channels=2, d_model=64, n_layers=2).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Zbudowano model MambaRBDClassifier (Liczba parametrów: {total_params:,})")

    # 2. Pobranie batcha z datasetu
    dataset = RBDEpochDataset(subject_ids=["rbd1"], include_legs=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    signals = batch["signal"].to(device)  # [8, 2, 6000]
    labels = batch["label"].to(device)

    # 3. Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(signals)
        probs = torch.softmax(logits, dim=-1)

    print(f"[+] Wejście (EMG):    {signals.shape}")
    print(f"[+] Wyjście (Logity): {logits.shape}")
    print(f"[+] Prawdopodobieństwo RBD dla próbki: {probs[:, 1].tolist()}")


if __name__ == "__main__":
    main()