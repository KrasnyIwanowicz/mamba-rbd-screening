"""
src/training/train_rbd.py
Pętla treningowa dla MambaRBDClassifier z Group-KFold CV (podział per pacjent).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
import numpy as np
from src.data.rbd_dataset import RBDEpochDataset
from src.models.mamba_rbd import MambaRBDClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = out.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * len(y)
            preds = out.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total


def run_training(data_dir: str = "data/raw/capslpdb", epochs: int = 10, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Rozpoczynam trening na: {device}")

    # Lista dostępnych podmiotów
    edf_files = list(Path(data_dir).glob("*.edf"))
    available_subjects = sorted([f.stem for f in edf_files])
    
    if len(available_subjects) < 2:
        print("[!] Potrzebujesz co najmniej 2 pacjentów do walidacji (np. rbd1 i n1).")
        return

    dataset = RBDEpochDataset(subject_ids=available_subjects, data_dir=data_dir)
    
    # Grupy do GroupKFold (id pacjenta jako grupa)
    groups = [s["subject_id"] for s in dataset.samples]
    unique_groups = list(set(groups))
    n_splits = min(5, len(unique_groups))

    gkf = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(dataset))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        print(f"\n--- FOLD {fold + 1}/{n_splits} ---")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        model = MambaRBDClassifier(in_channels=2, d_model=64, n_layers=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoka {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        # Zapisujemy model z pierwszego foldu jako demonstracyjny
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/mamba_rbd_fold{fold+1}.pt")
        break  # Zatrzymujemy po 1 foldzie demonstracyjnie


if __name__ == "__main__":
    run_training()