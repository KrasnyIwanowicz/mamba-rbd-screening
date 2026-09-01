from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from src.data.cap_loader import CAPSleepLoader, EpochData


class RBDEpochDataset(Dataset):
    def __init__(
        self,
        subject_ids: list[str],
        data_dir: str | Path = "data/raw/capslpdb",
        target_fs: int = 200,
        include_legs: bool = True,
        normalize: bool = True
    ):
        self.loader = CAPSleepLoader(data_dir=data_dir, target_fs=target_fs)
        self.include_legs = include_legs
        self.normalize = normalize
        self.samples: list[dict] = []

        self._load_all_subjects(subject_ids)

    def _preprocess_signal(self, sig: np.ndarray) -> np.ndarray:
        # Filtracja pasmowa / standaryzacja z-score per epoka
        if self.normalize:
            std = np.std(sig)
            if std > 1e-6:
                sig = (sig - np.mean(sig)) / std
            else:
                sig = sig - np.mean(sig)
        return sig.astype(np.float32)

    def _load_all_subjects(self, subject_ids: list[str]):
        print(f"[*] Budowanie datasetu dla {len(subject_ids)} pacjentów...")
        for sub_id in subject_ids:
            try:
                rem_epochs = self.loader.load_subject(sub_id, stages_filter=["REM"])
                for ep in rem_epochs:
                    chin = self._preprocess_signal(ep.emg_chin)
                    
                    if self.include_legs and ep.emg_leg is not None:
                        leg = self._preprocess_signal(ep.emg_leg)
                        # Połączenie w 2 kanały: [2, 6000]
                        stacked_emg = np.stack([chin, leg], axis=0)
                    else:
                        # 1 kanał: [1, 6000]
                        stacked_emg = np.expand_dims(chin, axis=0)

                    self.samples.append({
                        "subject_id": sub_id,
                        "epoch_idx": ep.epoch_idx,
                        "signal": stacked_emg,
                        "label": 1 if ep.is_rbd else 0  # 1 = RBD, 0 = Healthy/Control
                    })
            except Exception as e:
                print(f"[!] Pomijanie {sub_id}: {e}")

        print(f"[+] Załadowano łącznie {len(self.samples)} epok REM.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return {
            "signal": torch.from_numpy(sample["signal"]),  # Tensor [C, T]
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "subject_id": sample["subject_id"],
            "epoch_idx": sample["epoch_idx"]
        }