from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from src.data.cap_loader import CAPSleepLoader


# UWAGA o etykiecie: "label" to diagnoza PACJENTA (rbd vs reszta) przypisana
# kazdej jego epoce REM -- NIE etykieta RSWA danej epoki (tej CAP nie ma).
# Model uczony na tym uczy sie "czy ta epoka pochodzi od pacjenta z RBD";
# sensowna ewaluacja jest wiec tylko na poziomie pacjenta i tylko z podzialem
# grupowym po pacjencie (src/training/train_rbd.py).


class RBDEpochDataset(Dataset):
    def __init__(
        self,
        subject_ids: list[str],
        data_dir: str | Path = "data/raw/capslpdb",
        target_fs: int = 200,
        include_legs: bool = True,
        normalize: bool = True
    ):
        # preprocess_emg=True: to samo pasmo 10-90 Hz co scoring regulowy i pipeline.
        self.loader = CAPSleepLoader(data_dir=data_dir, target_fs=target_fs, preprocess_emg=True)
        self.include_legs = include_legs
        self.normalize = normalize
        self.samples: list[dict] = []
        self.n_channels = 2 if include_legs else 1

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
                    
                    if self.include_legs:
                        # Zawsze 2 kanaly [2, T]: pacjent bez EMG nogi dostaje
                        # kanal zerowy (ta sama konwencja co RBDScreeningPipeline).
                        # Wczesniej mieszanka [1, T] i [2, T] wysypywala collate
                        # DataLoadera, gdy tylko jeden pacjent nie mial kanalu nogi.
                        leg = (
                            self._preprocess_signal(ep.emg_leg)
                            if ep.emg_leg is not None
                            else np.zeros_like(chin)
                        )
                        stacked_emg = np.stack([chin, leg], axis=0)
                    else:
                        stacked_emg = np.expand_dims(chin, axis=0)

                    self.samples.append({
                        "subject_id": sub_id,
                        "epoch_idx": ep.epoch_idx,
                        "signal": stacked_emg,
                        "label": 1 if ep.is_rbd else 0,  # diagnoza pacjenta, nie RSWA epoki
                        "has_leg": ep.emg_leg is not None,
                    })
            except Exception as e:
                print(f"[!] Pomijanie {sub_id}: {e}")

        print(f"[+] Załadowano łącznie {len(self.samples)} epok REM.")

    @property
    def groups(self) -> list[str]:
        """subject_id kazdej probki -- do GroupKFold / LOSO."""
        return [s["subject_id"] for s in self.samples]

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
