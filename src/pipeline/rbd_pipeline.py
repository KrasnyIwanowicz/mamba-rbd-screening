"""
End-to-end screening jednej nocy CAP: (stadia) -> epoki REM -> metryki RSWA -> wynik nocy.

To daje WYNIK RYZYKA do dalszej oceny klinicznej, nie diagnoze (README,
docs/technical_premise.md). Konkretnie:
- metryki regulowe RSWA (src/rswa_scoring.py) liczone sa zawsze;
- wynik modelu Mamba jest opcjonalny i TYLKO z realnego checkpointu --
  brak pliku to blad, nie ciche losowe wagi (poprzednia wersja przy literowce
  w sciezce po cichu inicjalizowala model losowo i drukowala "RBD"/"CONTROL");
- etykieta decyzyjna tylko gdy podano prog skalibrowany w LOSO
  (scripts/evaluate_rswa.py); bez tego "UNCALIBRATED". Dawny prog 0.3
  opisany jako "kryterium kliniczne" nim nie byl.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.cap_loader import CAPSleepLoader, EpochData
from src.models.mamba_rbd import MambaRBDClassifier
from src.rswa_scoring import rem_atonia_index, score_rswa_mini_epochs
from src.staging.cap_stager import REM_INDEX, load_stager, predict_stages, prepare_eeg_epochs


@dataclass
class ScreeningResult:
    subject_id: str
    total_epochs: int
    rem_epochs_detected: int
    rem_source: str  # "reference_hypnogram" albo "auto_stager"
    rswa_mini_index: float  # odsetek 3-s mini-epok REM z aktywnoscia EMG
    tonic_epoch_fraction: float
    rem_atonia_index: float  # RAI, 1 = pelna atonia
    model_rbd_score: float | None  # srednie p(pacjent RBD) po epokach REM; None bez checkpointu
    predicted_label: str  # "ELEVATED_RSWA" / "NORMAL_RSWA" / "UNCALIBRATED" / "NO_REM_DETECTED"


class RBDScreeningPipeline:
    def __init__(
        self,
        rbd_model_path: str | Path | None = None,
        use_auto_stager: bool = False,
        stager_checkpoint: str | Path | None = None,
        stager_encoder: str = "mamba",
        target_fs: int = 200,
        rswa_mini_threshold: float | None = None,
    ):
        self.target_fs = target_fs
        self.use_auto_stager = use_auto_stager
        self.rswa_mini_threshold = rswa_mini_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rbd_model: MambaRBDClassifier | None = None
        if rbd_model_path is not None:
            path = Path(rbd_model_path)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint modelu RBD nie istnieje: {path}")
            self.rbd_model = MambaRBDClassifier(in_channels=2, d_model=64, n_layers=2).to(self.device)
            self.rbd_model.load_state_dict(torch.load(path, map_location=self.device))
            self.rbd_model.eval()

        self.stager = None
        if use_auto_stager:
            if stager_checkpoint is None:
                raise ValueError("use_auto_stager=True wymaga stager_checkpoint")
            self.stager = load_stager(stager_checkpoint, stager_encoder, device=self.device)

    @staticmethod
    def _preprocess_channel(sig: np.ndarray) -> np.ndarray:
        std = np.std(sig)
        if std > 1e-6:
            return (sig - np.mean(sig)) / std
        return sig - np.mean(sig)

    def _select_rem(self, epochs: list[EpochData]) -> list[EpochData]:
        if self.stager is None:
            return [e for e in epochs if e.stage == "REM"]
        eeg = [e.eeg_central for e in epochs if e.eeg_central is not None]
        if len(eeg) != len(epochs):
            raise ValueError("Auto-stager wymaga kanalu EEG, a loader go nie znalazl.")
        X = prepare_eeg_epochs(eeg, fs_in=self.target_fs)
        stages = predict_stages(self.stager, X, device=self.device)
        return [e for e, s in zip(epochs, stages) if s == REM_INDEX]

    def _model_score(self, rem_epochs: list[EpochData]) -> float | None:
        if self.rbd_model is None:
            return None
        batch = []
        for ep in rem_epochs:
            chin = self._preprocess_channel(ep.emg_chin)
            leg = self._preprocess_channel(ep.emg_leg) if ep.emg_leg is not None else np.zeros_like(chin)
            batch.append(np.stack([chin, leg], axis=0))
        tensor = torch.tensor(np.array(batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = torch.softmax(self.rbd_model(tensor), dim=-1)[:, 1].cpu().numpy()
        return float(np.mean(probs))

    def screen_epochs(self, subject_id: str, epochs: list[EpochData]) -> ScreeningResult:
        """Rdzen pipeline'u na juz wczytanych epokach (EMG po emg_bandpass)."""
        rem_source = "auto_stager" if self.stager is not None else "reference_hypnogram"
        rem_epochs = self._select_rem(epochs) if epochs else []
        if not rem_epochs:
            nan = float("nan")
            return ScreeningResult(subject_id, len(epochs), 0, rem_source, nan, nan, nan, None, "NO_REM_DETECTED")

        rem_signals = [e.emg_chin for e in rem_epochs]
        mini = score_rswa_mini_epochs(rem_signals, self.target_fs)
        rai = rem_atonia_index(rem_signals, self.target_fs, epoch_starts_s=np.array([e.start_sec for e in rem_epochs]))

        if self.rswa_mini_threshold is None:
            label = "UNCALIBRATED"
        else:
            label = "ELEVATED_RSWA" if mini.rswa_mini_index >= self.rswa_mini_threshold else "NORMAL_RSWA"

        return ScreeningResult(
            subject_id=subject_id,
            total_epochs=len(epochs),
            rem_epochs_detected=len(rem_epochs),
            rem_source=rem_source,
            rswa_mini_index=round(mini.rswa_mini_index, 4),
            tonic_epoch_fraction=round(mini.tonic_epoch_fraction, 4),
            rem_atonia_index=round(rai, 4),
            model_rbd_score=self._model_score(rem_epochs),
            predicted_label=label,
        )

    def screen_subject(self, data_dir: str | Path, subject_id: str) -> ScreeningResult:
        """Przeprowadza pelna analize pacjenta z plikow CAP."""
        loader = CAPSleepLoader(data_dir=data_dir, target_fs=self.target_fs, preprocess_emg=True)
        epochs = loader.load_subject(subject_id, stages_filter=None)
        return self.screen_epochs(subject_id, epochs)
