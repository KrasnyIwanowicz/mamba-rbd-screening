from dataclasses import dataclass
from pathlib import Path
import mne
import numpy as np
import torch
from src.data.cap_loader import CAPSleepLoader
from src.models.mamba_rbd import MambaRBDClassifier


@dataclass
class ScreeningResult:
    subject_id: str
    total_epochs: int
    rem_epochs_detected: int
    rswa_positive_epochs: int
    rswa_index: float  # Procent epok REM z RSWA
    rbd_probability: float  # Średnie prawdopodobieństwo RBD
    predicted_label: str  # "RBD" lub "CONTROL"


class RBDScreeningPipeline:
    def __init__(
        self,
        rbd_model_path: str | Path | None = None,
        use_auto_stager: bool = False,
        target_fs: int = 200,
        rswa_threshold: float = 0.5,
        patient_diag_threshold: float = 0.3  # >30% epok REM z RSWA = RBD (kryterium kliniczne)
    ):
        self.target_fs = target_fs
        self.rswa_threshold = rswa_threshold
        self.patient_diag_threshold = patient_diag_threshold
        self.use_auto_stager = use_auto_stager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inicjalizacja klasyfikatora Mamba
        self.rbd_model = MambaRBDClassifier(in_channels=2, d_model=64, n_layers=2).to(self.device)
        if rbd_model_path and Path(rbd_model_path).exists():
            self.rbd_model.load_state_dict(torch.load(rbd_model_path, map_location=self.device))
        self.rbd_model.eval()

    def _preprocess_channel(self, sig: np.ndarray) -> np.ndarray:
        std = np.std(sig)
        if std > 1e-6:
            return (sig - np.mean(sig)) / std
        return sig - np.mean(sig)

    def screen_subject(self, data_dir: str | Path, subject_id: str) -> ScreeningResult:
        """Przeprowadza pełną analizę pacjenta z plików bazowych."""
        loader = CAPSleepLoader(data_dir=data_dir, target_fs=self.target_fs)

        if not self.use_auto_stager:
            # Użycie adnotacji referencyjnych
            rem_epochs = loader.load_subject(subject_id, stages_filter=["REM"])
        else:
            raise NotImplementedError("Integracja z inferencją auto-stagera w toku.")

        if not rem_epochs:
            return ScreeningResult(
                subject_id=subject_id,
                total_epochs=0,
                rem_epochs_detected=0,
                rswa_positive_epochs=0,
                rswa_index=0.0,
                rbd_probability=0.0,
                predicted_label="NO_REM_DETECTED"
            )

        # Przygotowanie tensorów EMG dla epok REM
        batch_signals = []
        for ep in rem_epochs:
            chin = self._preprocess_channel(ep.emg_chin)
            leg = self._preprocess_channel(ep.emg_leg) if ep.emg_leg is not None else np.zeros_like(chin)
            stacked = np.stack([chin, leg], axis=0)
            batch_signals.append(stacked)

        tensor_data = torch.tensor(np.array(batch_signals), dtype=torch.float32).to(self.device)

        # Inferencja modelu Mamba
        with torch.no_grad():
            logits = self.rbd_model(tensor_data)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        rswa_positive = (probs >= self.rswa_threshold).sum()
        rswa_index = rswa_positive / len(probs)
        mean_rbd_prob = float(np.mean(probs))

        diagnosis = "RBD" if rswa_index >= self.patient_diag_threshold else "CONTROL"

        return ScreeningResult(
            subject_id=subject_id,
            total_epochs=len(rem_epochs),
            rem_epochs_detected=len(rem_epochs),
            rswa_positive_epochs=int(rswa_positive),
            rswa_index=round(float(rswa_index), 4),
            rbd_probability=round(mean_rbd_prob, 4),
            predicted_label=diagnosis
        )