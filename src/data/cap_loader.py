from dataclasses import dataclass
from pathlib import Path
import re
import mne
import numpy as np
import pandas as pd


@dataclass
class EpochRecord:
    subject_id: str
    epoch_idx: int
    stage: str
    emg_signal: np.ndarray  # [channels, samples]
    eeg_signal: np.ndarray | None  # [channels, samples]
    sampling_rate: int
    is_rbd: bool


class CAPSleepDataset:
    STAGE_MAP = {
        "W": "WAKE",
        "S1": "N1",
        "S2": "N2",
        "S3": "N3",
        "S4": "N3",
        "REM": "REM",
        "R": "REM",
        "MT": "MOVEMENT"
    }

    def __init__(self, data_dir: str | Path, target_fs: int = 200, epoch_len_sec: int = 30):
        self.data_dir = Path(data_dir)
        self.target_fs = target_fs
        self.epoch_len_sec = epoch_len_sec
        self.epoch_samples = target_fs * epoch_len_sec

    def parse_txt_annotations(self, txt_path: Path) -> pd.DataFrame:
        """Parsuje plik tekstowy adnotacji stadiów snu z CAP."""
        records = []
        with open(txt_path, "r", encoding="latin-1") as f:
            lines = f.readlines()

        start_reading = False
        for line in lines:
            line = line.strip()
            if "Sleep Stage" in line or "STAGE" in line:
                start_reading = True
                continue
            if not start_reading or not line:
                continue

            parts = re.split(r"\s+", line)
            # Format w CAP to zazwyczaj: [Sleep Stage, Time [hh:mm:ss], Duration[s], Position...]
            if len(parts) >= 3:
                stage_raw = parts[0]
                stage_norm = self.STAGE_MAP.get(stage_raw, "UNKNOWN")
                try:
                    duration = float(parts[2])
                    records.append({"stage": stage_norm, "duration": duration})
                except ValueError:
                    continue

        return pd.DataFrame(records)

    def load_subject(self, subject_id: str, rem_only: bool = True) -> list[EpochRecord]:
        """Wczytuje sygnały EMG/EEG i adnotacje dla danego pacjenta."""
        edf_path = self.data_dir / f"{subject_id}.edf"
        txt_path = self.data_dir / f"{subject_id}.txt"

        if not edf_path.exists():
            raise FileNotFoundError(f"Brak pliku {edf_path}")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        
        # Wykrywanie kanału EMG brody
        chin_candidates = [ch for ch in raw.ch_names if re.search(r"(?i)chin|submental|emg1", ch)]
        if not chin_candidates:
            print(f"[!] Pomijanie {subject_id}: brak wykrytego EMG brody.")
            return []
        
        chin_ch = chin_candidates[0]
        raw.pick_channels([chin_ch])

        # Resampling do target_fs (np. 200 Hz)
        if raw.info["sfreq"] != self.target_fs:
            raw.resample(self.target_fs, npad="auto")

        emg_data = raw.get_data()  # [1, total_samples]
        total_epochs = emg_data.shape[1] // self.epoch_samples

        epochs = []
        is_rbd = subject_id.lower().startswith("rbd")

        # Jeśli są adnotacje .txt
        if txt_path.exists():
            df_stages = self.parse_txt_annotations(txt_path)
            for i in range(min(total_epochs, len(df_stages))):
                stage = df_stages.iloc[i]["stage"]
                if rem_only and stage != "REM":
                    continue

                start = i * self.epoch_samples
                end = start + self.epoch_samples
                epoch_sig = emg_data[:, start:end]

                epochs.append(
                    EpochRecord(
                        subject_id=subject_id,
                        epoch_idx=i,
                        stage=stage,
                        emg_signal=epoch_sig,
                        eeg_signal=None,
                        sampling_rate=self.target_fs,
                        is_rbd=is_rbd
                    )
                )
        return epochs