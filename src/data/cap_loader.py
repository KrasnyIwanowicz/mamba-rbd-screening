from dataclasses import dataclass
from pathlib import Path
import re
import warnings
import mne
import numpy as np
import pandas as pd

# Kotwiczymy parser na jednym, stałoformatowym tokenie -- czasie hh:mm:ss --
# zamiast dzielić linię po białych znakach. Patrz uzasadnienie w docstringu
# parse_txt_annotations() poniżej.
_TIME_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")


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
        """Parsuje plik tekstowy adnotacji stadiów snu z CAP (REMlogic export).

        POPRAWKA (2026-08-28): poprzednia wersja dzieliła linię przez
        `re.split(r"\\s+", line)` i zakładała, że parts[2] to Duration.
        Realny format wiersza to:
            Stage \\t Position \\t Time[hh:mm:ss] \\t Event \\t Duration[s] \\t Location
        Pole Position może zawierać spację ("Unknown Position"), a NAWET
        gdy jest jednym słowem ("Left"), parts[2] to i tak Time, nie
        Duration. W obu przypadkach `float(parts[2])` rzuca ValueError na
        KAŻDYM wierszu -> df_stages wychodziło puste dla każdego pacjenta,
        a load_subject() cicho zwracało [] bez żadnego błędu. Sprawdzone
        na realnym rbd1.txt z physionet.org.

        Nowe podejście: kotwiczymy się na jedynym stałoformatowym tokenie w
        wierszu -- czasie hh:mm:ss. Wszystko przed nim to Stage+Position
        (Position może mieć spację), wszystko po nim to Event, Duration,
        Location -- te trzy pola NIE mają spacji, więc zwykły split() już
        działa bezpiecznie.

        Dodatkowo plik miesza wiersze makrostruktury (Event zaczyna się od
        "SLEEP-", Duration==30, jedna epoka na wiersz) z wierszami CAP
        mikrostruktury (Event zaczyna się od "MCAP-A1/A2/A3", zmienny czas
        trwania, mogą wypadać W TEJ SAMEJ epoce co wiersz SLEEP-*). Bez
        filtrowania po prefiksie "SLEEP-" te dodatkowe wiersze zostałyby
        policzone jako osobne "epoki" i zepsuły wyrównanie index-do-index
        z oknami wyciętymi z sygnału w load_subject().
        """
        with open(txt_path, "r", encoding="latin-1") as f:
            lines = f.read().splitlines()

        header_idx = next((i for i, l in enumerate(lines) if "Sleep Stage" in l), None)
        if header_idx is None:
            warnings.warn(f"{txt_path.name}: nie znaleziono nagłówka 'Sleep Stage' -- zwracam pustą ramkę.")
            return pd.DataFrame(columns=["stage", "duration", "time"])

        records = []
        for line in lines[header_idx + 1 :]:
            line = line.strip()
            if not line:
                continue

            m = _TIME_RE.search(line)
            if not m:
                continue

            before = line[: m.start()].strip()
            after = line[m.end() :].strip()

            before_parts = before.split(None, 1)
            if not before_parts:
                continue
            stage_raw = before_parts[0]

            after_parts = after.split()
            if len(after_parts) < 3:
                continue
            event, duration_str, _location = after_parts[0], after_parts[1], after_parts[2]

            if not event.startswith("SLEEP-"):
                continue  # pomijamy zdarzenia CAP (MCAP-A1/A2/A3) -- to nie są 30s epoki

            try:
                duration = float(duration_str)
            except ValueError:
                continue

            stage_norm = self.STAGE_MAP.get(stage_raw, "UNKNOWN")
            if stage_norm == "UNKNOWN":
                warnings.warn(f"{txt_path.name}: nieznana etykieta stadium '{stage_raw}' w wierszu: {line!r}")

            records.append({"stage": stage_norm, "duration": duration, "time": m.group(0)})

        return pd.DataFrame(records, columns=["stage", "duration", "time"])

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
            if len(df_stages) == 0:
                warnings.warn(
                    f"[!] {subject_id}: 0 epok stadium wczytanych z {txt_path.name} -- "
                    f"to prawie na pewno błąd parsera, nie brak danych. Sprawdź plik ręcznie "
                    f"przed zaufaniem pustemu wynikowi."
                )
                return []
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