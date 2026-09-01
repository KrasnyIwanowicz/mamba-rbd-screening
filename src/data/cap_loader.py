# -*- coding: utf-8 -*-
"""
src/data/cap_loader.py
Kompletny loader dla CAP Sleep Database (EDF + RemLogic TXT).

Historia: poprzednia wersja (CAPSleepDataset, reczny split po bialych
znakach) miala bug, przez ktory df_stages zawsze wychodzilo puste, bo
"Unknown Position" ma spacje w srodku, a kod zakladal sztywna pozycje
kolumny. Ta wersja parsuje przez pandas.read_csv z separatorem "jeden lub
wiecej tabulatorow" (sep=r"\t+") -- poniewaz prawdziwym separatorem kolumn
w tym pliku sa tabulatory, a nie spacje, spacja wewnatrz "Unknown Position"
nie jest traktowana jako granica kolumny. Zweryfikowane na syntetycznych
danych w tym samym formacie co realny plik z physionet.org (2026-08-29):
poprawnie zachowuje "Unknown Position" jako jedna wartosc i poprawnie
filtruje wiersze mikrostruktury CAP (MCAP-A1/A2/A3) przez
Event.str.startswith("SLEEP-").
"""

from dataclasses import dataclass
from pathlib import Path
import re
import mne
import numpy as np
import pandas as pd


@dataclass
class EpochData:
    subject_id: str
    epoch_idx: int
    stage: str
    start_sec: float
    duration_sec: float
    emg_chin: np.ndarray  # [samples]
    emg_leg: np.ndarray | None  # [samples] lub None
    eeg_central: np.ndarray | None  # [samples] lub None
    sampling_rate: int
    is_rbd: bool


class CAPSleepLoader:
    STAGE_MAP = {
        "W": "WAKE",
        "S0": "WAKE",
        "S1": "N1",
        "S2": "N2",
        "S3": "N3",
        "S4": "N3",
        "REM": "REM",
        "R": "REM",
        "MT": "MOVEMENT",
        "UNSCORED": "UNKNOWN",
        "?": "UNKNOWN",
    }

    def __init__(self, data_dir: str | Path, target_fs: int = 200, epoch_sec: int = 30):
        self.data_dir = Path(data_dir)
        self.target_fs = target_fs
        self.epoch_sec = epoch_sec

    def parse_remlogic_txt(self, txt_path: Path) -> pd.DataFrame:
        """Parsuje plik RemLogic Event Export i zwraca epoki stadiow snu.

        Separator to "jeden lub wiecej tabulatorow" (sep=r"\t+"), NIE
        biale znaki ogolnie -- to jest kluczowe, bo pole Position bywa
        wieloczlonowe ("Unknown Position") i zawiera spacje, ktora nie
        jest tabulatorem, wiec nie rozbija kolumny.
        """
        with open(txt_path, "r", encoding="latin-1") as f:
            lines = f.readlines()

        header_idx = -1
        for idx, line in enumerate(lines):
            if "Sleep Stage" in line and "Time [hh:mm:ss]" in line:
                header_idx = idx
                break

        if header_idx == -1:
            raise ValueError(f"Nie znaleziono naglowka tabeli w {txt_path}")

        df = pd.read_csv(
            txt_path,
            skiprows=header_idx,
            sep=r"\t+",
            engine="python",
            encoding="latin-1",
        )
        df.columns = [c.strip() for c in df.columns]

        # Filtrujemy tylko zdarzenia stadiow snu (ignorujemy mikrostrukture CAP: MCAP-A1/A2/A3)
        if "Event" in df.columns:
            df = df[df["Event"].str.startswith("SLEEP-", na=False)].copy()

        df["stage_clean"] = df["Sleep Stage"].astype(str).str.strip().map(
            lambda s: self.STAGE_MAP.get(s, "UNKNOWN")
        )

        # UWAGA: start_sec liczony z POZYCJI wiersza (i * epoch_sec), nie z
        # kolumny Time -- zaklada, ze po odfiltrowaniu MCAP-* zostaja
        # wylacznie ciagle, kolejne 30-sekundowe epoki bez przerw. To
        # standardowe zalozenie dla hipnogramu AASM/R&K, ale CAP to
        # archiwum wielu laboratoriow na przestrzeni lat (patrz
        # docs/technical_premise.md) -- jesli ktorys realny plik ma luke w
        # wyniku (np. brakujacy fragment nagrania), start_sec przestanie
        # sie zgadzac z kolumna Time. Warto to sprawdzic na kilku
        # pierwszych realnych plikach zanim zaufa sie temu bezkrytycznie
        # na calym zbiorze.
        df["duration_clean"] = pd.to_numeric(df["Duration[s]"], errors="coerce").fillna(self.epoch_sec)
        df["start_sec"] = np.arange(len(df)) * self.epoch_sec

        return df[["stage_clean", "start_sec", "duration_clean"]]

    def _find_channel(self, ch_names: list[str], patterns: list[str]) -> str | None:
        for pat in patterns:
            for ch in ch_names:
                if re.search(pat, ch):
                    return ch
        return None

    def load_subject(self, subject_id: str, stages_filter: list[str] | None = None) -> list[EpochData]:
        """
        Wczytuje sygnaly i adnotacje dla danego pacjenta.
        np. stages_filter=['REM'] wyciaga tylko epoki REM.
        stages_filter=None (domyslnie) zwraca WSZYSTKIE epoki -- potrzebne
        np. do policzenia linii bazowej NREM w src/rswa_scoring.py.
        """
        edf_path = self.data_dir / f"{subject_id}.edf"
        txt_path = self.data_dir / f"{subject_id}.txt"

        if not edf_path.exists() or not txt_path.exists():
            raise FileNotFoundError(f"Brak pliku EDF lub TXT dla {subject_id}")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        ch_names = raw.ch_names

        chin_ch = self._find_channel(ch_names, [r"(?i)chin", r"(?i)submental", r"(?i)emg1[-_]emg2"])
        leg_ch = self._find_channel(ch_names, [r"(?i)dx\d*[-_]dx\d*", r"(?i)dx", r"(?i)tibial"])
        eeg_ch = self._find_channel(ch_names, [r"(?i)c4[-_]a1", r"(?i)c3[-_]a2", r"(?i)c4", r"(?i)c3"])

        if chin_ch is None:
            raise ValueError(f"Pacjent {subject_id} nie posiada kanalu Chin EMG w {ch_names}")

        if raw.info["sfreq"] != self.target_fs:
            raw.resample(self.target_fs, npad="auto")

        df_stages = self.parse_remlogic_txt(txt_path)
        is_rbd = subject_id.lower().startswith("rbd")

        epoch_samples = int(self.epoch_sec * self.target_fs)
        epochs = []

        chin_data = raw.get_data(picks=[chin_ch])[0]
        leg_data = raw.get_data(picks=[leg_ch])[0] if leg_ch else None
        eeg_data = raw.get_data(picks=[eeg_ch])[0] if eeg_ch else None
        total_samples = len(chin_data)

        for i, row in df_stages.iterrows():
            stage = row["stage_clean"]
            if stages_filter and stage not in stages_filter:
                continue

            start_idx = int(row["start_sec"] * self.target_fs)
            end_idx = start_idx + epoch_samples

            if end_idx > total_samples:
                break

            epochs.append(
                EpochData(
                    subject_id=subject_id,
                    epoch_idx=i,
                    stage=stage,
                    start_sec=row["start_sec"],
                    duration_sec=row["duration_clean"],
                    emg_chin=chin_data[start_idx:end_idx],
                    emg_leg=leg_data[start_idx:end_idx] if leg_data is not None else None,
                    eeg_central=eeg_data[start_idx:end_idx] if eeg_data is not None else None,
                    sampling_rate=self.target_fs,
                    is_rbd=is_rbd,
                )
            )

        return epochs
