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
nie jest traktowana jako granica kolumny.

Wyrownanie czasowe (poprawka 2026-09-23): start epoki liczony jest z kolumny
Time [hh:mm:ss] wzgledem godziny startu nagrania w naglowku EDF, a NIE z
numeru wiersza. Poprzednio start_sec = i * 30 zakladal, ze (1) hipnogram
zaczyna sie dokladnie w chwili startu EDF i (2) nie ma luk w scoringu. Jesli
ktorekolwiek z tych zalozen nie zachodzi, KAZDA epoka jest przesunieta
wzgledem sygnalu -- i "REM" wycina fragment innego stadium. Obie sytuacje sa
teraz obslugiwane jawnie i raportowane przez warnings.warn, zamiast cicho
przesuwac dane.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.preprocessing import emg_bandpass

# Grupy w CAP Sleep Database. Kolejnosc ma znaczenie przy dopasowaniu
# prefiksu: "nfle"/"narco" musza byc sprawdzone przed "n" (zdrowi), inaczej
# pacjent z padaczka czolowa ("nfle3") zostalby policzony jako kontrola.
CAP_GROUPS = ("rbd", "nfle", "narco", "ins", "plm", "sdb", "brux", "n")
_SUBJECT_ID_RE = re.compile(r"^(?P<group>" + "|".join(CAP_GROUPS) + r")(?P<num>\d+)$", re.IGNORECASE)

CHIN_PATTERNS = [r"(?i)chin", r"(?i)submental", r"(?i)emg1[-_]emg2"]
LEG_PATTERNS = [r"(?i)dx\d*[-_]dx\d*", r"(?i)dx", r"(?i)tibial"]
EEG_PATTERNS = [r"(?i)c4[-_]a1", r"(?i)c3[-_]a2", r"(?i)c4", r"(?i)c3"]

_SECONDS_PER_DAY = 24 * 3600


def subject_group(subject_id: str) -> str:
    """Zwraca grupe CAP ("rbd", "n", "nfle", ...) dla identyfikatora pacjenta.

    Rzuca ValueError dla identyfikatorow spoza schematu CAP, zamiast
    zgadywac -- cicho zgadnieta etykieta grupy to wyciek do ewaluacji.
    """
    match = _SUBJECT_ID_RE.match(subject_id.strip())
    if match is None:
        raise ValueError(f"Nieznany identyfikator pacjenta CAP: {subject_id!r}")
    return match.group("group").lower()


def _time_to_seconds(value: str) -> float | None:
    """'22:15:30' albo '22.15.30' -> sekundy od polnocy; None gdy nieczytelne."""
    match = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})[:.](\d{2})(?:[.,](\d+))?\s*", str(value))
    if match is None:
        return None
    h, m, s, frac = match.groups()
    seconds = float(int(h) * 3600 + int(m) * 60 + int(s))
    if frac:
        seconds += float(f"0.{frac}")
    return float(seconds)


def _unwrap_midnight(seconds_of_day: np.ndarray) -> np.ndarray:
    """Zamienia godziny zegarowe na monotoniczna os czasu przez polnoc.

    Hipnogram nocny przechodzi przez 00:00:00, wiec 23:59:30 -> 00:00:00
    to +30 s, a nie -86370 s. Kazdy spadek wiekszy niz pol doby traktujemy
    jako przejscie przez polnoc.
    """
    out = seconds_of_day.astype(float).copy()
    day_offset = 0.0
    for i in range(1, len(out)):
        if seconds_of_day[i] + day_offset < out[i - 1] - _SECONDS_PER_DAY / 2:
            day_offset += _SECONDS_PER_DAY
        out[i] = seconds_of_day[i] + day_offset
    return out


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

    def __init__(
        self,
        data_dir: str | Path,
        target_fs: int = 200,
        epoch_sec: int = 30,
        chin_patterns: list[str] | None = None,
        leg_patterns: list[str] | None = None,
        eeg_patterns: list[str] | None = None,
        preprocess_emg: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.target_fs = target_fs
        self.epoch_sec = epoch_sec
        self.chin_patterns = chin_patterns or CHIN_PATTERNS
        self.leg_patterns = leg_patterns or LEG_PATTERNS
        self.eeg_patterns = eeg_patterns or EEG_PATTERNS
        # True -> EMG (broda, noga) filtrowane emg_bandpass() na CALEJ nocy przed
        # cieciem na epoki (brak artefaktow brzegowych filtra w kazdej epoce).
        # Domyslnie False, zeby nie zmieniac wejscia istniejacych modeli.
        self.preprocess_emg = preprocess_emg

    def parse_remlogic_txt(self, txt_path: Path, recording_start: dt_time | None = None) -> pd.DataFrame:
        """Parsuje plik RemLogic Event Export i zwraca epoki stadiow snu.

        Separator to "jeden lub wiecej tabulatorow" (sep=r"\t+"), NIE
        biale znaki ogolnie -- pole Position bywa wieloczlonowe ("Unknown
        Position") i zawiera spacje, ktora nie jest tabulatorem.

        start_sec:
        - recording_start podany (godzina startu z naglowka EDF): sekundy od
          startu nagrania, liczone z kolumny Time.
        - recording_start=None: sekundy od pierwszej epoki hipnogramu (luki
          w scoringu nadal sa zachowane, bo liczymy z Time, nie z indeksu).
        - brak czytelnej kolumny Time: fallback do i * epoch_sec z ostrzezeniem.
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
        df = df.reset_index(drop=True)

        df["stage_clean"] = df["Sleep Stage"].astype(str).str.strip().map(
            lambda s: self.STAGE_MAP.get(s, "UNKNOWN")
        )
        df["duration_clean"] = pd.to_numeric(df["Duration[s]"], errors="coerce").fillna(self.epoch_sec)
        df["start_sec"] = self._epoch_onsets(df, txt_path, recording_start)

        return df[["stage_clean", "start_sec", "duration_clean"]]

    def _epoch_onsets(self, df: pd.DataFrame, txt_path: Path, recording_start: dt_time | None) -> np.ndarray:
        index_based = np.arange(len(df), dtype=float) * self.epoch_sec
        if len(df) == 0:
            return index_based

        time_col = next((c for c in df.columns if c.startswith("Time")), None)
        clock = (
            np.array([_time_to_seconds(v) for v in df[time_col]], dtype=object)
            if time_col is not None
            else None
        )
        if clock is None or any(v is None for v in clock):
            warnings.warn(
                f"{txt_path.name}: brak czytelnej kolumny Time -- start epok liczony z numeru wiersza "
                "(zakladamy start hipnogramu = start EDF i brak luk)."
            )
            return index_based

        elapsed = _unwrap_midnight(clock.astype(float))
        steps = np.diff(elapsed)
        if len(steps) and not np.allclose(steps, self.epoch_sec):
            n_gaps = int(np.sum(~np.isclose(steps, self.epoch_sec)))
            warnings.warn(
                f"{txt_path.name}: {n_gaps} odstepow miedzy epokami != {self.epoch_sec}s "
                f"(min {steps.min():.0f}s, max {steps.max():.0f}s) -- uzywam czasu z kolumny Time."
            )

        if recording_start is None:
            return elapsed - elapsed[0]

        start_of_day = recording_start.hour * 3600 + recording_start.minute * 60 + recording_start.second
        # Przesuniecie w (-12h, 12h]: hipnogram moze zaczac sie chwile PRZED
        # startem EDF (ujemne start_sec -- takie epoki pomija load_subject).
        half_day = _SECONDS_PER_DAY / 2
        offset = (elapsed[0] - start_of_day + half_day) % _SECONDS_PER_DAY - half_day
        return elapsed - elapsed[0] + offset

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

        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        ch_names = raw.ch_names

        chin_ch = self._find_channel(ch_names, self.chin_patterns)
        leg_ch = self._find_channel(ch_names, self.leg_patterns)
        eeg_ch = self._find_channel(ch_names, self.eeg_patterns)

        if chin_ch is None:
            raise ValueError(f"Pacjent {subject_id} nie posiada kanalu Chin EMG w {ch_names}")

        # Wczytujemy i resamplujemy tylko potrzebne kanaly (nie cale 15+).
        picks = [ch for ch in dict.fromkeys([chin_ch, leg_ch, eeg_ch]) if ch is not None]
        raw.pick(picks)
        raw.load_data(verbose="ERROR")
        if raw.info["sfreq"] != self.target_fs:
            raw.resample(self.target_fs, npad="auto", verbose="ERROR")

        meas_date = raw.info.get("meas_date")
        recording_start = meas_date.time() if meas_date is not None else None
        if recording_start is None:
            warnings.warn(f"{subject_id}: brak godziny startu w naglowku EDF -- zakladam start hipnogramu = start EDF.")

        df_stages = self.parse_remlogic_txt(txt_path, recording_start=recording_start)
        recording_sec = raw.n_times / self.target_fs
        if len(df_stages) and df_stages["start_sec"].iloc[0] >= recording_sec:
            # Godzina w naglowku EDF bywa zanonimizowana/niespojna -- wtedy
            # przesuniecie wychodzi poza nagranie. Nie udajemy, ze wiemy lepiej.
            warnings.warn(
                f"{subject_id}: pierwsza epoka hipnogramu wypada po koncu nagrania "
                f"({df_stages['start_sec'].iloc[0]:.0f}s >= {recording_sec:.0f}s) -- godzina startu EDF "
                "niespojna z plikiem TXT; zakladam start hipnogramu = start EDF."
            )
            df_stages = self.parse_remlogic_txt(txt_path, recording_start=None)
        elif len(df_stages) and df_stages["start_sec"].iloc[0] != 0:
            warnings.warn(
                f"{subject_id}: hipnogram zaczyna sie {df_stages['start_sec'].iloc[0]:+.0f}s wzgledem startu EDF "
                "-- epoki przesuniete zgodnie z kolumna Time."
            )

        is_rbd = subject_group(subject_id) == "rbd"

        epoch_samples = int(self.epoch_sec * self.target_fs)
        epochs = []

        chin_data = raw.get_data(picks=[chin_ch])[0]
        leg_data = raw.get_data(picks=[leg_ch])[0] if leg_ch else None
        if self.preprocess_emg:
            chin_data = emg_bandpass(chin_data, self.target_fs)
            leg_data = emg_bandpass(leg_data, self.target_fs) if leg_data is not None else None
        eeg_data = raw.get_data(picks=[eeg_ch])[0] if eeg_ch else None
        total_samples = len(chin_data)

        for i, row in enumerate(df_stages.itertuples(index=False)):
            stage = row.stage_clean
            if stages_filter and stage not in stages_filter:
                continue

            start_idx = int(round(row.start_sec * self.target_fs))
            end_idx = start_idx + epoch_samples

            if start_idx < 0:
                continue  # epoka zaczeta przed startem nagrania EDF
            if end_idx > total_samples:
                break

            epochs.append(
                EpochData(
                    subject_id=subject_id,
                    epoch_idx=i,
                    stage=stage,
                    start_sec=float(row.start_sec),
                    duration_sec=float(row.duration_clean),
                    emg_chin=chin_data[start_idx:end_idx],
                    emg_leg=leg_data[start_idx:end_idx] if leg_data is not None else None,
                    eeg_central=eeg_data[start_idx:end_idx] if eeg_data is not None else None,
                    sampling_rate=self.target_fs,
                    is_rbd=is_rbd,
                )
            )

        return epochs
