"""Faza 1: audyt kanalow EDF + kompletnosci/wyrownania hipnogramow CAP -> reports/cap_channel_audit.csv."""
from pathlib import Path
import re
import sys
import warnings

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.cap_loader import CAPSleepLoader

# Mapowanie reguł dla kluczowych kanałów
CHANNEL_PATTERNS = {
    "EMG_CHIN": [
        r"(?i)emg.*chin", r"(?i)chin.*emg", r"(?i)chin\d*[-_]chin\d*", 
        r"(?i)submental", r"(?i)emg1[-_]emg2"
    ],
    "EMG_LEG": [
        r"(?i)tibial", r"(?i)leg", r"(?i)dx\d*[-_]dx\d*", r"(?i)sx\d*[-_]sx\d*", 
        r"(?i)dx\d*[-_]sn\d*", r"(?i)rat[-_]lat", r"(?i)emg.*leg", r"(?i)dx", r"(?i)sx"
    ],
    "EEG_CENTRAL": [
        r"(?i)c4[-_]a1", r"(?i)c3[-_]a2", r"(?i)c4[-_]m1", r"(?i)c3[-_]m2", 
        r"(?i)c4[-_]p4", r"(?i)c4", r"(?i)c3"
    ],
    "EEG_OCCIPITAL": [
        r"(?i)o2[-_]a1", r"(?i)o1[-_]a2", r"(?i)p4[-_]o2", r"(?i)o2", r"(?i)o1"
    ],
    "EOG": [
        r"(?i)roc[-_]loc", r"(?i)eog", r"(?i)e1[-_]m2", r"(?i)e2[-_]m2"
    ]
}

def find_matching_channel(ch_names: list[str], patterns: list[str]) -> str | None:
    for pat in patterns:
        for ch in ch_names:
            if re.search(pat, ch):
                return ch
    return None


def audit_hypnogram(txt_file: Path, recording_start, recording_sec: float) -> dict:
    """Kompletnosc i wyrownanie hipnogramu wzgledem EDF (bez wczytywania sygnalu)."""
    empty = {"n_scored_epochs": 0, "n_rem_epochs": 0, "n_nrem_epochs": 0,
             "hyp_offset_s": np.nan, "hyp_gaps": np.nan, "hyp_end_past_edf_s": np.nan, "hyp_warning": ""}
    if not txt_file.exists():
        return {**empty, "hyp_warning": "brak pliku TXT"}
    loader = CAPSleepLoader(txt_file.parent)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            df = loader.parse_remlogic_txt(txt_file, recording_start=recording_start)
        except ValueError as e:
            return {**empty, "hyp_warning": str(e)}
    starts = df["start_sec"].to_numpy()
    steps = np.diff(starts)
    return {
        "n_scored_epochs": len(df),
        "n_rem_epochs": int((df["stage_clean"] == "REM").sum()),
        "n_nrem_epochs": int(df["stage_clean"].isin(["N2", "N3"]).sum()),
        "hyp_offset_s": float(starts[0]) if len(starts) else np.nan,
        "hyp_gaps": int(np.sum(~np.isclose(steps, loader.epoch_sec))) if len(steps) else 0,
        # >0: hipnogram wychodzi poza koniec EDF -> ostatnie epoki beda odciete
        "hyp_end_past_edf_s": float(starts[-1] + loader.epoch_sec - recording_sec) if len(starts) else np.nan,
        "hyp_warning": " | ".join(str(w.message) for w in caught),
    }


def audit_cap_database(data_dir: str | Path, output_csv: str = "reports/cap_channel_audit.csv"):
    data_path = Path(data_dir)
    edf_files = sorted(list(data_path.glob("*.edf")))
    
    if not edf_files:
        print(f"[!] Nie znaleziono plików .edf w {data_path}")
        return

    records = []
    print(f"[*] Rozpoczynam audyt {len(edf_files)} plików EDF...")

    for edf_file in tqdm(edf_files):
        subject_id = edf_file.stem
        txt_file = edf_file.with_suffix(".txt")
        has_txt = txt_file.exists()

        try:
            # Wczytujemy tylko nagłówek EDF bez ładowania sygnałów do RAM
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose="ERROR")
            ch_names = raw.ch_names
            sfreq = raw.info["sfreq"]
            duration_hours = raw.n_times / (sfreq * 3600)

            meas_date = raw.info.get("meas_date")
            hyp = audit_hypnogram(txt_file, meas_date.time() if meas_date else None, raw.n_times / sfreq)

            detected = {
                cat: find_matching_channel(ch_names, patterns)
                for cat, patterns in CHANNEL_PATTERNS.items()
            }

            records.append({
                "subject_id": subject_id,
                "cohort": re.sub(r"\d+", "", subject_id).upper(),
                "duration_h": round(duration_hours, 2),
                "base_sfreq": sfreq,
                "total_channels": len(ch_names),
                "has_staging_txt": has_txt,
                "chin_emg_ch": detected["EMG_CHIN"],
                "leg_emg_ch": detected["EMG_LEG"],
                "eeg_central_ch": detected["EEG_CENTRAL"],
                "eeg_occipital_ch": detected["EEG_OCCIPITAL"],
                "eog_ch": detected["EOG"],
                "edf_start": meas_date.strftime("%H:%M:%S") if meas_date else "",
                **hyp,
                "all_channels": "; ".join(ch_names)
            })

        except Exception as e:
            print(f"[!] Błąd przy przetwarzaniu {edf_file.name}: {e}")

    df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print("\n" + "="*50)
    print("AUDYT ZAKOŃCZONY — PODSUMOWANIE:")
    print("="*50)
    print(f"Liczba nagrań: {len(df)}")
    print(f"Dostępność EMG brody (Chin EMG): {df['chin_emg_ch'].notna().sum()} / {len(df)}")
    print(f"Dostępność EMG kończyn (Leg EMG): {df['leg_emg_ch'].notna().sum()} / {len(df)}")
    print(f"Dostępność EEG centralnego (C3/C4): {df['eeg_central_ch'].notna().sum()} / {len(df)}")
    print(f"Dostępność plików TXT (stadia): {df['has_staging_txt'].sum()} / {len(df)}")
    print(f"Hipnogram przesunięty względem startu EDF: {(df['hyp_offset_s'].fillna(0) != 0).sum()} / {len(df)}")
    print(f"Hipnogram z lukami w scoringu: {(df['hyp_gaps'].fillna(0) > 0).sum()} / {len(df)}")
    print(f"Hipnogram dłuższy niż EDF: {(df['hyp_end_past_edf_s'].fillna(0) > 0).sum()} / {len(df)}")
    usable = df['chin_emg_ch'].notna() & (df['n_rem_epochs'] > 0) & (df['n_nrem_epochs'] > 0)
    print(f"Użyteczne do RSWA (chin EMG + REM + N2/N3): {usable.sum()} / {len(df)}")
    print(f"\nSzczegółowy raport zapisano do: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/capslpdb")
    parser.add_argument("--output", default="reports/cap_channel_audit.csv")
    args = parser.parse_args()
    audit_cap_database(args.data_dir, args.output)
