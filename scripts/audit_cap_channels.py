from pathlib import Path
import re
import mne
import pandas as pd
from tqdm import tqdm

# Mapowanie reguł dla kluczowych kanałów
CHANNEL_PATTERNS = {
    "EMG_CHIN": [r"(?i)emg.*chin", r"(?i)chin.*emg", r"(?i)chin\d*[-_]chin\d*", r"(?i)submental", r"(?i)emg1[-_]emg2"],
    "EMG_LEG": [r"(?i)tibial", r"(?i)leg", r"(?i)dx\d*[-_]sn\d*", r"(?i)rat[-_]lat", r"(?i)emg.*leg"],
    "EEG_CENTRAL": [r"(?i)c4[-_]a1", r"(?i)c3[-_]a2", r"(?i)c4[-_]m1", r"(?i)c3[-_]m2", r"(?i)c4", r"(?i)c3"],
    "EEG_OCCIPITAL": [r"(?i)o2[-_]a1", r"(?i)o1[-_]a2", r"(?i)o2[-_]m1", r"(?i)o1[-_]m2", r"(?i)o2", r"(?i)o1"],
    "EOG": [r"(?i)eog", r"(?i)roc[-_]loc", r"(?i)e1[-_]m2", r"(?i)e2[-_]m2"]
}


def find_matching_channel(ch_names: list[str], patterns: list[str]) -> str | None:
    for pat in patterns:
        for ch in ch_names:
            if re.search(pat, ch):
                return ch
    return None


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
    print(f"\nSzczegółowy raport zapisano do: {output_csv}")


if __name__ == "__main__":
    audit_cap_database("data/raw/capslpdb")