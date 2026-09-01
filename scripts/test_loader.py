import sys
from pathlib import Path

# Dodanie głównego katalogu projektu do sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cap_loader import CAPSleepLoader


def main():
    loader = CAPSleepLoader(data_dir="data/raw/capslpdb", target_fs=200, epoch_sec=30)
    
    print("[*] Testowanie loadera na pacjencie rbd1...")
    
    # 1. Wszystkie epoki
    all_epochs = loader.load_subject("rbd1", stages_filter=None)
    print(f"[+] Załadowano łącznie {len(all_epochs)} epok (30s).")

    # Rozkład stadiów
    stages_count = {}
    for ep in all_epochs:
        stages_count[ep.stage] = stages_count.get(ep.stage, 0) + 1
    print(f"[*] Rozkład stadiów w rbd1: {stages_count}")

    # 2. Tylko epoki REM
    rem_epochs = [ep for ep in all_epochs if ep.stage == "REM"]
    print(f"[+] Liczba epok REM: {len(rem_epochs)} ({len(rem_epochs) * 30 / 60:.1f} minut fazy REM)")

    if rem_epochs:
        first_rem = rem_epochs[0]
        print("\n--- Próbka pierwszej epoki REM ---")
        print(f"Subject: {first_rem.subject_id} (RBD={first_rem.is_rbd})")
        print(f"Chin EMG kształt sygnału: {first_rem.emg_chin.shape} (powinno być 6000 próbek dla 30s @ 200Hz)")
        if first_rem.emg_leg is not None:
            print(f"Leg EMG kształt sygnału:  {first_rem.emg_leg.shape}")
        if first_rem.eeg_central is not None:
            print(f"EEG Central kształt sygn: {first_rem.eeg_central.shape}")


if __name__ == "__main__":
    main()