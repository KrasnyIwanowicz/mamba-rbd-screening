"""Inspect CAP channel selection and the raw EMG separation for one subject.

This is a diagnostic aid, not an RSWA scorer.  In particular, an EDF label
such as ``EMG1-EMG2`` is an acquisition-system alias: it cannot by itself
prove that the electrodes were placed on the chin.  Confirm that mapping in
the recording documentation before interpreting a score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.cap_loader import CAPSleepLoader
from src.rswa_scoring import compute_nrem_baseline


NREM_STAGES = {"N2", "N3"}


def rms(signal: np.ndarray) -> float:
    """Return RMS of one epoch as a Python float."""
    return float(np.sqrt(np.mean(np.square(signal))))


def describe(values: list[float]) -> str:
    """Format robust descriptive statistics, including the empty case."""
    if not values:
        return "n=0"
    array = np.asarray(values)
    return (
        f"n={len(array)}  min={array.min():.3e}  median={np.median(array):.3e}  "
        f"mean={array.mean():.3e}  max={array.max():.3e}  std={array.std():.3e}"
    )


def diagnose(loader: CAPSleepLoader, subject_id: str) -> int:
    """Print channel and raw-RMS diagnostics; return a shell exit status."""
    edf_path = loader.data_dir / f"{subject_id}.edf"
    if not edf_path.exists():
        print(f"Brak pliku EDF: {edf_path}", file=sys.stderr)
        return 2

    # Loading only the header here keeps the channel listing independent of
    # the loader's resampling and epoch extraction.
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    ch_names = raw.ch_names
    chin_ch = loader._find_channel(ch_names, [r"(?i)chin", r"(?i)submental", r"(?i)emg1[-_]emg2"])
    leg_ch = loader._find_channel(ch_names, [r"(?i)dx\d*[-_]dx\d*", r"(?i)dx", r"(?i)tibial"])
    eeg_ch = loader._find_channel(ch_names, [r"(?i)c4[-_]a1", r"(?i)c3[-_]a2", r"(?i)c4", r"(?i)c3"])

    print(f"=== {subject_id} ===\n\n--- Wszystkie kanaly w pliku EDF ---")
    for index, channel in enumerate(ch_names):
        print(f"  [{index}] {channel}")
    print(f"\nCzestotliwosc probkowania w pliku: {raw.info['sfreq']} Hz")
    print("\n--- Wybrane kanaly przez _find_channel ---")
    print(f"  chin (EMG brody):  {chin_ch!r}")
    print(f"  leg  (EMG nogi):   {leg_ch!r}")
    print(f"  eeg  (centralne):  {eeg_ch!r}")
    print("\nUWAGA: nazwa EDF nie potwierdza lokalizacji elektrody; zweryfikuj ")
    print("mapowanie EMG1-EMG2 w dokumentacji danego nagrania/laboratorium.")

    print("\n--- Wczytuje wszystkie epoki (moze chwile potrwac) ---")
    epochs = loader.load_subject(subject_id)
    stage_counts: dict[str, int] = {}
    for epoch in epochs:
        stage_counts[epoch.stage] = stage_counts.get(epoch.stage, 0) + 1
    print(f"Wczytano {len(epochs)} epok lacznie.")
    print(f"Rozklad stadiow: {stage_counts}")

    rem_rms = [rms(epoch.emg_chin) for epoch in epochs if epoch.stage == "REM"]
    nrem_signals = [epoch.emg_chin for epoch in epochs if epoch.stage in NREM_STAGES]
    nrem_rms = [rms(signal) for signal in nrem_signals]
    print("\n--- RMS EMG per grupa epok (surowe, przed progowaniem) ---")
    print(f"  REM : {describe(rem_rms)}")
    print(f"  NREM (N2+N3): {describe(nrem_rms)}")
    if not rem_rms or not nrem_signals:
        print("\nNie da sie porownac REM z NREM: brakuje wymaganych epok.")
        return 1

    baseline = compute_nrem_baseline(nrem_signals)
    ratio = max(rem_rms) / baseline
    print(f"\nNajwyzszy RMS w REM / mediana RMS w NREM = {ratio:.2f}x")
    print("To jest diagnostyka surowego RMS, a nie walidacja klinicznego RSWA.")
    print("Epoki REM powyzej progu, dla roznych mnoznikow:")
    for multiplier in (1.5, 2.0, 3.0):
        count = sum(value > multiplier * baseline for value in rem_rms)
        suffix = "  <- domyslny threshold_multiplier" if multiplier == 2.0 else ""
        print(f"  > {multiplier:.1f}x mediana NREM: {count} / {len(rem_rms)}{suffix}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_id", help="np. rbd1")
    parser.add_argument("--data-dir", default="data/raw/capslpdb", help="Folder z plikami CAP EDF/TXT")
    args = parser.parse_args()
    return diagnose(CAPSleepLoader(args.data_dir), args.subject_id)


if __name__ == "__main__":
    raise SystemExit(main())
