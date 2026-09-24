"""Wizualna kontrola: co rswa_mini_index uznaje za "aktywnosc" w REM danego pacjenta.

Dla kazdej wybranej epoki REM (najbardziej "aktywne" + losowe, dla porownania):
- gora: EMG brody po emg_bandpass [uV], zacienione 3-s mini-epoki oznaczone jako
  aktywne, linia progu (2x atoniczne tlo REM);
- dol: EKG z wykrytymi zalamkami R.
Jesli "wybuchy" EMG pokrywaja sie z zalamkami R -- to przesluch EKG, nie miesien.

  python scripts/plot_rem_epochs.py rbd2 n1
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.artifacts import detect_r_peaks, ecg_contamination_ratio
from src.data.cap_loader import CAPSleepLoader, EpochData
from src.rswa_scoring import MiniEpochRSWAResult, score_rswa_mini_epochs

MINI_EPOCH_S = 3.0
THRESHOLD_MULTIPLIER = 2.0


def choose_epochs(active: np.ndarray, n_top: int, n_random: int, seed: int = 0) -> list[int]:
    """Indeksy n_top najbardziej aktywnych epok + n_random losowych sposrod pozostalych."""
    order = np.argsort(-active.mean(axis=1), kind="stable")
    top = [int(i) for i in order[:n_top]]
    rest = [int(i) for i in order[n_top:]]
    rng = np.random.default_rng(seed)
    extra = rng.choice(rest, size=min(n_random, len(rest)), replace=False).tolist() if rest else []
    return top + sorted(int(i) for i in extra)


def plot_subject(subject_id: str, rem: list[EpochData], mini: MiniEpochRSWAResult, picks: list[int], out: Path) -> Path:
    fs = float(rem[0].sampling_rate)
    rows = 2 if all(e.ecg is not None for e in rem) else 1
    fig, axes = plt.subplots(len(picks) * rows, 1, figsize=(14, 2.2 * rows * len(picks)), squeeze=False)
    threshold_uv = THRESHOLD_MULTIPLIER * mini.background_rms * 1e6
    for k, idx in enumerate(picks):
        ep = rem[idx]
        t = np.arange(len(ep.emg_chin)) / fs
        ax = axes[k * rows, 0]
        ax.plot(t, ep.emg_chin * 1e6, lw=0.4, color="black")
        n_mini = mini.active.shape[1]
        for j in range(n_mini):
            if mini.active[idx, j]:
                ax.axvspan(j * MINI_EPOCH_S, (j + 1) * MINI_EPOCH_S, color="tab:red", alpha=0.15)
        for sign in (1, -1):
            ax.axhline(sign * threshold_uv, color="tab:red", lw=0.6, ls="--")
        ax.set_ylabel("EMG [uV]")
        ax.set_title(
            f"{subject_id}  REM epoka #{ep.epoch_idx}  t={ep.start_sec / 3600:.2f} h od startu EDF  "
            f"aktywne mini-epoki: {mini.active[idx].mean():.0%}",
            fontsize=9,
        )
        if rows == 2 and ep.ecg is not None:
            ecg_ax = axes[k * rows + 1, 0]
            peaks = detect_r_peaks(ep.ecg, fs)
            ecg_ax.plot(t, ep.ecg * 1e6, lw=0.4, color="tab:blue")
            ecg_ax.plot(peaks / fs, ep.ecg[peaks] * 1e6, "v", color="tab:orange", ms=3)
            for p in peaks:
                ax.axvline(p / fs, color="tab:orange", lw=0.3, alpha=0.6)
            ecg_ax.set_ylabel("EKG [uV]")
    axes[-1, 0].set_xlabel("czas w epoce [s]")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subjects", nargs="+", help="np. rbd2 n1")
    parser.add_argument("--data-dir", default="data/raw/capslpdb")
    parser.add_argument("--n-top", type=int, default=3, help="ile najbardziej aktywnych epok REM")
    parser.add_argument("--n-random", type=int, default=2, help="ile losowych epok REM dla porownania")
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args()

    loader = CAPSleepLoader(args.data_dir, preprocess_emg=True)
    status = 0
    for sid in args.subjects:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rem = loader.load_subject(sid, stages_filter=["REM"])
        except (FileNotFoundError, ValueError) as e:
            print(f"[-] {sid}: {e}")
            status = 1
            continue
        if not rem:
            print(f"[-] {sid}: brak epok REM")
            status = 1
            continue
        fs = float(rem[0].sampling_rate)
        mini = score_rswa_mini_epochs([e.emg_chin for e in rem], fs)
        picks = choose_epochs(mini.active, args.n_top, args.n_random)
        out = plot_subject(sid, rem, mini, picks, Path(args.out_dir) / f"{sid}_rem_epochs.png")
        ecg = [e.ecg for e in rem if e.ecg is not None]
        ratio = ecg_contamination_ratio([e.emg_chin for e in rem], ecg, fs) if len(ecg) == len(rem) else float("nan")
        print(f"{sid}: rswa_mini_index={mini.rswa_mini_index:.3f}  ECG_ratio={ratio:.2f}  -> {out}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
