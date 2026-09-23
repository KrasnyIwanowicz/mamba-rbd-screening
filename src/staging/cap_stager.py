"""
Faza 2: uruchomienie SleepStager z submodulu mamba-eeg-sleep-staging na EEG z CAP.

Dopasowanie wejscia do tego, na czym stager byl trenowany (sprawdzone w
external/sleep_staging/src/data_loader.py, 2026-09-23):
- Sleep-EDF-20, kanal "EEG Fpz-Cz", 100 Hz, epoki 30 s = 3000 probek;
- SUROWY sygnal w woltach: data_loader.py NIE wywoluje bandpass ani zscore
  z preprocessing.py tego repo (te funkcje sa tam martwym kodem). Dlatego
  domyslnie NIE normalizujemy -- podanie z-score do modelu uczonego na
  woltach (~1e-5) to dodatkowe, sztuczne przesuniecie domeny.
- okna seq_len=20 kolejnych epok (tak trenowano).

Rzeczy, ktore sa realnym przesunieciem domeny i trzeba je ZMIERZYC
(scripts/evaluate_stager_on_cap.py), nie zakladac: inna derywacja (CAP nie
ma Fpz-Cz; najblizej C4-A1 / Fp2-F4 / F4-C4), inny sprzet i wzmocnienie,
populacja starsza i chora (RBD) vs zdrowi 25-34 lata w SC Sleep-EDF.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_MODELS_DIR = REPO_ROOT / "external" / "sleep_staging" / "src" / "models"
_ALIAS = "sleep_staging_models"

STAGER_FS = 100
EPOCH_SEC = 30
STAGER_CLASSES = ["W", "N1", "N2", "N3", "REM"]
# Mapowanie etykiet CAPSleepLoader -> indeksy klas stagera; reszta (MOVEMENT,
# UNKNOWN) jest wykluczana z ewaluacji (-1), tak jak w Sleep-EDF.
CAP_STAGE_TO_INDEX = {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
REM_INDEX = 4


def _import_submodule_models():
    """Laduje external/sleep_staging/src/models jako pakiet `sleep_staging_models`.

    Oba repo nazywaja swoj pakiet `src`, wiec zwykly sys.path.insert dalby
    kolizje nazw (nasz src vs ich src). Ladowanie pod aliasem jej unika;
    importy wzgledne w submodule (.cnn_encoder, .mamba_block) dzialaja dalej.
    """
    if _ALIAS in sys.modules:
        return sys.modules[_ALIAS]
    init = SUBMODULE_MODELS_DIR / "__init__.py"
    if not init.exists():
        raise FileNotFoundError(
            f"Brak {init}. Zainicjalizuj submodul: git submodule update --init"
        )
    spec = importlib.util.spec_from_file_location(_ALIAS, init, submodule_search_locations=[str(SUBMODULE_MODELS_DIR)])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ALIAS] = module
    spec.loader.exec_module(module)
    return module


def build_stager(sequence_encoder: str = "mamba") -> torch.nn.Module:
    _import_submodule_models()
    from sleep_staging_models.sleep_stager import SleepStager  # type: ignore[import-not-found]

    return SleepStager(sequence_encoder=sequence_encoder, num_classes=len(STAGER_CLASSES))


def load_stager(checkpoint: str | Path, sequence_encoder: str = "mamba", device: str | torch.device = "cpu") -> torch.nn.Module:
    """Wczytuje checkpoint; brak pliku = glosny blad, nigdy losowe wagi."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Brak checkpointu stagera: {checkpoint}. Checkpointy nie sa w git (.gitignore: *.pt). "
            "Wytrenuj w submodule: cd external/sleep_staging && python src/train.py "
            f"--sequence_encoder {sequence_encoder} (zapisuje results/{sequence_encoder}_best.pt), "
            "albo skopiuj istniejacy plik .pt pod te sciezke."
        )
    model = build_stager(sequence_encoder)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def prepare_eeg_epochs(eeg_epochs: list[np.ndarray] | np.ndarray, fs_in: float, scale_to_std: float | None = None) -> np.ndarray:
    """Epoki EEG z CAP (w woltach, fs_in) -> (n_epochs, 3000) float32 @ 100 Hz.

    Resampling na calej konkatenacji (bez artefaktow brzegowych per epoka).
    scale_to_std: opcjonalnie przeskaluj cala noc do zadanego odchylenia
    standardowego (w woltach) -- eksperyment na adaptacje amplitudy miedzy
    laboratoriami; domyslnie None = surowe wolty, jak w treningu.
    """
    epochs = np.asarray(eeg_epochs, dtype=np.float64)
    if epochs.ndim != 2:
        raise ValueError("Oczekiwano (n_epochs, n_samples)")
    fs_int = int(round(fs_in))
    if abs(fs_in - fs_int) > 1e-6:
        raise ValueError(f"Niecalkowite fs_in={fs_in} nieobslugiwane")
    g = np.gcd(fs_int, STAGER_FS)
    flat = resample_poly(epochs.reshape(-1), STAGER_FS // g, fs_int // g)
    n_out = EPOCH_SEC * STAGER_FS
    flat = flat[: len(epochs) * n_out]
    if len(flat) < len(epochs) * n_out:
        flat = np.pad(flat, (0, len(epochs) * n_out - len(flat)), mode="edge")
    if scale_to_std is not None:
        flat = (flat - flat.mean()) / (flat.std() + 1e-12) * scale_to_std
    return flat.reshape(len(epochs), n_out).astype(np.float32)


@torch.no_grad()
def predict_stages(model: torch.nn.Module, X: np.ndarray, seq_len: int = 20, device: str | torch.device = "cpu", batch_size: int = 16) -> np.ndarray:
    """(n_epochs, 3000) -> etykiety 0-4 per epoka.

    Nienachodzace okna seq_len; ostatnie niepelne okno liczone jako ostatnie
    seq_len epok nocy (uzupelniamy tylko brakujace pozycje). Noc krotsza niz
    seq_len jest przetwarzana jako jedno krotsze okno.
    """
    n = len(X)
    if n == 0:
        return np.empty(0, dtype=int)
    if n <= seq_len:
        windows = [(0, n)]
    else:
        windows = [(s, s + seq_len) for s in range(0, n - seq_len + 1, seq_len)]
        if windows[-1][1] < n:
            windows.append((n - seq_len, n))

    preds = np.full(n, -1, dtype=int)
    for i in range(0, len(windows), batch_size):
        chunk = windows[i:i + batch_size]
        batch = torch.from_numpy(np.stack([X[s:e] for s, e in chunk])).to(device)
        labels = model(batch).argmax(dim=-1).cpu().numpy()
        for (s, e), lab in zip(chunk, labels):
            fill = preds[s:e] == -1
            preds[s:e][fill] = lab[fill]
    return preds


def cap_stages_to_indices(stages: list[str]) -> np.ndarray:
    return np.array([CAP_STAGE_TO_INDEX.get(s, -1) for s in stages], dtype=int)


def rem_windows(stage_indices: np.ndarray, epoch_starts_s: np.ndarray, epoch_sec: float = EPOCH_SEC) -> list[tuple[float, float]]:
    """Ciagle odcinki REM jako (start_s, end_s) -- wyjscie Fazy 2 dla detektora RSWA."""
    windows: list[tuple[float, float]] = []
    for idx in np.flatnonzero(stage_indices == REM_INDEX):
        start = float(epoch_starts_s[idx])
        if windows and np.isclose(windows[-1][1], start):
            windows[-1] = (windows[-1][0], start + epoch_sec)
        else:
            windows.append((start, start + epoch_sec))
    return windows
