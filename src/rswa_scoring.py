"""
Scoring RSWA (REM Sleep Without Atonia) z sygnału EMG podbródkowego.

WAŻNA KOREKTA ZAŁOŻENIA (2026-08-28): CAP Sleep Database NIE zawiera
żadnych per-epokowych etykiet RSWA/utraty atonii mięśniowej. Adnotacje w
plikach .txt to wyłącznie: (1) makrostruktura R&K (W/S1-S4/REM) i
(2) mikrostruktura CAP (fazy A/B, podtypy A1/A2/A3) -- zjawisko EEG
zupełnie niezwiązane z napięciem mięśniowym. Sprawdzone bezpośrednio na
stronie physionet.org/content/capslpdb/1.0.0/ (sekcja "Annotations").

To oznacza, że pierwotny plan Fazy 3 ("wytrenuj klasyfikator na etykietach
EMG") nie ma w tym zbiorze żadnych etykiet do trenowania na poziomie epoki.
Zamiast tego RSWA liczymy tu regułą kliniczną (uproszczoną z kryteriów
SINBAR/Montreal -- Frauscher i wsp. 2012, Montplaisir i wsp. 2010), a nie
uczonym klasyfikatorem. To akurat lepiej pasuje do natury problemu: RSWA
to wielkość KLINICZNIE ZDEFINIOWANA (podwyższone napięcie/wybuchy EMG
względem linii bazowej NREM), a nie wzorzec do "odkrycia" przez model.

Co TO pozwala zwalidować na CAP: czy rswa_index policzony tą regułą jest
wyższy, na poziomie pacjenta, w grupie "rbd" niż w grupie "n" (zdrowi) --
to legalne porównanie statystyczne, bo etykiety grupowe SĄ prawdziwym
ground truth w tym zbiorze (diagnoza kliniczna, nie coś co sami wymyśliliśmy).

Czego to NIE pozwala zwalidować: precyzji/recall per-epokowego względem
złotego standardu scoringu RSWA -- bo takiej etykiety tu po prostu nie ma
(wymagałaby to oryginalnego przeglądu wideo-PSG, którego nie ma w tym
publicznym zbiorze). Trzeba być precyzyjnym, które z tych dwóch twierdzeń
faktycznie popiera dany wynik -- nie mieszać jednego z drugim w raporcie.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d


@dataclass
class RSWAScoreResult:
    epoch_rms: np.ndarray  # (n_rem_epochs,) RMS EMG podbródkowego per epoka REM
    nrem_baseline_rms: float  # mediana RMS EMG z epok N2/N3 tego pacjenta
    atonia_lost: np.ndarray  # (n_rem_epochs,) bool
    rswa_index: float  # odsetek epok REM oznaczonych jako utrata atonii -- wynik na poziomie pacjenta


def compute_nrem_baseline(nrem_epoch_signals: list[np.ndarray]) -> float:
    """
    Mediana RMS EMG podbródkowego z epok N2/N3 tego pacjenta -- referencyjny
    poziom "spokojnego mięśnia". Liczona per pacjent (nie jeden stały próg
    dla wszystkich), bo surowa amplituda EMG zależy od impedancji elektrod
    i wzmocnienia, które różnią się między 108 nagraniami w tym
    wieloośrodkowym archiwum (ta sama ostrożność co przy nazwach kanałów
    udokumentowana w docs/technical_premise.md).
    """
    if not nrem_epoch_signals:
        raise ValueError("Potrzeba co najmniej jednej epoki NREM (N2/N3) do ustalenia linii bazowej.")
    rms_values = [float(np.sqrt(np.mean(np.square(epoch)))) for epoch in nrem_epoch_signals]
    return float(np.median(rms_values))


def score_rswa(
    rem_epoch_signals: list[np.ndarray],
    nrem_epoch_signals: list[np.ndarray],
    threshold_multiplier: float = 2.0,
) -> RSWAScoreResult:
    """
    threshold_multiplier=2.0 to punkt startowy, NIE zwalidowany próg
    kliniczny -- kryteria SINBAR/Montreal używają konkretnych warunków
    amplitudy ORAZ czasu trwania (rozróżnienie tonic/phasic, minimalny
    czas trwania wybuchu), znacznie bardziej szczegółowych niż pojedynczy
    stosunek RMS. To celowo prosty pierwszy baseline (ta sama dyscyplina
    "klasyczny baseline przed głębokim modelem" co w parkinsons-eeg-
    classifier), do dopracowania dopiero gdy realne nagrania CAP będą
    dostępne do strojenia -- nie gotowy scorer kliniczny.
    """
    baseline = compute_nrem_baseline(nrem_epoch_signals)
    epoch_rms = np.array([np.sqrt(np.mean(np.square(epoch))) for epoch in rem_epoch_signals])
    atonia_lost = epoch_rms > (threshold_multiplier * baseline)
    rswa_index = float(np.mean(atonia_lost)) if len(atonia_lost) > 0 else float("nan")
    return RSWAScoreResult(
        epoch_rms=epoch_rms,
        nrem_baseline_rms=baseline,
        atonia_lost=atonia_lost,
        rswa_index=rswa_index,
    )


# ---------------------------------------------------------------------------
# Dlaczego powyzszy score_rswa dal rswa_index=0.0 dla rbd1 (2026-09-23)
# ---------------------------------------------------------------------------
# To nie musi byc blad danych -- to w duzej mierze wlasnosc samej reguly:
# (1) fizjologicznie napiecie EMG brody w REM jest NIZSZE niz w NREM, wiec
#     prog "2x mediana NREM" jest bardzo wysoki wzgledem atonicznego tla REM;
# (2) aktywnosc fazowa RSWA to wybuchy 0.1-5 s -- RMS z calych 30 s je
#     rozmywa (5 s wybuchu o 3x amplitudzie tla daje RMS epoki ~1.6x tla);
# (3) RMS surowego sygnalu (bez pasma 10-100 Hz) mierzy tez dryf i EKG.
# Ponizej dwie metryki blizsze temu, jak RSWA liczy literatura. Obie sa
# CIAGLE (bez progu klinicznego) -- prog decyzyjny ma byc wybierany w LOSO
# (src/evaluate.py), a nie przepisany z publikacji o innym pasmie/sprzecie.


def mini_epoch_rms(signal: np.ndarray, fs: float, mini_epoch_s: float = 3.0) -> np.ndarray:
    """RMS w kolejnych, rozlacznych mini-epokach (reszta na koncu odrzucana)."""
    n = int(round(mini_epoch_s * fs))
    if n <= 0:
        raise ValueError("mini_epoch_s * fs musi dawac co najmniej 1 probke")
    n_mini = len(signal) // n
    if n_mini == 0:
        return np.empty(0)
    blocks = np.asarray(signal[: n_mini * n], dtype=np.float64).reshape(n_mini, n)
    return np.sqrt(np.mean(np.square(blocks), axis=1))


@dataclass
class MiniEpochRSWAResult:
    mini_rms: np.ndarray  # (n_rem_epochs, n_mini) RMS per 3-s mini-epoka
    background_rms: float  # atoniczne tlo REM tego pacjenta
    active: np.ndarray  # (n_rem_epochs, n_mini) bool
    rswa_mini_index: float  # odsetek mini-epok REM z aktywnoscia ("any", SINBAR-podobne)
    tonic_epoch_fraction: float  # odsetek epok 30 s z aktywnoscia w >=50% mini-epok


def score_rswa_mini_epochs(
    rem_epoch_signals: list[np.ndarray],
    fs: float,
    mini_epoch_s: float = 3.0,
    background_percentile: float = 10.0,
    threshold_multiplier: float = 2.0,
    tonic_min_fraction: float = 0.5,
) -> MiniEpochRSWAResult:
    """RSWA w 3-s mini-epokach wzgledem atonicznego tla REM (inspirowane SINBAR).

    SINBAR (Frauscher i wsp. 2012) liczy aktywnosc w 3-s mini-epokach REM
    wzgledem amplitudy tla (atonii) -- nie wzgledem NREM. Tu tlo to
    `background_percentile` rozkladu RMS mini-epok REM tego pacjenta.
    Ograniczenie: przy niemal ciaglej aktywnosci tonicznej (ciezkie RBD)
    niski percentyl tez rosnie i indeks jest ZANIZONY -- to blad
    konserwatywny, ale realny. Sygnaly powinny byc juz po emg_bandpass().

    To NIE jest wizualny scoring SINBAR (brak kryteriow czasu trwania
    wybuchu 0.1-5 s, brak FDS) -- nie porownywac liczbowo z progami z
    publikacji bez walidacji.
    """
    if not rem_epoch_signals:
        return MiniEpochRSWAResult(np.empty((0, 0)), float("nan"), np.empty((0, 0), dtype=bool), float("nan"), float("nan"))
    mini = np.stack([mini_epoch_rms(sig, fs, mini_epoch_s) for sig in rem_epoch_signals])
    if mini.size == 0:
        raise ValueError(f"Epoki REM krotsze niz jedna mini-epoka ({mini_epoch_s} s przy fs={fs} Hz).")
    background = float(np.percentile(mini, background_percentile))
    if background <= 0:
        raise ValueError("Tlo EMG REM <= 0 -- kanal plaski albo odlaczony elektrodowo.")
    active = mini > threshold_multiplier * background
    per_epoch = active.mean(axis=1)
    return MiniEpochRSWAResult(
        mini_rms=mini,
        background_rms=background,
        active=active,
        rswa_mini_index=float(active.mean()),
        tonic_epoch_fraction=float(np.mean(per_epoch >= tonic_min_fraction)),
    )


def _contiguous_runs(epoch_starts: np.ndarray, epoch_len_s: float) -> list[np.ndarray]:
    """Dzieli indeksy epok na ciagi kolejnych (start co epoch_len_s)."""
    if len(epoch_starts) == 0:
        return []
    breaks = np.where(~np.isclose(np.diff(epoch_starts), epoch_len_s))[0] + 1
    return np.split(np.arange(len(epoch_starts)), breaks)


def rem_atonia_index(
    rem_epoch_signals: list[np.ndarray],
    fs: float,
    epoch_starts_s: np.ndarray | None = None,
    to_microvolts: float = 1e6,
    noise_window_s: int = 60,
) -> float:
    """REM Atonia Index (Ferri i wsp. 2008; korekcja szumu: Ferri i wsp. 2010).

    1) sygnal po emg_bandpass(), wyprostowany, srednia amplituda w 1-s mini-epokach [uV];
    2) korekcja szumu: odjecie minimum z okna ruchomego `noise_window_s` mini-epok
       wycentrowanego na kazdej (liczone w obrebie ciaglych odcinkow REM);
    3) RAI = %(amp <= 1 uV) / (100 - %(1 < amp <= 2 uV)).
    RAI w [0, 1]; 1 = pelna atonia. Zalezy od BEZWZGLEDNEJ amplitudy w uV --
    wiec od wzmocnienia/impedancji w danym laboratorium CAP. Progi z
    literatury (ok. 0.8-0.9) traktowac jako orientacyjne, nie przenosic 1:1.

    `to_microvolts=1e6` zaklada wejscie w woltach (tak zwraca MNE).
    """
    if not rem_epoch_signals:
        return float("nan")
    per_epoch_amp = [
        np.abs(np.asarray(sig, dtype=np.float64) * to_microvolts)[: (len(sig) // int(fs)) * int(fs)]
        .reshape(-1, int(fs))
        .mean(axis=1)
        for sig in rem_epoch_signals
    ]
    epoch_len_s = len(rem_epoch_signals[0]) / fs
    starts = (
        np.asarray(epoch_starts_s, dtype=float)
        if epoch_starts_s is not None
        else np.arange(len(rem_epoch_signals)) * epoch_len_s
    )
    corrected = []
    for run in _contiguous_runs(starts, epoch_len_s):
        amp = np.concatenate([per_epoch_amp[i] for i in run])
        floor = minimum_filter1d(amp, size=noise_window_s + 1, mode="nearest")
        corrected.append(np.clip(amp - floor, 0.0, None))
    amp_all = np.concatenate(corrected)
    p_atonic = np.mean(amp_all <= 1.0)
    p_mid = np.mean((amp_all > 1.0) & (amp_all <= 2.0))
    denom = 1.0 - p_mid
    return float(p_atonic / denom) if denom > 0 else float("nan")
