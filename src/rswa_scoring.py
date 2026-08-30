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
