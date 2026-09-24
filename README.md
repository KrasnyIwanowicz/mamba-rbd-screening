# mamba-rbd-screening

**Can a dry-electrode EEG + IMU wearable flag REM Sleep Without Atonia (RSWA) — the prodromal signature of Parkinson's/Lewy-body disease — using a sequence model that already exists, or does the EMG→IMU substitution break the signal entirely?**

A screening pipeline that chains two already-validated components — [`mamba-eeg-sleep-staging`](https://github.com/KrasnyIwanowicz/mamba-eeg-sleep-staging) (REM/N1/N2/N3/Wake staging) and a new RSWA/atonia-loss detector — into a full-night "REM Behavior Disorder risk score", benchmarked against a submental-EMG ground truth on public polysomnography data.

![Status](https://img.shields.io/badge/status-phases%201%E2%80%933%20code%20ready%2C%20awaiting%20CAP%20run-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Framing, stated up front

This is a **research / engineering portfolio project**, not a diagnostic device and not yet a company. It produces a risk score for further clinical evaluation, never a diagnosis. Two things it explicitly does **not** claim, in contrast to an earlier concept draft:

1. **The dt-aware Mamba discretization is a hypothesis to test in this domain, not a proven advantage.** In [`mamba-plasticc-transients`](https://github.com/KrasnyIwanowicz/mamba-plasticc-transients), the dt-aware variant *lost* to plain Mamba by ~0.01–0.02 macro-F1 after fixing its instability. Whether irregular artifact/movement gaps in overnight EEG behave differently than PLAsTiCC's observational cadence is an open, testable question (see Phase 4 / `docs/technical_premise.md`), not a settled commercial edge.
2. **The training ground truth (chin EMG) is not the deployment sensor (IMU).** Clinical RSWA scoring uses submental EMG. The target wearable has no EMG channel — only EEG + accelerometer. The gap between "detect atonia loss from EMG" and "detect atonia loss from wrist/head acceleration" is a real, unsolved validation problem, addressed explicitly in Phase 2/4, not assumed away.

## Why this project

RBD (REM Sleep Behavior Disorder) precedes motor symptoms of Parkinson's disease and Lewy body dementia by years to decades, with conversion rates reported above 80% within ~14 years of RBD onset — making it one of the earliest available biomarkers of synucleinopathy. Diagnosis today requires overnight video-PSG in a sleep lab: expensive, low-throughput, and inaccessible for population-level screening. A cheap, dry-electrode home EEG that flags RSWA risk could shift the entry point of screening from "already has motor symptoms" to "years before symptoms" — if the sensor-substitution and modeling questions below actually check out.

## Architecture (planned)

```
Full-night EEG (+ IMU) recording
        │
        ▼
┌───────────────────────────┐
│ Stage 1: Sleep stager      │  reused from mamba-eeg-sleep-staging (git submodule)
│ (CNN epoch encoder +       │  → per-30s-epoch stage: Wake/N1/N2/N3/REM
│  Mamba sequence head)      │
└───────────────────────────┘
        │  REM epochs only
        ▼
┌───────────────────────────┐
│ Stage 2: RSWA detector     │  NEW — this repo's contribution
│ (trained on EMG ground     │  → per-REM-epoch: atonia maintained / lost
│  truth, evaluated for      │
│  transfer to IMU proxy)    │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Stage 3: Night-level score │  aggregate REM-epoch RSWA fraction → RBD risk score
└───────────────────────────┘
```

## Datasets

- **[CAP Sleep Database](https://physionet.org/content/capslpdb/1.0.0/)** (PhysioNet, open access) — full-night PSG including **submental EMG**, with a dedicated `rbd` patient group (n=22) plus 16 healthy controls (`n1`–`n16`) and other pathology groups. This is the primary dataset: it's the only freely-available PSG set with both EEG and chin EMG for a clinically-labeled RBD cohort. **To verify before Phase 2**: exact channel montage per subject (CAP is a heterogeneous multi-lab archive, montages are not fully standardized — check per-recording, don't assume).
- **[Sleep-EDF-20](https://physionet.org/content/sleep-edfx/)** — already used in `mamba-eeg-sleep-staging`; healthy-control staging data, reused here only for the sleep-stager component, not for RSWA labels (no EMG).

## Repo structure

```
mamba-rbd-screening/
├── external/
│   └── sleep_staging/           # git submodule → mamba-eeg-sleep-staging (not duplicated)
├── data/                        # download scripts only — no raw data committed
│   └── README.md
├── src/
│   ├── data/
│   │   ├── cap_loader.py        # CAP EDF + RemLogic TXT, time-aligned epochs, subject groups
│   │   └── rbd_dataset.py       # REM-epoch EMG Dataset for the Mamba classifier
│   ├── preprocessing.py         # EMG band 10-100 Hz + 50 Hz notch, RMS envelope
│   ├── rswa_scoring.py          # rule-based RSWA: 30-s rule, 3-s mini-epochs, REM Atonia Index
│   ├── evaluate.py              # subject-level AUC + LOSO-threshold sens/spec (rbd vs n)
│   ├── staging/cap_stager.py    # Phase 2: submodule SleepStager on CAP EEG
│   ├── models/                  # mamba_rbd.py (Bi-Mamba EMG classifier), rswa_detector.py
│   ├── training/train_rbd.py    # LOSO x seeds CV, subject-level metrics
│   └── pipeline/rbd_pipeline.py # night -> REM -> RSWA metrics -> risk score (not a diagnosis)
├── scripts/                     # download, audit, RSWA scoring/evaluation, stager transfer, diagnostics
├── configs/
│   └── config.yaml
├── tests/                       # synthetic-data tests, no dataset download needed for CI
├── docs/
│   ├── technical_premise.md     # honest status of the dt-aware hypothesis + EMG→IMU gap
│   └── regulatory_notes.md      # what "real startup" actually requires beyond code
├── requirements.txt
├── ROADMAP.md
└── README.md
```

## Setup

```bash
git clone https://github.com/KrasnyIwanowicz/mamba-rbd-screening.git
cd mamba-rbd-screening
git submodule update --init          # external/sleep_staging (already registered in .gitmodules)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                  # synthetic-data tests, no download needed
```

## Workflow on the real CAP data

```bash
python scripts/download_cap.py                               # rbd1-22 + n1-16, .edf + .txt
python scripts/audit_cap_channels.py                         # Phase 1: channels + hypnogram alignment
python scripts/run_rswa_pipeline.py                          # Phase 3: RSWA metrics per subject
python scripts/evaluate_rswa.py                              # rbd vs n: AUC, LOSO sens/spec, negative controls
python scripts/plot_rem_epochs.py rbd2 n1                    # look at what counts as "activity" (EMG + ECG)
python scripts/evaluate_stager_on_cap.py --checkpoint external/sleep_staging/results/mamba_best.pt   # Phase 2
python src/training/train_rbd.py --seeds 0 1 2               # Mamba classifier, LOSO x seeds
```

## Diagnostyka kanału EMG

Przed interpretacją wyniku RSWA sprawdź, który kanał EDF został wybrany jako
EMG brody i czy surowy RMS w REM w ogóle rozdziela się od NREM:

```bash
python scripts/diagnose_subject.py rbd1 --data-dir data/raw/capslpdb
```

Skrypt wypisuje pełną listę kanałów, wybór loadera i statystyki RMS przed
progowaniem. Alias w rodzaju `EMG1-EMG2` **nie potwierdza sam w sobie**, że
jest to EMG podbródkowe; trzeba zweryfikować montaż w dokumentacji konkretnego
zapisu. Brak epok REM ponad nawet łagodniejszym progiem oznacza, że obecny
baseline RMS nie dostarcza sygnału do rozdzielania tej nocy — nie że należy
automatycznie obniżyć próg albo że kanał jest potwierdzony.

### Brakujący plik hipnogramu

Pobierz lub sprawdź pojedynczy plik bez pobierania całej bazy:

```bash
python scripts/download_cap.py --subjects n1 --extensions .txt --data-dir data/raw/capslpdb
```

Skrypt kończy się błędem i wypisuje brakujący plik, jeśli źródło nie udostępnia
go pod oczekiwaną nazwą. Rekord bez odpowiadającego `.txt` nie może zostać
użyty do analizy REM/RSWA.

## Status

- ✅ Phase 0: scaffold, premise doc, submodule.
- 🟡 Phases 1–3: code is done and tested on synthetic data (loader alignment
  fix, audit, RSWA metrics, subject-level LOSO evaluation, stager transfer, CV
  training). **No results on real CAP data yet**: the audit has only covered
  rbd1, and the legacy rule gave `rswa_index=0.0` there. Why that is expected
  of that rule is explained in `docs/technical_premise.md`.
- ✅ Phase 8 (partial): CI with synthetic tests + mypy.

See [ROADMAP.md](ROADMAP.md) for the full phase plan.

## License

MIT
