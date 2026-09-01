# mamba-rbd-screening

**Can a dry-electrode EEG + IMU wearable flag REM Sleep Without Atonia (RSWA) — the prodromal signature of Parkinson's/Lewy-body disease — using a sequence model that already exists, or does the EMG→IMU substitution break the signal entirely?**

A screening pipeline that chains two already-validated components — [`mamba-eeg-sleep-staging`](https://github.com/KrasnyIwanowicz/mamba-eeg-sleep-staging) (REM/N1/N2/N3/Wake staging) and a new RSWA/atonia-loss detector — into a full-night "REM Behavior Disorder risk score", benchmarked against a submental-EMG ground truth on public polysomnography data.

![Status](https://img.shields.io/badge/status-phase%200%20%E2%80%94%20scaffold-lightgrey)
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

- **[CAP Sleep Database](https://physionet.org/content/capslpdb/1.0.0/)** (PhysioNet, open access) — full-night PSG including **submental EMG**, with a dedicated `rbd` patient group (n≈16) plus healthy controls and other pathology groups. This is the primary dataset: it's the only freely-available PSG set with both EEG and chin EMG for a clinically-labeled RBD cohort. **To verify before Phase 2**: exact channel montage per subject (CAP is a heterogeneous multi-lab archive, montages are not fully standardized — check per-recording, don't assume).
- **[Sleep-EDF-20](https://physionet.org/content/sleep-edfx/)** — already used in `mamba-eeg-sleep-staging`; healthy-control staging data, reused here only for the sleep-stager component, not for RSWA labels (no EMG).

## Repo structure

```
mamba-rbd-screening/
├── external/
│   └── sleep_staging/           # git submodule → mamba-eeg-sleep-staging (not duplicated)
├── data/                        # download scripts only — no raw data committed
│   └── README.md
├── src/
│   ├── data_loader.py           # CAP Sleep Database EDF/annotation parsing
│   ├── preprocessing.py         # filtering, EMG RMS envelope, REM-epoch extraction
│   ├── dataset.py               # per-subject REM-epoch Dataset, group-aware split
│   ├── models/
│   │   └── rswa_detector.py     # EMG-trained atonia-loss classifier (Phase 3)
│   ├── pipeline.py              # end-to-end: EEG night -> stage -> RSWA -> risk score
│   ├── evaluate.py              # subject-level sensitivity/specificity/AUC vs EMG ground truth
│   └── explainability.py        # Phase 6
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
git submodule add https://github.com/KrasnyIwanowicz/mamba-eeg-sleep-staging.git external/sleep_staging
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
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

## Status

🔲 Phase 0 (this scaffold) — repo structure, honest premise doc, config, stubs.
See [ROADMAP.md](ROADMAP.md) for the full phase plan.

## License

MIT
