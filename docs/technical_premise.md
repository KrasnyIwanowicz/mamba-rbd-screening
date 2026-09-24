# Technical premise: what's proven, what's hypothesis, what's currently wrong

Written before Phase 1 starts, so scope creep can be checked against it later.
Update this file, don't quietly let the README drift from it.

## Proven (from my existing repos)

- **Sleep staging works reasonably well from single-channel EEG.**
  `mamba-eeg-sleep-staging`: CNN+Mamba reaches 81.2% accuracy / 0.777 macro-F1
  on Sleep-EDF-20, subject-independent. REM is one of the harder classes but
  the overall pipeline (epoch encoder + sequence head) is validated.
- **Resting-state EEG carries *some* PD-related signal, but it's weak.**
  `parkinsons-eeg-classifier`: LSTM gets 0.667±0.074 accuracy / 0.686±0.075
  AUC, LOSO-CV, n=31. Real, above-chance, but the dataset's own curators
  explicitly warn against treating this as diagnostically meaningful at this
  sample size - that warning applies with equal force here.

## Corrected 2026-08-28, after real CAP data was downloaded and audited

- **CAP Sleep Database has NO per-epoch RSWA/atonia-loss ground truth.**
  Verified directly against physionet.org/content/capslpdb/1.0.0/: the only
  annotations are R&K sleep macrostructure (W/S1-S4/REM) and CAP
  microstructure (phase A/B, subtypes A1/A2/A3) - an EEG phenomenon
  unrelated to muscle tone. Phase 3's original framing ("train a
  classifier on EMG ground truth") has no epoch-level labels to train
  against in this dataset. `src/rswa_scoring.py` replaces the supervised-
  classifier framing with a rule-based scorer (chin EMG RMS vs. each
  subject's own NREM baseline), which is arguably more appropriate anyway
  since RSWA is a clinically *defined* quantity, not a pattern to discover.
  What IS still valid to check with CAP: whether this rule's rswa_index is
  higher, at the subject level, in the "rbd" group than the "n" group -
  because subject-level group labels are real ground truth here. Per-epoch
  precision/recall against a gold standard is NOT something this dataset
  can validate.
- **The `.txt` annotation parser had a silent, universal bug** (see
  PATCH_NOTES.md) that made `parse_txt_annotations` return zero rows for
  every subject, regardless of the Position field's format. The current
  loader (`src/data/cap_loader.py`) parses with a tab separator
  (`sep=r"\t+"`), so the space inside "Unknown Position" no longer splits a
  column. (PATCH_NOTES.md describes an earlier regex-based fix that was later
  replaced.) Covered by `tests/test_cap_loader.py`.

## Corrected 2026-09-23

- **Epoch onsets were computed from the row index, not from time.** The
  loader set `start_sec = i * 30`, which silently assumes the hypnogram starts
  at the exact EDF start time and has no scoring gaps. If either assumption
  fails, every epoch is shifted and "REM" is cut from a different stage.
  Onsets now come from the `Time [hh:mm:ss]` column relative to the EDF
  header start time (midnight crossings and gaps handled; warnings when
  they occur). `scripts/audit_cap_channels.py` now reports the offset, gap
  count and overrun per subject. **How many CAP records actually had an
  offset is not yet known** — the audit has only been run on rbd1.
- **`rswa_index = 0.0` for rbd1 is largely a property of the rule, not proof of
  bad data.** "Whole 30-s REM epoch RMS > 2x median NREM RMS" is nearly blind to
  RSWA: chin tone is physiologically *lower* in REM than in NREM, phasic bursts
  (0.1–5 s) are averaged away over 30 s, and raw RMS (no 10–100 Hz band)
  also measures drift and ECG. `tests/test_rswa_scoring.py` reproduces exactly
  this: RBD-like phasic bursts give `rswa_index == 0` but a clearly non-zero
  mini-epoch index. Added: 3-s mini-epoch activity vs. the REM atonic floor
  (SINBAR-inspired, *not* visual SINBAR scoring) and the REM Atonia Index
  (Ferri 2008/2010). Decision thresholds are chosen under LOSO
  (`src/evaluate.py`), never copied from papers that used other hardware.
  Whether any of these separates rbd from n on real CAP data is still open.
- **The sleep-stager checkpoint expects raw volts, not normalized EEG.**
  The submodule's `data_loader.py` never calls its own `preprocessing.py`
  (bandpass/z-score are dead code there), so the Sleep-EDF model was trained
  on unfiltered Fpz-Cz in volts at 100 Hz. `src/staging/cap_stager.py`
  feeds CAP EEG the same way; amplitude/derivation differences between CAP
  labs and Sleep-EDF are part of the domain shift Phase 2 has to measure.
- **Label unit decided (Phase 1):** the only ground truth in CAP is the
  subject-level diagnosis (22 rbd, 16 n). Per-epoch RSWA values are rule-derived
  *measurements*, not labels. The Mamba classifier in `src/models/mamba_rbd.py`
  is trained on the subject label copied to each REM epoch, so it learns
  "epoch comes from an RBD patient", and it is evaluated only at subject
  level with patient-grouped CV.

## First real-data check, 2026-09-24 (6 recordings: rbd1-3, n1-3)

- **The alignment bug was real and large.** 4/6 hypnograms start after the
  EDF start (n1 +3.5 min, n3 +51 min, rbd1 +2 h 35 min, n2 +3 h 7 min) and
  4/6 have single-epoch scoring gaps. With the Time-based alignment, the
  hypnogram ends within 1 s of the EDF end for rbd1, n1 and n3, which is strong
  evidence that the EDF header time is correct and the offset is real (the
  scoring starts at lights-off). Every result computed with the old
  row-index loader was cut from the wrong part of the night.
- **Mini-epoch index and RAI separate 3 rbd from 3 n perfectly**; the old
  30-s rule does not. With 3 vs 3 that has p = 1/20 by chance alone. Not evidence yet.
- **Open confound: overall chin EMG amplitude.** NREM chin RMS is 4.6-5.8 uV
  in rbd1-3 vs 0.65-1.9 uV in n1-3: it separates the groups on its own, and
  it is not RSWA. RAI uses absolute uV thresholds, so it is exposed to this;
  the mini-epoch index is relative to each subject's own REM floor and is
  not. `scripts/evaluate_rswa.py` now scores NREM/REM amplitude and ECG
  leakage as negative controls and warns when they separate the groups as well
  as a metric does. `ecg_contamination_ratio` / `rswa_mini_index_ecg_gated`
  (src/artifacts.py) test whether "activity" is heartbeat leakage.

## Hypothesis - to be tested

- **Dt-aware Mamba discretization might help on irregularly-gapped overnight
  EEG** (artifact rejection, electrode dropout, stage-transition boundaries
  create real timing irregularity, unlike Sleep-EDF's uniform 30s epochs).
  This is *plausible* but **not supported by existing evidence** - see below.
- **IMU-derived movement can proxy for EMG-derived atonia loss** well enough
  to be clinically useful. No evidence for or against this yet in this
  project; needs a real check against any movement channel available in the
  chosen dataset (Phase 5).

## Currently wrong / not supported - correct before repeating

- **"dt-aware Mamba is a proven commercial advantage."** This is contradicted
  by `mamba-plasticc-transients`: after fixing the `dt_obs` clamping
  instability, dt-aware Mamba trailed plain Mamba by ~0.01-0.02 macro-F1,
  consistently. The project's own conclusion: "content-based selectivity
  alone is already competitive with... explicit real-time awareness." Any
  pitch material should say "we are testing whether X holds in a new domain,"
  not "X is a proven edge."
- **RBD detection as a monolithic target.** Clinical RBD requires video-PSG
  plus a clinical interview; what a wearable can actually estimate is a proxy
  measure - RSWA burden across a night. RBD detection
  rather than RSWA risk score is an overstatement what any of this pipeline does,
  even at 100% technical success.
