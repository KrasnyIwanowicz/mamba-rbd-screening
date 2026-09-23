# Roadmap

Legend: `[x]` done · `[~]` code ready and tested on synthetic data, still
needs a run on the real CAP download · `[ ]` not started.

Phased like the other two projects. Each phase should end with something that
runs on synthetic data (for CI) before it's tried on the real CAP dataset —
same discipline as `parkinsons-eeg-classifier`'s 17 synthetic-data tests.

## Phase 0 — Scaffold + honest premise (this commit)
- [x] Repo structure, README, config, requirements, license
- [x] `docs/technical_premise.md`: state clearly which claims are proven
      (sleep staging works, PD-vs-control from resting EEG has weak-but-real
      signal) vs. hypotheses to test (dt-aware Mamba in this domain, EMG→IMU
      transfer) vs. things that are just wrong until shown otherwise
      (dt-aware Mamba is NOT a proven commercial advantage — see PLAsTiCC result)
- [x] `git submodule add` the sleep-staging repo instead of copying `mamba_block.py`
      a third time — one canonical implementation, not three forks of it

## Phase 1 — Data audit (before writing the loader)
- [~] Download CAP Sleep Database, inventory actual montages per RBD subject
      (`scripts/download_cap.py`, `scripts/audit_cap_channels.py`; so far only
      rbd1 is audited in `reports/cap_channel_audit.csv`)
      (CAP is a multi-lab archive — don't assume every recording has the same
      channels; the parkinsons-eeg-classifier README's "structure as actually
      shipped" section is the right template for how to document this)
- [~] Confirm which subjects have both a hypnogram AND scored RSWA/EMG events
      (not all CAP annotations are equally complete). CAP has **no** scored
      RSWA events (see docs/technical_premise.md), so this becomes: hypnogram
      present, aligned to the EDF, with REM and N2/N3 epochs. The audit now
      reports this per subject (`hyp_offset_s`, `hyp_gaps`, `n_rem_epochs`, ...)
- [x] Decide the label unit: per-30s-REM-epoch atonia loss (fine-grained) vs.
      per-subject RBD/control (coarse) — probably need both, like the PD project
      did (epoch-level AND subject-level accuracy).
      **Decision:** subject-level diagnosis is the only ground truth. Per-epoch
      RSWA values are rule-derived measurements, not labels, so no per-epoch
      accuracy is reported against them.

## Phase 2 — Sleep stager transfer
- [~] Run the existing `mamba-eeg-sleep-staging` checkpoint on CAP recordings
      (`scripts/evaluate_stager_on_cap.py` + `src/staging/cap_stager.py`;
      blocked on the checkpoint, which is not in git: train it in the submodule
      or copy `mamba_best.pt` to `external/sleep_staging/results/`)
      (different hardware/montage than Sleep-EDF-20 — expect a domain-shift
      accuracy drop, measure it, don't assume it transfers cleanly)
- [ ] If the drop is large: decide fine-tune vs. retrain-from-scratch on CAP
- [~] Output: per-subject list of REM epoch windows (`reports/stager_rem_windows.json`)

## Phase 3 — RSWA detector (EMG ground truth)
- [x] Baseline: EMG RMS/amplitude threshold per REM epoch (the actual clinical
      scoring heuristic — this is the floor to beat, same role as the SVM
      baseline in parkinsons-eeg-classifier). `src/rswa_scoring.py`: legacy
  30-s rule plus 3-s mini-epoch index and REM Atonia Index. Subject-level
  evaluation with LOSO-chosen thresholds: `scripts/evaluate_rswa.py`.
  Numbers on real CAP are still missing (`[~]`).
- [~] Learned classifier: EMG + EEG features -> atonia maintained/lost per
      REM epoch. Leave-one-subject-out CV (same rigor as the PD project — this
      dataset is also small, n≈16 RBD + controls, so subject leakage is the
      main way to get a fake-looking result). `src/training/train_rbd.py` now
  runs every LOSO fold over rbd/n subjects only, for several seeds. Note: with no
  per-epoch labels it predicts *subject* diagnosis, not per-epoch atonia.
- [~] Report accuracy, sensitivity/specificity, AND per-seed variance — the PD
      project's LSTM had a 15-point spread across 3 seeds; expect similar here
      and report it, don't hide behind a single lucky run

## Phase 4 — dt-awareness, tested not assumed
- [ ] State the null hypothesis explicitly: content-based Mamba selectivity
      is already sufficient, exactly like it was on PLAsTiCC — dt-awareness
      needs to earn its place with a real head-to-head, same protocol as
      `mamba-plasticc-transients` Phase 4 (identical training recipe across
      heads, multiple seeds, clamp dt_obs before training it)
- [ ] If it doesn't beat plain Mamba here either: report that honestly, same
      as the other two repos do with their negative/mixed findings. A second
      honest negative result is a *stronger* portfolio signal than a cherry-
      picked positive one, not a weaker one — reviewers who read all three
      repos will notice the pattern of rigor either way

## Phase 5 — EMG → IMU transfer (the actual hard problem)
- [ ] This is the load-bearing, unsolved piece of the whole concept. EMG
      measures muscle electrical activity directly; IMU measures resulting
      *movement*, which only appears if atonia loss is severe enough to
      produce visible motion (many RSWA epochs show EMG tone increase with
      no visible movement at all — "tonic" RSWA vs "phasic" RSWA in the
      literature)
- [ ] If CAP recordings include any actigraphy/movement channel: use it as
      an IMU proxy and measure how much RSWA signal survives EMG->movement
      substitution. If none exist: this needs to be stated as an open
      hardware-validation question for the co-founder, not quietly assumed
      solved in the pipeline

## Phase 6 — End-to-end pipeline + night-level score
- [~] Chain Phase 2 + Phase 3(or 5) into `src/pipeline/rbd_pipeline.py`
      (reference hypnogram or auto-stager -> REM -> RSWA metrics -> night score;
      no label without a LOSO-calibrated threshold)
- [ ] Compare against the aktygrafia/GBT baseline the concept doc references
      as the incumbent approach — that's the actual competitive benchmark,
      not "no baseline"

## Phase 7 — Explainability
- [ ] Extend the SHAP/attention-extraction pattern from the other two repos
      to this pipeline: which features/channels drove the RSWA call

## Phase 8 — MLOps
- [x] Tests on synthetic data (no download needed for CI), CI workflow, mypy
      — same bar as `parkinsons-eeg-classifier`

## Phase 9 — "Real startup" layer (not code)
See `docs/regulatory_notes.md`. Short version: this phase is a scholarship
portfolio piece and a credible conversation-starter with Adamed / Politechnika
Śląska mentors right now. It becomes an actual company only after clinical
validation with a real institutional partner — code quality doesn't shortcut
that requirement, and no pitch deck should imply otherwise.
