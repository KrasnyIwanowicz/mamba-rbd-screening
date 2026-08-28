# Roadmap

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
- [ ] `git submodule add` the sleep-staging repo instead of copying `mamba_block.py`
      a third time — one canonical implementation, not three forks of it

## Phase 1 — Data audit (before writing the loader)
- [ ] Download CAP Sleep Database, inventory actual montages per RBD subject
      (CAP is a multi-lab archive — don't assume every recording has the same
      channels; the parkinsons-eeg-classifier README's "structure as actually
      shipped" section is the right template for how to document this)
- [ ] Confirm which subjects have both a hypnogram AND scored RSWA/EMG events
      (not all CAP annotations are equally complete)
- [ ] Decide the label unit: per-30s-REM-epoch atonia loss (fine-grained) vs.
      per-subject RBD/control (coarse) — probably need both, like the PD project
      did (epoch-level AND subject-level accuracy)

## Phase 2 — Sleep stager transfer
- [ ] Run the existing `mamba-eeg-sleep-staging` checkpoint on CAP recordings
      (different hardware/montage than Sleep-EDF-20 — expect a domain-shift
      accuracy drop, measure it, don't assume it transfers cleanly)
- [ ] If the drop is large: decide fine-tune vs. retrain-from-scratch on CAP
- [ ] Output: per-subject list of REM epoch windows

## Phase 3 — RSWA detector (EMG ground truth)
- [ ] Baseline: EMG RMS/amplitude threshold per REM epoch (the actual clinical
      scoring heuristic — this is the floor to beat, same role as the SVM
      baseline in parkinsons-eeg-classifier)
- [ ] Learned classifier: EMG + EEG features -> atonia maintained/lost per
      REM epoch. Leave-one-subject-out CV (same rigor as the PD project — this
      dataset is also small, n≈16 RBD + controls, so subject leakage is the
      main way to get a fake-looking result)
- [ ] Report accuracy, sensitivity/specificity, AND per-seed variance — the PD
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
- [ ] Chain Phase 2 + Phase 3(or 5) into `src/pipeline.py`
- [ ] Compare against the aktygrafia/GBT baseline the concept doc references
      as the incumbent approach — that's the actual competitive benchmark,
      not "no baseline"

## Phase 7 — Explainability
- [ ] Extend the SHAP/attention-extraction pattern from the other two repos
      to this pipeline: which features/channels drove the RSWA call

## Phase 8 — MLOps
- [ ] Tests on synthetic data (no download needed for CI), CI workflow, mypy
      — same bar as `parkinsons-eeg-classifier`

## Phase 9 — "Real startup" layer (not code)
See `docs/regulatory_notes.md`. Short version: this phase is a scholarship
portfolio piece and a credible conversation-starter with Adamed / Politechnika
Śląska mentors right now. It becomes an actual company only after clinical
validation with a real institutional partner — code quality doesn't shortcut
that requirement, and no pitch deck should imply otherwise.
