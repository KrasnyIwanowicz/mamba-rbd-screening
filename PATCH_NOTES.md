# Patch notes — 2026-08-28

Based on reviewing the repo as of commit `441362b` ("load CAP, work on pipeline").
I can't push to your GitHub directly (no write access, only read via public
clone), so these are the changed/new files to copy into your local clone
and commit yourself.

## 1. Critical bug fix: `src/data/cap_loader.py`

`parse_txt_annotations` used `re.split(r"\s+", line)` and assumed
`parts[2]` was the Duration column. It never was — the real row format is
`Stage \t Position \t Time[hh:mm:ss] \t Event \t Duration[s] \t Location`,
and `parts[2]` is either part of "Unknown Position" or the Time field,
depending on whether Position is one word or two. Either way,
`float(parts[2])` throws on every single row, every subject, so
`df_stages` came out empty and `load_subject()` silently returned `[]`
with no error.

**Fix:** anchor the split on the one fixed-format token in the row — the
`hh:mm:ss` time — via regex, instead of a naive whitespace split. Also now
filters strictly on `Event.startswith("SLEEP-")` so CAP microstructure
rows (`MCAP-A1/A2/A3`, interleaved with the real epochs, variable
duration) don't get miscounted as extra 30s epochs and shift the
alignment between `df_stages` rows and the signal windows sliced in
`load_subject`.

Added `tests/test_cap_loader.py` — regression tests using synthetic data
in the real format, so this specific bug can't silently come back.

## 2. Premise correction: RSWA has no per-epoch labels in this dataset

Verified against physionet.org/content/capslpdb/1.0.0/: CAP's annotations
are R&K sleep macrostructure + CAP (EEG) microstructure only — nothing
about muscle tone events. The original Phase 3 plan ("train a classifier
on EMG ground truth") assumed a label that doesn't exist here.

Added `src/rswa_scoring.py`: a rule-based scorer (chin EMG RMS per REM
epoch vs. that subject's own median NREM RMS as baseline,
`threshold_multiplier` controls sensitivity) instead of a supervised
classifier. This is validated with `tests/test_rswa_scoring.py`. It
produces a per-subject `rswa_index` — legitimately comparable between the
"rbd" and "n" groups (real ground truth), but NOT a per-epoch
precision/recall number against any gold standard (none exists here).

Updated `docs/technical_premise.md` and this note explain the distinction
so a future readme/pitch doesn't quietly conflate the two.

## 3. Housekeeping

- `requirements.txt`: added `tqdm` (used by `scripts/download_cap.py` and
  `scripts/audit_cap_channels.py`, wasn't listed).
- `src/data_loader.py`: added a header note flagging it as the old Phase 0
  stub, now superseded by `src/data/cap_loader.py`. Recommend deleting it
  (and `src/dataset.py` if nothing imports it yet) rather than maintaining
  two loaders in parallel — didn't delete it myself since that's a bigger
  call than a bug fix.

## Not done yet (didn't want to guess)

- `scripts/audit_cap_channels.py` needs to actually run against your local
  40GB download to know the real per-subject channel names — I can't run
  that here (the files aren't on this machine and physionet.org's raw
  files are too large / not in this sandbox's reachable domains). Once you
  run it, share `reports/cap_channel_audit.csv` and I can help interpret
  it and wire the confirmed channel names into `CAPSleepDataset`.
- `src/pipeline.py` and `src/dataset.py` are still the original Phase 0
  stubs — wiring the sleep-stager submodule + `rswa_scoring.py` into an
  actual end-to-end script is the natural next step once the channel audit
  confirms which EEG channel to feed the stager.
