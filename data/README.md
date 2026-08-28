# Data

No raw data is committed to this repo (clinical PSG data, and too large for git).

## CAP Sleep Database (primary)

```bash
pip install wfdb
python -c "import wfdb; wfdb.dl_database('capslpdb', dl_dir='data/raw/capslpdb')"
```

Source: https://physionet.org/content/capslpdb/1.0.0/
License: check the current PhysioNet license page before any commercial use --
this project currently uses it for research/portfolio purposes only.

**Before writing any loading code (Phase 1):** inventory the actual channel
names and annotation completeness per subject. CAP is an archive of recordings
from multiple sleep labs over many years -- montage and annotation format are
NOT guaranteed uniform across subjects, unlike Sleep-EDF-20's cleaner single-
protocol structure. Document what you actually find, the way the
`parkinsons-eeg-classifier` README documents ds002778's real structure instead
of the structure one might assume.

## Sleep-EDF-20 (for the sleep-stager submodule only)

Already handled inside the `mamba-eeg-sleep-staging` submodule -- see its own
README, not duplicated here.
