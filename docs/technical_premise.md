# Technical premise: what's proven, what's hypothesis, what's currently wrong

Written before Phase 1 starts, so scope creep can be checked against it later.
Update this file, don't quietly let the README drift from it.

## Proven (from your existing repos)

- **Sleep staging works reasonably well from single-channel EEG.**
  `mamba-eeg-sleep-staging`: CNN+Mamba reaches 81.2% accuracy / 0.777 macro-F1
  on Sleep-EDF-20, subject-independent. REM is one of the harder classes but
  the overall pipeline (epoch encoder + sequence head) is validated.
- **Resting-state EEG carries *some* PD-related signal, but it's weak.**
  `parkinsons-eeg-classifier`: LSTM gets 0.667±0.074 accuracy / 0.686±0.075
  AUC, LOSO-CV, n=31. Real, above-chance, but the dataset's own curators
  explicitly warn against treating this as diagnostically meaningful at this
  sample size -- that warning applies with equal force here.

## Hypothesis -- to be tested, not assumed

- **Dt-aware Mamba discretization might help on irregularly-gapped overnight
  EEG** (artifact rejection, electrode dropout, stage-transition boundaries
  create real timing irregularity, unlike Sleep-EDF's uniform 30s epochs).
  This is *plausible* but **not supported by existing evidence** -- see below.
- **IMU-derived movement can proxy for EMG-derived atonia loss** well enough
  to be clinically useful. No evidence for or against this yet in this
  project; needs a real check against any movement channel available in the
  chosen dataset (Phase 5).

## Currently wrong / not supported -- correct before repeating

- **"dt-aware Mamba is a proven commercial advantage."** This is contradicted
  by `mamba-plasticc-transients`: after fixing the `dt_obs` clamping
  instability, dt-aware Mamba trailed plain Mamba by ~0.01-0.02 macro-F1,
  consistently. The project's own conclusion: "content-based selectivity
  alone is already competitive with... explicit real-time awareness." Any
  pitch material should say "we are testing whether X holds in a new domain,"
  not "X is a proven edge."
- **"RBD detection" as a monolithic target.** Clinical RBD requires video-PSG
  plus a clinical interview; what a wearable can actually estimate is a proxy
  measure -- RSWA burden across a night. Calling the output "RBD detection"
  rather than "RSWA risk score" overstates what any of this pipeline does,
  even at 100% technical success.
