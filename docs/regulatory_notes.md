# What "real startup" requires beyond the code -- read before pitching

This is not legal advice (I'm not a lawyer), just a map of the gap between
"working ML pipeline" and "medical screening company," so the roadmap doesn't
accidentally promise the second while building the first.

## Data & ethics

- Any *new* EEG data collected from real people (even friends/family testing
  a prototype) requires informed consent and, for anything beyond a handful
  of self-experiments, likely ethics-board (komisja bioetyczna) review --
  especially data framed as being about disease risk.
- Public datasets used here (CAP Sleep Database, Sleep-EDF, ds002778) each
  have their own reuse terms. `parkinsons-eeg-classifier`'s README already
  documents ds002778's explicit request to contact the curator before any
  formal (non-portfolio) publication -- same diligence applies to CAP Sleep
  Database's license before any commercial use.

## Regulatory

- A tool that outputs a disease-risk score sits squarely in medical device
  software territory (EU MDR / IVDR, and FDA "Software as a Medical Device"
  if ever sold in the US). This applies regardless of how the tool is
  described -- "wellness screening" framing does not exempt a Parkinson's-risk
  claim from device regulation once it's marketed, not just researched.
- None of this blocks building and validating the algorithm now. It blocks
  *selling* it as a screening product before going through that process.

## IP & partnerships

- Talking to Adamed or Politechnika Slaska contacts about this is a genuine
  asset -- but put anything non-public (unpublished results, specific
  hardware plans) under at least an informal mutual understanding before
  sharing, and keep a dated record (even just git commit history + a private
  doc) of what you built when, in case IP questions come up later.
- A GitHub repo with a real commit history, honestly reported negative
  results included, is itself decent evidence of authorship and timeline --
  don't rewrite history to look cleaner than the actual process was.

## What's realistic right now

- **A rigorously validated software pipeline + an honest writeup of its
  limitations** is a strong scholarship/portfolio artifact and a credible
  reason for a mentor conversation. That's an achievable Phase 0-8 outcome.
- **A funded medical-device company** requires a clinical partner, IRB-
  approved data collection, and regulatory strategy that a high-school
  project timeline cannot produce alone. Treat Phase 9 as "here's the
  on-ramp," not "here's the company."
