# Model Card — PTSD Facial-Affect Screening

A short, honest description of what the model is, what it's for, and where it
must not be used. Inspired by the model-cards framework (Mitchell et al.).

## Model details

- **Task:** binary classification of a short facial-reaction video clip into
  `control` vs `ptsd_risk` (a screening *flag*, not a diagnosis).
- **Input:** a `(T, 8)` temporal sequence of FACS-grounded Action Units derived
  from MediaPipe FaceLandmarker (478 landmarks), normalised and resampled.
- **Architecture:** temporal 1D-CNN (`TemporalAUNet`, PyTorch) as the headline
  model; `HistGradientBoostingClassifier` over 50 aggregate features as a
  dependency-light baseline.
- **Output:** `P(ptsd_risk)` plus a calibrated **risk band** at a threshold
  chosen under an FPR budget.
- **Version:** 0.2.0.

## Intended use

- **In scope:** research; methods development; a *decision-support* aid that
  flags clips for review by a qualified clinician, used **alongside** validated
  instruments.
- **Out of scope (do not do this):** autonomous or sole-basis diagnosis;
  triage without a human in the loop; employment, insurance, legal, forensic,
  immigration, or any high-stakes determination about a person; surveillance;
  inferring mental-health status of non-consenting individuals.

## Factors & performance

The only performance numbers in this repository come from **synthetic** data
(README → *Demo results*): test ROC-AUC ≈ 0.86 at a 0.20 FPR budget. These are
illustrative of the pipeline working end-to-end and **do not** characterise
real-world accuracy for any population.

Performance has **not** been evaluated across demographic groups, skin tones,
ages, cameras, lighting, or recording protocols. Facial-analysis systems are
known to vary across such factors; this must be measured before any real use.

## Ethical considerations

- **Affect cues are correlates, not criteria.** PTSD is a clinical diagnosis;
  facial dynamics can at most contribute weak evidence for human review.
- **Asymmetric harms.** A false "risk" label can stigmatise; a false "control"
  label can falsely reassure. The FPR-budget design targets the first, but both
  matter and depend on deployment context.
- **Consent & dignity.** Facial video is sensitive biometric data; processing
  it requires informed consent and appropriate governance (see
  [`DATA_PRIVACY.md`](DATA_PRIVACY.md)).
- **Human oversight is mandatory.** The tool is built to assist clinicians, not
  replace their judgement.

## Limitations

- Synthetic-only evidence here; no validated real-world metrics are claimed.
- AU estimation from landmark geometry approximates true FACS coding and
  degrades with occlusion, extreme pose, and poor video quality (low-quality
  clips are rejected via a missing-frame threshold).
- Small held-out sets make thresholded metrics noisy; the AUC is the more
  stable summary.

## Caveats & recommendations

Before any deployment beyond research: obtain ethics/IRB approval; validate on
a representative, consented dataset with subject-level evaluation; run a
fairness analysis across relevant subgroups; calibrate and re-select the
operating point with clinicians; and keep a clinician in the loop for every
decision.
