# Product age gate: specification (draft)

**Status:** specification only. Nothing in this document changes code or behaviour. The gate that runs today is the
one described in [LoRA trainer](../LORA.md) (VLM judge and MiVOLO v2, every person 20 or older, fail-closed), and it
stays the default until a later change implements this spec, ships its tests and passes calibration.

## Goal

Decide whether every person in a photo is an adult the way a careful average human reviewer would: from several
adult markers together, not from one number. Two targets, in this order:

1. **Zero passes below 18.** No calibration result may trade this away.
2. **Few false refusals for ordinary adults**, including adults in their early twenties.

The minors ban and the age check apply to every photo of every kind. There is no setting and no override.

## Signals

| Signal | What it is | Licence |
| --- | --- | --- |
| VLM adult-marker rubric | The judge VLM answers a fixed rubric per person: body proportions, adult secondary features, facial maturity, context (setting, clothing, props, text). Each item is `adult`, `child_or_adolescent` or `unsure`, with a confidence and an apparent age. | judge model as today (Apache-2.0) |
| Estimator 1 | MiVOLO v2 (face and body), one age per person. | Apache-2.0 |
| Estimator 2 | A second, independently trained estimator. Candidates below. | must be Apache-2.0 |
| Metadata | File names, captions, tags and EXIF keywords checked against the child-keyword list. | - |

Estimator 2 candidates (model card licence checked on Hugging Face, 2026-10-18):

- `prithivMLmods/open-age-detection`: Apache-2.0, fine-tuned from `google/siglip2-base-patch16-512` (Apache-2.0). It
  classifies age groups (child 0-12, teenager 13-20, adult 21-44, ...) rather than giving one number, so it is used as a
  youth signal: P(child) + P(teenager) above a calibrated threshold counts as an estimate under 20.
- `dima806/facial_age_image_detection`: Apache-2.0 card; training data is not stated, so it needs a data-terms review
  before it can ship.
- Rejected: `abhilash88/age-gender-prediction`. The card says Apache-2.0, but it is trained on UTKFace, whose terms
  allow only non-commercial research use.

The final pick needs a training-data terms review as well as the card licence. The ensemble median is the median of
every available estimate for a person (MiVOLO face, MiVOLO body, estimator 2 where it gives a number).

## Decision

Rules run in this order. The first one that matches decides.

1. **Hard drop** (and a clear minor signal blocks the whole project, as today):
   - any rubric item answered `child_or_adolescent`, for any person;
   - any estimate from any estimator below 18;
   - a child keyword in the metadata;
   - doubt: any rubric item `unsure` on proportions or facial maturity, low confidence, judges that disagree by more
     than the calibrated margin, a malformed answer, a judge error, a missing judge or a stale record. Fail-closed.
2. **Youngest-adult band.** Any signal from 20 to 22, a VLM apparent age of about 20, or a student or school-uniform
   context: allowed only when the content is not flagged `adult_content`. With `adult_content=true` the photo is set
   aside (`lora.drop.safety`).
3. **Pass** when the rubric says clearly adult for every person and the ensemble median is 20 or higher.
4. Anything else is set aside.

A photo with no visible face follows the existing minimal body check for close-up sets; rules 1 and 2 still apply to
every signal it has.

## What does not change

- The face cutoff is not lowered. No rule above passes a face estimate that the current gate would drop for being under 20.
- Internal and synthetic test sets keep the strict shared gate (cutoff 20, fail-closed). The new rules may only run
  there as a shadow log.
- The drop reason stays neutral (`lora.drop.safety`). Wording in the GUI stays neutral.
- Outputs carry ids and sha256 prefixes only; no image or prompt text is logged.

## Calibration

Inputs, as aggregate numbers only:

- **Shadow v3 logs** from the internal runs: per frame, the strict decision, the shadow decision, every estimate and
  the rubric answers. They are replayed through this spec and compared with the strict decision.
- **Internal synthetic run, 15 frames** (one fictional adult identity, anchored at a stated age of 28, prompted
  at 20, 30 and 35; 5 non-explicit, 10 flagged `adult_content`): the strict gate kept 15 of 15. MiVOLO read 26.5-30.6 on the 14 frames with one person.
  The VLM apparent age was 25 on all 5 non-explicit frames and on 5 of the 10 `adult_content` frames, and **20 on 5 of
  the 10**, where MiVOLO read 27.3-29.6. MiVOLO alone would have passed those 5; under this spec they are set aside
  (rule 2), and the 7-10 year gap between the judges counts as disagreement (rule 1).
- **Internal anchor runs** (96 and 48 portraits of one identity): MiVOLO medians moved only 27.2-33.8 across prompt
  changes. One estimator is not a reliable age control; the margin between judges has to be calibrated, not assumed.

Procedure:

1. Replay the logs; report pass, set-aside and drop counts per rule.
2. A human reviews the disagreements and every pass with a signal under 23. Their labels set the disagreement margin
   and the estimator 2 youth threshold.
3. Acceptance: no pass with any signal under 18; no pass that a reviewer labels childlike, adolescent or doubtful;
   a false refusal rate for clearly adult photos that is lower than the current gate's on the same set, reported with
   its count.

## Open questions

- The disagreement margin (starting point: 6 years between the VLM apparent age and the ensemble median).
- The second estimator choice, after the training-data terms review.
