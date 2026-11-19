# Gate config

Settings in `voxprint_forge/lora/gate.py`: `PRODUCT_RULE`, `INTERNAL_RULE`, `AVG_MIN`, `PHOTO_FLOOR`, `AGE_MIN`,
`AGE_HARD_MIN`, `MIN_CONFIDENCE`, `INDICATORS`, `BODY_MARKERS`, `MINOR_SIGNALS`, `CHILD_WORDS`, `CHILD_STEMS`,
`CHILD_WORDS_RU`.

Job policy (rc.12): `TIERS`, `MINOR_AGE`, `YOUNG_AGE`, `KEEP_AGE`, `CLEAR_PERCENT`, `ALLOWANCE_MIN`, `ALLOWANCE_MAX`,
`ALLOWANCE_FROM`, `ALLOWANCE_TO`, `UNCHECKED`. See docs/LORA.md, "Safety-check log".

Face selection (rule 5): `FACE_RULES`, `PRIMARY_MIN_SCORE`, `STRAY_MAX_AREA`.

No-face majority: `NO_FACE_MAJORITY` (default `0.80`), `NO_FACE_KEEP`. After every photo is decided,
`apply_no_face_majority()` computes the share of the checked photos kept with a detected face (no reasons, the `full`
path and the `clear` tier; body-only and no-person photos do not count) over all checked photos (an empty set is 0).
At `NO_FACE_MAJORITY` or more, a photo whose only reason is `estimator_no_face` (a face was seen but the age estimator
found none, so its age could not be checked) is kept: tier `clear`, no reasons, and `keep_reason`
`kept_no_face_majority` plus `no_face_ratio` in its trace and `gate_trace.jsonl` row; its provisional safety drop is
removed from `drops.jsonl`. A photo that is `minor`, `strong` or has a judge age under `MINOR_AGE` stays dropped, and
so does any photo with another reason (an under-age or margin face, markers, a check that did not run). Below the
share, nothing changes. The pass runs before the cross-photo average and `decide_job()`.

Attested-adult person sets (final rule, 2026-11-18): `AGE_FLAG_KEEP_SHARE` (`0.10`), `NO_FACE_KEEP_ATTESTED`,
`AGE_FLAG_KEEP`, `AGE_FLAG_REASONS`, `AGE_FLAG_KEPT`, `AGE_FLAG_DROPPED`. `apply_attested_keeps()` runs in
`lora prepare` (product rule only) right after the no-face majority, before the cross-photo average and `decide_job()`:

- Rule 1, no face -> keep (`no_face_keepable()`): MiVOLO ran (`ok`) and found no face, the only reason is
  `estimator_no_face` (or none, the 18-20 judge-only margin), and the judge answered with confidence of at least
  `MIN_CONFIDENCE` and no age under `MINOR_AGE`. The photo is kept whatever the no-face ratio: tier `clear`, no reasons,
  `keep_reason` `kept_no_face`, `kept_reasons` (the reasons before the keep).
- Rule 2, age flags (`age_flagged()`): the face path, reasons only from `AGE_FLAG_REASONS` (`single_under_16`,
  `average_under_18`), MiVOLO's youngest age under `MINOR_AGE`, while the judge answered with confidence of at least
  `MIN_CONFIDENCE`, an age of `KEEP_AGE` (20) or older, no `present` indicator and no `child_or_adolescent`. These
  MiVOLO false positives are kept up to `keep_limit = floor(AGE_FLAG_KEEP_SHARE * input photos)` (`age_flag_keep_limit()`,
  exact arithmetic): 18 of 180. Small sets round down: 9 inputs give `floor(0.9) = 0`, so a single flagged photo is
  dropped. The input photos are the distinct photos given to the prepare run, the ones the filters set aside included.
  Which flagged photos are kept is deterministic: they are sorted by file name (plain string order, then photo id), never
  by batch or processing order, and the first `keep_limit` are kept (tier `clear`, `keep_reason`
  `kept_age_flag_false_positive`, `age_flag` `kept_false_positive`). The rest keep their tier and reasons, get
  `age_flag` `dropped_over_limit`, and are dropped with the normal `lora.drop.safety` row.
- There is no refusal, no stop and nothing to confirm: `decide_job(exempt=...)` does not count the photos dropped past the
  limit toward `allowance(n)`. `decide_job(relaxed=...)` does not count the photos these rules kept toward the
  `CLEAR_PERCENT` share, so a judge-side strong photo refuses exactly as before.
- Unchanged: sets that are not attested, judge-side minor findings (markers, `child_or_adolescent`, a judge age under 18,
  or under 20 next to a MiVOLO flag), low judge confidence, child keywords and text checks, the identity average, and
  every check that did not run (`UNAVAILABLE` still refuses; `judge_incomplete` stays unchecked).
- Audit: `gate.json` gets `age_flags` (`inputs`, `age_flag_keep_share`, `age_flag_keep_limit`, `age_flagged`,
  `age_flag_kept`, `age_flag_dropped`, `no_face_kept`, each entry `file`, `id`, `reasons`, final `tier`); a caption
  re-pin keeps it and `verify()` does not read it. Every `gate_trace.jsonl` row has `kept_reasons`, `age_flag` and
  `age_flag_limit`.

## Job-level refusal

`voxprint_forge/lora/job_gate.py` refuses the whole job, not one photo, before training starts:

- the photos' tiers, by the job policy (`decide_job()`): only clear photos are trained on. At `CLEAR_PERCENT`
  clear photos the job trains; below it a strong photo refuses, and more ambiguous photos than `allowance(n)`
  refuse. Margin and unchecked photos are dropped and never refuse by themselves;
- `child_keyword()` at any ratio: in a file name or photo metadata, in captions (after captioning and after
  every edit), the trigger, the project folder and subject names, sample prompts and settings. It reads EN, RU (declensions, diminutives, slang),
  transliterated RU, leetspeak, mixed Latin/Cyrillic look-alikes, ages under 18 (`16yo`, `16 лет`), and
  school uniform plus an age;
- a check that is missing or errors (judge, estimator, captioner, the preflight itself), whatever the tiers:
  `lora.err.check_unavailable`;
- a prepared file or `captions.jsonl` that changed, was added or was removed after the gate. `pin()` writes
  `gate.json` (sha256 per file); `preflight_refuse()` verifies it: `lora.err.dataset_changed`.

`train()` calls `preflight_refuse()` itself, so the CLI and the app cannot skip it. A refusal sets
`gate_passed` to false, deletes `gate.json`, and logs the source to `drops.jsonl` and `refusals.jsonl`.
The message shown is neutral.

`lora prepare` writes `gate_run.json` before the first photo and sets it to `complete` when the run
finishes. Each trace row and each safety drop is synced as soon as that photo is decided. If the
marker is left `running` or `interrupted`, training will not start until prepare runs again. A photo
with no trace row was not checked and is not treated as kept. That block does not set a photo aside
and does not add a refusal row.
