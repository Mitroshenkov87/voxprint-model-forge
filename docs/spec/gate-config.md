# Gate config

Settings in `voxprint_forge/lora/gate.py`: `PRODUCT_RULE`, `INTERNAL_RULE`, `AVG_MIN`, `PHOTO_FLOOR`, `AGE_MIN`,
`AGE_HARD_MIN`, `MIN_CONFIDENCE`, `INDICATORS`, `BODY_MARKERS`, `MINOR_SIGNALS`, `CHILD_WORDS`, `CHILD_STEMS`,
`CHILD_WORDS_RU`.

Job policy (rc.12): `TIERS`, `MINOR_AGE`, `YOUNG_AGE`, `KEEP_AGE`, `CLEAR_PERCENT`, `ALLOWANCE_MIN`, `ALLOWANCE_MAX`,
`ALLOWANCE_FROM`, `ALLOWANCE_TO`, `UNCHECKED`. See docs/LORA.md, "Safety-check log".

Face selection (rule 5): `FACE_RULES`, `PRIMARY_MIN_SCORE`, `STRAY_MAX_AREA`.

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
