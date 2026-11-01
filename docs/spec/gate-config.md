# Gate config

Settings in `voxprint_forge/lora/gate.py`: `PRODUCT_RULE`, `INTERNAL_RULE`, `AVG_MIN`, `PHOTO_FLOOR`, `AGE_MIN`,
`AGE_HARD_MIN`, `MIN_CONFIDENCE`, `INDICATORS`, `BODY_MARKERS`, `MINOR_SIGNALS`, `CHILD_WORDS`, `CHILD_STEMS`,
`CHILD_WORDS_RU`.

## Job-level refusal

`voxprint_forge/lora/job_gate.py` refuses the whole job, not one photo, before training starts:

- any photo with a minor signal (`minor_signal()`: child markers, `child_or_adolescent`, a reading under 16,
  a child keyword). `average_under_18` alone is left to the per-person average, so adults who look 18-20 pass;
- `child_keyword()` on captions (after captioning and after every edit), the trigger, the project folder and
  subject names, sample prompts and settings. It reads EN, RU (declensions, diminutives, slang),
  transliterated RU, leetspeak, mixed Latin/Cyrillic look-alikes, ages under 18 (`16yo`, `16 лет`), and
  school uniform plus an age;
- a check that is missing or errors (judge, estimator, captioner, the preflight itself): `lora.err.check_unavailable`;
- a prepared file or `captions.jsonl` that changed, was added or was removed after the gate. `pin()` writes
  `gate.json` (sha256 per file); `preflight_refuse()` verifies it: `lora.err.dataset_changed`.

`train()` calls `preflight_refuse()` itself, so the CLI and the app cannot skip it. A refusal sets
`gate_passed` to false, deletes `gate.json`, and logs the source to `drops.jsonl` and `refusals.jsonl`.
The message shown is neutral.
