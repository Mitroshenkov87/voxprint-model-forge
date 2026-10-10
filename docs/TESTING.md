# Lean testing

Cheap checks come first. A GPU run is a few hard cases, and it stops when the
answer is already clear. A person reviews a handful of full-resolution labelled
images.

## Order

1. **Plain Linux, no GPU.** CLI help, `--dry-run`, and `--json` contracts. Pytest
   on Ubuntu and Windows is this tier. It mocks remotes and does not load a
   model onto a device.
2. **A few hard cases.** Image checks use the lean lists in
   `voxprint_forge.diffusion.prompts` (fine detail, a clothed adult, thin
   structure, one of each control type). One seed. The full prompt catalogues
   stay in the file and are not the job.
3. **Stop when the interval decides.** After a minimum count, a 95% interval
   that lies entirely inside a gate, or entirely outside it, ends that
   component. Refusal rate uses a Wilson interval around one half. LPIPS and
   CLIP use a Student-t interval on the per-sample scores. A clear failure of
   one metric stops the component. The recorded mean is the samples that ran.
4. **A review pack, not a thumbnail wall.** The job writes up to four
   full-resolution reference and candidate PNGs plus `labels.md` under
   `checks/review-<name>/`.

## Caps

| Check | Hard cases | Minimum before a stop |
| --- | --- | --- |
| Base and Qwen image | 8 prompts, seed 0 | 4 |
| Control | 6 typed cases, seed 0, one of each type first | 3 |
| Inpaint | 4 prompts | 2 |
| Edit | 6 pairs, seed 0 | 4 |
| LLM refusal | plan cap, default 4 prompts per language | 4 |
| Perplexity and prose | 1 chunk, 1 sample | already one shot |

`~/forge-out/EDIT_TRIM` can only shrink the edit list. It does not turn the
lean list back into the 24-pair catalogue.
