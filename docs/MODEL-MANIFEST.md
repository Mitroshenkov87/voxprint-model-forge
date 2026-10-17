# Model manifest

`voxprint-model.json` is the shared model record agreed with VOXPrint Components.
The schema id is `voxprint-model/1`. It is provisional: Components still has to
review the matching MDL change, so Forge does not push the model index unless
you pass `--push`.

This file is not the Plotweaver run export. `run-all` still writes
`manifests/<date>-a100.json` and `manifests/<date>-a100.MODELS.md`. Canvas
still writes `manifests/<date>-canvas-zimage.json` and
`manifests/<date>-canvas-zimage.MODELS.md`. Copy those back as
before. `voxprint-model.json` is one document per published model, uploaded
to the repo.

## Where it is uploaded

The file sits at the repo root, next to `README.md`, on every mirror that
actually receives the publish:

- Hugging Face: `HF_USER/voxprint-<id>` (canvas repos keep their existing slugs).
- Kaggle: the private model, in each variation, under the name `voxprint-model.json`.
- Drive: `Voxprint/model-forge/<date>/<model>/voxprint-model.json`, when Drive is the destination.

The manifest does not list itself under `files`. `index.json` stores the
sha256 of `voxprint-model.json` separately.

If Hugging Face and Kaggle fail and Drive is the fallback, the file is
rewritten so `mirrors` contains only the Drive folder. A failed Hub upload
is not recorded as a mirror. INT4 lists both Hub and Drive when both are
attempted. If Hub succeeds and the later Drive copy fails, the Hub file can
still name that Drive folder.

## Fields

Every field is required. The SPDX key is `license`.

| Field | Meaning |
| --- | --- |
| `schema` | `voxprint-model/1` |
| `id` | Model id. Hugging Face slugs drop a leading `voxprint-`. |
| `display_name` | Card title. LLM publishes use `VOXPrint <id>`. |
| `app_ids` | `plotweaver` and `audiobook-builder` for text models. `canvas` for image, edit, and video. |
| `task` | `text-generation`, `text-to-image`, `image-to-image`, or `text-to-video`. |
| `license` | Lowercase SPDX id, such as `apache-2.0`. |
| `files` | `{path, size, sha256}` for every uploaded file except this manifest. |
| `mirrors` | `{kind, ref, revision}`. `kind` is `hf`, `kaggle`, `gdrive`, `modelscope`, or `url`. |
| `tier` | `min_vram_gib`, `recommended_vram_gib`, `gpu_min_cc`. |
| `runtime` | `engine` is `llama.cpp`, `diffusers`, or `nunchaku`, plus `version_req`. |
| `base_model` | Upstream repo id. Empty when a dry-run backfill could not read one. |
| `card_url` | `https://huggingface.co/<repo>` for the mirror that holds this file. An `hf` mirror with `ours` set, or the single `Mitroshenkov87` mirror, wins over an upstream author listed first. One `hf` mirror is that repo. Empty when there is no `hf` mirror. |
| `flags` | Includes `least-censored` when the plan ablates or the base was already abliterated. |
| `on_demand` | `true` for these downloadable weights. |
| `build` | `forge_version`, `tag`, `date`. `tag` is `""` until a release tag is passed in. |

`files[].path` is relative to the repo root and is the same string on every
mirror. Nested canvas paths stay nested (`checks/grid.avif`). The Kaggle
variation uploader still flattens `/` to `__` for those nested bytes; the
manifest does not rename them. `voxprint-model.json` itself has no slash, so
its name is the same on Hugging Face, Kaggle, and Drive.

Forge does not publish to ModelScope. `modelscope` and `url` are in the
schema so a later mirror can use them.

## VRAM and compute capability

`min_vram_gib` and `recommended_vram_gib` are GiB. MDL treats a 24 GB card
as about 23.5 GiB. Forge converts a decimal GB budget with `gb * 1000 / 1024`
and rounds to the nearest 0.5, so 24 GB becomes 23.5 and 16 GB becomes 15.5.
The smaller tier is `min_vram_gib`. The larger tier is `recommended_vram_gib`.
One tier sets both.

`gpu_min_cc` defaults to 6.0 for `llama.cpp` and 8.0 for `diffusers` and
`nunchaku`. INT4 already treats compute capability 8.0 as the floor.

## Runtime

| Publish | `engine` | `version_req` |
| --- | --- | --- |
| LLM GGUF | `llama.cpp` | Plan `llama_cpp_tag` (default `b11476`) |
| Canvas dtype conversion | `diffusers` | `>=0.37.1,<0.38` |
| Wan 2.2 GGUF and FP8 | `llama.cpp` | `b3962` |

The Wan repo also contains FP8 weights. Those files are listed in `files`.
`runtime.engine` stays `llama.cpp` because the GGUF quants are the llama.cpp
deliverable.

## Index

`voxprint-forge-cli` and `forge` share this parser.

```text
voxprint-forge-cli index build --from DIR --out index.json
voxprint-forge-cli index build --from DIR --out index.json --revision REPO=40HEX
voxprint-forge-cli index push --index index.json --push
voxprint-forge-cli index host --from DIR --push
voxprint-forge-cli index backfill --out DIR
```

`index build` scans `DIR` for `voxprint-model.json` and writes:

```json
[{"id": "tiny", "repo": "Mitroshenkov87/voxprint-tiny", "revision": "<40-hex commit of voxprint-model.json>", "sha256": "<sha256 of voxprint-model.json>"}]
```

`repo` is the mirror that holds `voxprint-model.json`: the `hf` mirror with
`ours` set, or the single mirror under `Mitroshenkov87`. The first `hf`
mirror may be the upstream author, and that author repo has no descriptor.
A manifest with no such mirror is an error. `revision` is the commit in
that repo that contains `voxprint-model.json`. The build asks Hugging Face
for the last commit that touches the file on the published ref, or takes
`--revision REPO=SHA` (repeatable, 40 lowercase hex digits). It does not
use the mirror's `revision`. That field is often a later weights commit,
and the descriptor is not in it. If the commit cannot be resolved, the
build fails and writes no index. `sha256` is the manifest file, not a weight.

`index push` without `--push` exits 2 and does not upload. With `--push` it
uploads `index.json` to the dataset `Mitroshenkov87/voxprint-model-index`.
CI does not pass `--push`.

`index backfill` lists `Mitroshenkov87/voxprint-*` model repos, skips
`voxprint-model-index`, and writes `DIR/<owner>/<repo>/voxprint-model.json`.
It does not upload. A repo with no licence tag, or a file larger than 8 MiB
that has no LFS sha256, is skipped and named in the result. VRAM defaults
to 16 GB and 24 GB when the file names do not contain `16gb` or `24gb`.

## Official-source models

A model that Forge uses straight from its author (a LoRA base, the captioner,
the age-gate judges) has no `Mitroshenkov87` repo, and no mirror is made. Its
descriptor lists only the author's source: the `hf` repo at a pinned commit,
or a `url` folder for a file that is not on Hugging Face. Each file has its
size and sha256 from that commit. `card_url` is the author's page.

The descriptor is kept in the index dataset itself:

```text
Mitroshenkov87/voxprint-model-index (dataset)
  descriptors/<id>/voxprint-model.json
```

The index row keeps the same four keys:

```json
{"id": "z-image", "repo": "datasets/Mitroshenkov87/voxprint-model-index", "revision": "<dataset commit with the descriptor>", "sha256": "<sha256 of voxprint-model.json>"}
```

`repo` is the Hub path of the dataset, so the descriptor URL is
`https://huggingface.co/<repo>/resolve/<revision>/descriptors/<id>/voxprint-model.json`.
A reader checks that file against `sha256` like any other descriptor. The
weights still come from the author's URLs listed in the descriptor.

A descriptor is official-source when it has an `hf` or `url` mirror, no
mirror marked `ours`, and no mirror under `Mitroshenkov87`. Publish order:

```text
forge index host --from DIR --push             # one dataset commit; prints "commit"
forge index build --from DIR --out index.json \
  --revision datasets/Mitroshenkov87/voxprint-model-index=<commit>
forge index push --index index.json --push     # signs and uploads, next serial
```

Without the pin, `index build` asks the Hub for the last commit that touches
`descriptors/<id>/voxprint-model.json`.

## Schema

`schemas/voxprint-model.schema.json` is JSON Schema draft 2020-12. Tests
validate built documents against it. Publish also checks the same required
shape in process, so a bad document is not uploaded when the schema package
is absent.
