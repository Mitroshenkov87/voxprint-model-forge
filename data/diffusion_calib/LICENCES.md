# Calibration set licences

Checked 2026-10-11. These files are inputs to SVDQuant. They are not copied into the model weights.

| Item | Licence | Notes |
| --- | --- | --- |
| `prompts.json` | Apache-2.0 | Original Russian and English instructions written for Voxprint. They cover edits, people, text, and hands. They are not COCO Captions and not MJHQ. |
| `images/*.png` | Apache-2.0 | Original 16×16 synthetic swatches generated for this set. They are not photographs and not pictures of real people. |
| Qwen-Image-Edit-2511 and Qwen-Image weights | Apache-2.0 | Upstream models. Public, not gated. |
| QuantFunc INT4 comparison weights | Apache-2.0 | Third-party SVDQuant exports used only as eval arms. |
| deepcompressor | Apache-2.0 | Pinned commit `69f3473f5e1c1504bae35cc50c7858ef900a9b17`. The `LICENSE` file is the Apache-2.0 text, copyright 2024 Yujun Lin, Muyang Li, Zhekai Zhang, Haotian Tang, Shang Yang, and Song Han. |
| city96 ComfyUI-GGUF | Apache-2.0 | Pinned commit `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`. |
| llama.cpp tag `b3962` | MIT | Commit `c8c07d658a6cefc5a50cfdf6be7d726503612303`. The city96 patch targets this tag. |
| Wan2.2 code and Wan2.2-T2V-A14B weights | Apache-2.0 | Code commit `1ea34ff48f87168174e12956e200b1d908b1c5ff`. Weights revision `c8c270b13ee05bfa474194ac9fb07a5868a97cea`. |
| tonera vitoom-nunchaku | Apache-2.0 | Wheel commit `85bad5510d409570527b018387491fd0c81e4727`. Used for Qwen-Image-Edit-2511 inference because `zero_cond_t` is true. |

The English identity, hands, and text lines are the canvas hard cases. The Russian lines are the same edits for calibration. A rental can point `FORGE_CALIB_IMAGE_DIR` at clothed-adult canvas inputs later; the committed pixels stay the default refs.
