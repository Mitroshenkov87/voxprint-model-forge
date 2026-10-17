Vendored so bootstrap does not fetch GitHub.

- `voxprint_common-0.2.5-py3-none-any.whl`: the Yitro shell, https://github.com/Mitroshenkov87/voxprint-common/releases/tag/v0.2.5 (tag commit 8a94b531cb4f3526425a80a86a1e64624b5de606). CI, docs and release builds install it. The `gui` extra pins `voxprint-common[ui]==0.2.5`.
- `voxprint_common-0.1.1-py3-none-any.whl`: https://github.com/Mitroshenkov87/voxprint-common/releases/tag/v0.1.1. The VM bootstrap scripts still install it; the core pin `voxprint-common>=0.1.1,<0.3` accepts both.
