"""Prove a Z-Image INT4 arm loaded the nunchaku kernels.

nunchaku-test-2 logged ``nunchaku: missing`` while a Z-Image arm still
returned images. A compare INT4 arm must fail closed instead.
"""

from __future__ import annotations

from pathlib import Path

from voxprint_forge.compare.codes import CompareSetup


def assert_int4_stack(version: str, class_module: str) -> dict[str, str]:
    """Accept a loaded ``NunchakuZImageTransformer2DModel`` and its package version.

    ``version`` is ``nunchaku.__version__``. ``class_module`` is the module of
    the loaded transformer class. Either one missing is exit 2.
    """
    if not version or version == "missing":
        raise CompareSetup("INT4 kernels did not load: nunchaku version is missing", code="compare_int4")
    expected = "nunchaku.models.transformers.transformer_zimage"
    if class_module != expected and not class_module.startswith(expected + "."):
        raise CompareSetup(
            "INT4 kernels did not load: NunchakuZImageTransformer2DModel is missing",
            code="compare_int4",
        )
    return {"nunchaku": version, "class_module": class_module}


def load_zimage_int4(path: Path) -> tuple[object, dict[str, str]]:
    """Load one INT4 transformer. Import failure is exit 2, not a silent fallback.

    ``nunchaku`` is imported here. It is an optional extra and must not load
    for the LLM commands.
    """
    try:
        import nunchaku
        from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel
    except ImportError as exc:
        raise CompareSetup("INT4 kernels did not load: nunchaku is missing", code="compare_int4") from exc
    import torch

    version = str(getattr(nunchaku, "__version__", "") or "")
    try:
        model = NunchakuZImageTransformer2DModel.from_pretrained(str(path), torch_dtype=torch.bfloat16)
    except Exception as exc:
        raise CompareSetup(f"INT4 kernels did not load: {type(exc).__name__}", code="compare_int4") from exc
    info = assert_int4_stack(version, type(model).__module__)
    return model, info
