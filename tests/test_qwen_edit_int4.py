"""CPU proof for Qwen-Image-Edit INT4 convert and the QuantFunc smooth shim."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from voxprint_forge.diffusion.int4.patch import (
    _BUILD_ANCHOR,
    _COLLECT_SYH_ANCHOR,
    _FEATURES_ANCHOR,
    _TASK_ANCHOR,
    _W4_SCALE_ANCHOR,
    _YAML_ANCHOR,
    apply_checkout,
)
from voxprint_forge.diffusion.int4.pins import DEEPCOMPRESSOR_COMMIT, DEEPCOMPRESSOR_REPO
from voxprint_forge.diffusion.int4.quantfunc_smooth import (
    adapt_quantfunc_state,
    fetch_safetensors_header,
    preflight_missing_keys,
    quantfunc_header_url,
    quantfunc_smooth_plan,
)
from voxprint_forge.diffusion.int4.qwen_pack import (
    convert_to_nunchaku_w4x16_linear_weight,
    dequant_tinychat_w4,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "vm_run_diffusion_int4.sh"


def test_pin_is_the_apache_syh_commit() -> None:
    """The Qwen struct lives on lantudou syh, not on the 69f3473 base."""
    assert DEEPCOMPRESSOR_COMMIT == "019cbdeae7e2add52eaf4c27a361cfc1e57993b8"
    assert DEEPCOMPRESSOR_REPO == "https://github.com/lantudou/deepcompressor2"
    text = SCRIPT.read_text(encoding="utf-8")
    assert "lantudou/deepcompressor2" in text
    assert DEEPCOMPRESSOR_COMMIT in text
    assert "--smoke-tiny" in text


def test_issue_116_broadcasts_a_scalar_scale() -> None:
    """A [1, 1, 1, 1] img_mod scale converts when the output width splits by 6."""
    oc, ic = 192, 128
    weight = torch.linspace(-3, 3, oc * ic).reshape(oc, ic).to(dtype=torch.bfloat16)
    scale = torch.ones((1, 1, 1, 1), dtype=torch.bfloat16)
    packed, stored_scale, stored_zero, bias = convert_to_nunchaku_w4x16_linear_weight(
        weight, scale, adanorm_splits=6
    )
    assert packed.dtype == torch.int32
    assert list(packed.shape) == [oc // 4, ic // 2]
    assert stored_scale.shape == stored_zero.shape
    assert list(bias.shape) == [oc]
    assert torch.isfinite(stored_scale).all()
    assert torch.isfinite(bias).all()


def test_w4_unpack_and_preflight() -> None:
    """Unpack lands on the post-adanorm float, and the wheel preflight then passes."""
    oc, ic, group = 192, 128, 64
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn((oc, ic), generator=generator).clamp(-2, 2).to(dtype=torch.bfloat16)
    groups = ic // group
    viewed = weight.float().view(oc, groups, group)
    scale = (viewed.abs().amax(dim=-1).clamp_min(1e-4) / 6.0).view(oc, 1, groups, 1).to(dtype=torch.bfloat16)
    packed, stored_scale, stored_zero, bias = convert_to_nunchaku_w4x16_linear_weight(
        weight, scale, adanorm_splits=6
    )
    restored = dequant_tinychat_w4(packed, stored_scale, stored_zero, group_size=group)
    reordered = weight.float().view(6, oc // 6, ic).transpose(0, 1).reshape(oc, ic)
    step = float(scale.float().max())
    assert float((restored - reordered).abs().max()) <= step * 0.51 + 1e-2
    prefix = "transformer_blocks.0.img_mod.1"
    before = {
        f"{prefix}.qweight": packed,
        f"{prefix}.wscales": stored_scale,
        f"{prefix}.wzeros": stored_zero,
        f"{prefix}.bias": bias,
    }
    module_keys = [
        f"{prefix}.qweight",
        f"{prefix}.wscales",
        f"{prefix}.bias",
        f"{prefix}.smooth_factor",
        f"{prefix}.smooth_factor_orig",
    ]
    missing = preflight_missing_keys(module_keys, before, {prefix})
    assert f"{prefix}.smooth_factor" in missing
    adapted = adapt_quantfunc_state(dict(before))
    assert f"{prefix}.qweight" not in adapted
    assert list(adapted[f"{prefix}.weight"].shape) == [oc, ic]
    assert list(adapted[f"{prefix}.smooth_factor"].shape) == [ic]
    assert preflight_missing_keys(module_keys, adapted, {prefix}) == []


def _python(code: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a child so this process does not import diffusers."""
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_tiny_edit_chain_on_cpu() -> None:
    """Calib, rank-8 INT4, convert, and the Nunchaku key contract, without CUDA kernels."""
    proc = _python(
        "import json\n"
        "from pathlib import Path\n"
        "import torch\n"
        "from voxprint_forge.diffusion.int4.qwen_tiny import prove_tiny_edit\n"
        f"result = prove_tiny_edit(Path({str(REPO)!r}), device=torch.device('cpu'))\n"
        "print(json.dumps({k: result[k] for k in ('device', 'zero_cond_t', 'rank', 'packed_rank',"
        " 'cuda_kernels', 'output_shape', 'keys')}))\n"
        "shapes = result['shapes']\n"
        "assert 'transformer_blocks.0.img_mod.1.smooth_factor' not in shapes\n"
        "assert shapes['transformer_blocks.0.attn.to_qkv.proj_up'][-1] == 16\n"
        "assert shapes['transformer_blocks.0.img_mod.1.qweight'][0] > 0\n"
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["device"] == "cpu"
    assert result["zero_cond_t"] is True
    assert result["rank"] == 8
    assert result["packed_rank"] == 16
    assert result["cuda_kernels"] is False
    assert result["output_shape"] == [1, 8, 16]
    assert result["keys"] == 68


def test_smoke_tiny_writes_a_checkpoint_without_nunchaku(tmp_path: Path) -> None:
    """CPU smoke writes the tiny file and does not claim a nunchaku load."""
    dest = tmp_path / "tiny.safetensors"
    proc = _python(
        "import json\n"
        "from pathlib import Path\n"
        "import torch\n"
        "from voxprint_forge.diffusion.int4.qwen_tiny import smoke_tiny\n"
        f"result = smoke_tiny(Path({str(REPO)!r}), device=torch.device('cpu'), dest=Path({str(dest)!r}))\n"
        "print(json.dumps({'loaded': result['nunchaku_loaded'], 'image': result['image'],"
        " 'vae': result['vae_decode']}))\n"
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["loaded"] is False
    assert payload["image"] == "transformer latent"
    assert payload["vae"] is False
    assert dest.is_file()
    with safe_open(str(dest), framework="pt") as handle:
        keys = set(handle.keys())
        meta = handle.metadata() or {}
    assert "transformer_blocks.0.img_mod.1.qweight" in keys
    assert "transformer_blocks.0.img_mod.1.smooth_factor" not in keys
    config = json.loads(meta["config"])
    assert config["zero_cond_t"] is True
    assert json.loads(meta["quantization_config"])["rank"] == 16
    assert meta["voxprint_svd_rank"] == "8"


def test_syh_collect_and_scale_patch(tmp_path: Path) -> None:
    """The syh collect call and the issue 116 assert are patched, then left alone."""
    root = tmp_path / "deepcompressor"
    files = {
        root / "deepcompressor" / "app" / "diffusion" / "pipeline" / "config.py": _TASK_ANCHOR + _BUILD_ANCHOR,
        root / "deepcompressor" / "app" / "diffusion" / "dataset" / "data" / "__init__.py": _FEATURES_ANCHOR
        + _YAML_ANCHOR,
        root / "deepcompressor" / "app" / "diffusion" / "dataset" / "collect" / "calib.py": _COLLECT_SYH_ANCHOR,
        root / "deepcompressor" / "backend" / "nunchaku" / "utils.py": _W4_SCALE_ANCHOR,
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    licence = root / "LICENSE"
    licence.write_text("Apache License\nVersion 2.0\n", encoding="utf-8")
    first = apply_checkout(root)
    assert first["collect"] == "applied"
    assert first["convert"] == "applied"
    assert first["pipeline"] == "applied"
    assert 'task == "image-edit"' in Path(
        root / "deepcompressor" / "app" / "diffusion" / "dataset" / "collect" / "calib.py"
    ).read_text(encoding="utf-8")
    assert "target_seq_len" in Path(
        root / "deepcompressor" / "app" / "diffusion" / "dataset" / "collect" / "calib.py"
    ).read_text(encoding="utf-8")
    pipeline = (root / "deepcompressor" / "app" / "diffusion" / "pipeline" / "config.py").read_text(encoding="utf-8")
    assert "zero_cond_t=True" in pipeline
    utils = (root / "deepcompressor" / "backend" / "nunchaku" / "utils.py").read_text(encoding="utf-8")
    assert "issue 116" in utils
    assert "scale.expand" in utils
    second = apply_checkout(root)
    assert second["collect"] == "present"
    assert second["convert"] == "present"


def test_quantfunc_r128_header_plans_mod_smooth() -> None:
    """Range-read the QuantFunc r128 header and plan the missing mod smooth keys."""
    tensors = fetch_safetensors_header(quantfunc_header_url())
    mod = "transformer_blocks.0.img_mod.1"
    assert tensors[f"{mod}.qweight"] == {"dtype": "I32", "shape": [4608, 1536]}
    assert tensors[f"{mod}.wscales"]["shape"] == [48, 18432]
    assert f"{mod}.smooth_factor" not in tensors
    attention = [
        key
        for key, info in tensors.items()
        if key.endswith(".smooth_factor") and ".attn." in key and info["shape"] == [3072]
    ]
    assert attention
    plan = quantfunc_smooth_plan(tensors)
    assert plan[f"{mod}.smooth_factor"] == [3072]
    assert plan[f"{mod}.smooth_factor_orig"] == [3072]
    assert plan["transformer_blocks.0.txt_mod.1.smooth_factor"] == [3072]
    assert sum(1 for key in plan if key.endswith(".img_mod.1.smooth_factor")) == 60
    assert sum(1 for key in plan if key.endswith(".txt_mod.1.smooth_factor")) == 60
