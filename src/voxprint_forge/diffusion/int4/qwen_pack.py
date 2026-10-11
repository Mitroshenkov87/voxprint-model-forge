"""Nunchaku INT4 packers used by the Qwen-Image convert path.

The packing follows ``lantudou/deepcompressor2`` branch ``syh`` at
``019cbdeae7e2add52eaf4c27a361cfc1e57993b8`` (Apache-2.0). That tree is six
commits ahead of mit-han-lab ``69f3473`` and is the Qwen quant code the
nunchaku team had not published (nunchaku-ai/deepcompressor issue 104 is
still open). ``convert_to_nunchaku_w4x16_linear_weight`` broadcasts a
per-tensor scale before the ``scale.shape[0] == oc`` assert. That is the
failure in nunchaku-ai/deepcompressor issue 116, where ``img_mod`` /
adanorm scales arrive as ``[1, 1, 1, 1]``.
"""

from __future__ import annotations

import torch

GROUP_SIZE = 64
PACKED_RANK_TILE = 16


def ceil_divide(value: int, divisor: int) -> int:
    """Ceiling division for positive integers."""
    return (value + divisor - 1) // divisor


def pad(
    tensor: torch.Tensor,
    divisor: int | tuple[int, ...],
    dim: int | tuple[int, ...],
    fill_value: float = 0,
) -> torch.Tensor:
    """Pad ``tensor`` so each listed dimension is a multiple of ``divisor``."""
    if isinstance(divisor, int):
        divisors = (divisor,)
        dims: tuple[int, ...] = (dim,) if isinstance(dim, int) else tuple(dim)
        if len(dims) != 1 and not isinstance(dim, int):
            divisors = tuple(divisor for _ in dims)
    else:
        divisors = tuple(divisor)
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
    shape = list(tensor.shape)
    for axis, div in zip(dims, divisors, strict=True):
        if div > 1:
            shape[axis] = ceil_divide(shape[axis], div) * div
    if shape == list(tensor.shape):
        return tensor
    out = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    out[tuple(slice(0, extent) for extent in tensor.shape)] = tensor
    return out


def expand_w4_scale(scale: torch.Tensor, oc: int) -> torch.Tensor:
    """Broadcast a per-tensor W4 scale to one row per output channel.

    Issue 116 stores the ``img_mod`` scale as ``[1, 1, 1, 1]``. The convert
    assert requires ``scale.shape[0] == oc`` before the adanorm reorder.
    """
    if scale.ndim != 4:
        raise ValueError(f"scale tensor should be 4D, got {tuple(scale.shape)}")
    if scale.shape[0] == oc:
        return scale
    if int(scale.shape[0]) == 1:
        return scale.expand(oc, int(scale.shape[1]), int(scale.shape[2]), int(scale.shape[3])).contiguous()
    raise AssertionError(f"scale.shape[0]={scale.shape[0]} is not 1 or {oc}")


class NunchakuWeightPacker:
    """Pack a 4-bit matrix into the nunchaku MMA layout. ``warp_n`` is 128."""

    def __init__(self, bits: int = 4, warp_n: int = 128) -> None:
        self.bits = bits
        self.warp_n = warp_n
        self.comp_n = 16
        self.comp_k = 256 // bits
        self.insn_k = self.comp_k
        self.num_lanes = 32
        self.num_k_lanes = 4
        self.num_n_lanes = 8
        self.reg_k = 32 // bits
        self.reg_n = 1
        self.k_pack_size = self.comp_k // (self.num_k_lanes * self.reg_k)
        self.n_pack_size = self.comp_n // (self.num_n_lanes * self.reg_n)
        self.mem_k = self.comp_k
        self.mem_n = warp_n
        self.num_k_packs = 1
        self.num_n_packs = self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n)
        self.num_k_unrolls = 2

    def pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Pad ``[oc, ic]`` to the MMA tile."""
        return pad(weight, (self.mem_n, self.mem_k * self.num_k_unrolls), (0, 1))

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Pack int32 nibbles into an int8 ``[oc, ic / 2]`` matrix."""
        if weight.dtype != torch.int32:
            raise ValueError(f"quantized weight should be torch.int32, got {weight.dtype}")
        n, k = weight.shape
        n_tiles, k_tiles = n // self.mem_n, k // self.mem_k
        viewed = weight.reshape(
            n_tiles,
            self.num_n_packs,
            self.n_pack_size,
            self.num_n_lanes,
            self.reg_n,
            k_tiles,
            self.num_k_packs,
            self.k_pack_size,
            self.num_k_lanes,
            self.reg_k,
        )
        viewed = viewed.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
        viewed = viewed.bitwise_and(0xF)
        shift = torch.arange(0, 32, 4, dtype=torch.int32, device=viewed.device)
        packed = (viewed.bitwise_left_shift(shift)).sum(dim=-1, dtype=torch.int32)
        return packed.view(dtype=torch.int8).view(n, -1)

    def pad_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        """Pad a scale tensor to the warp tile."""
        if group_size > 0 and scale.numel() > scale.shape[0]:
            viewed = scale.view(scale.shape[0], 1, -1, 1)
            return pad(viewed, (self.warp_n, self.num_k_unrolls), (0, 2), fill_value=1)
        return pad(scale, self.warp_n, 0, fill_value=1)

    def pack_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        """Reorder scales. The returned shape is ``[groups, oc]`` or a vector."""
        n = scale.shape[0]
        s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
        num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
        num_s_packs = self.warp_n // (s_pack_size * num_s_lanes)
        viewed = scale.reshape(n // self.warp_n, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
        viewed = viewed.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return viewed.view(-1) if group_size == -1 else viewed.view(-1, n)

    def pad_lowrank_weight(self, weight: torch.Tensor, *, down: bool) -> torch.Tensor:
        """Pad the non-rank axis of a low-rank factor to ``warp_n``."""
        return pad(weight, self.warp_n, 1 if down else 0)

    def pack_lowrank_weight(self, weight: torch.Tensor, *, down: bool) -> torch.Tensor:
        """Pack a low-rank factor. Down is ``[rank, ic]``; up is ``[oc, rank]``.

        The factors stay bf16, so the K register is 2 (16-bit), not the 4-bit
        ``reg_k``. That tile is 16, which is why SVD rank 8 is stored as 16.
        """
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"low-rank weight should be fp16 or bf16, got {weight.dtype}")
        reg_n, reg_k = 1, 2
        pack_n = self.n_pack_size * self.num_n_lanes * reg_n
        pack_k = self.k_pack_size * self.num_k_lanes * reg_k
        weight = pad(weight, (pack_n, pack_k), (0, 1))
        if down:
            rank, ic = weight.shape
            r_packs, c_packs = rank // pack_n, ic // pack_k
            viewed = weight.view(r_packs, pack_n, c_packs, pack_k).permute(2, 0, 1, 3)
            channels, rank_out = ic, rank
        else:
            oc, rank = weight.shape
            c_packs, r_packs = oc // pack_n, rank // pack_k
            viewed = weight.view(c_packs, pack_n, r_packs, pack_k).permute(0, 2, 1, 3)
            channels, rank_out = oc, rank
        viewed = viewed.reshape(
            c_packs,
            r_packs,
            self.n_pack_size,
            self.num_n_lanes,
            reg_n,
            self.k_pack_size,
            self.num_k_lanes,
            reg_k,
        )
        viewed = viewed.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
        return viewed.view(channels, rank_out)


def ceil_num_groups(in_features: int, group_size: int) -> int:
    """TinyChat's padded group count for 4-bit weights."""
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} is not divisible by group {group_size}")
    num_groups = in_features // group_size
    pack_size = 8
    num_packs = ceil_divide(num_groups, pack_size)
    if group_size >= 128:
        factor = 1
    elif group_size == 64:
        factor = 2
    elif group_size == 32:
        factor = 4
    else:
        raise NotImplementedError(f"group size {group_size} is not a TinyChat W4 size")
    num_packs = ceil_divide(num_packs, factor) * factor
    return num_packs * pack_size


def pack_w4(weight: torch.Tensor) -> torch.Tensor:
    """Pack unsigned nibbles into the TinyChat int16 layout ``[oc / 4, ic]``."""
    oc, ic = weight.shape
    if ic % 32 != 0:
        raise ValueError(f"input channel size {ic} should be divisible by 32")
    viewed = weight.view(-1, 4, 8)
    folded = viewed[:, 0] | (viewed[:, 1] << 4) | (viewed[:, 2] << 8) | (viewed[:, 3] << 12)
    folded = folded.view(oc // 4, 4, ic // 64, 16).permute(0, 2, 1, 3).reshape(oc // 4, ic)
    return folded.to(torch.int16)


def unpack_w4(packed: torch.Tensor) -> torch.Tensor:
    """Invert ``pack_w4``. Accepts int16 ``[oc / 4, ic]`` or int32 ``[oc / 4, ic / 2]``."""
    if packed.dtype == torch.int32:
        packed = packed.view(torch.int16)
    if packed.dtype != torch.int16:
        raise ValueError(f"packed W4 weight should be int16 or int32, got {packed.dtype}")
    rows, ic = packed.shape
    oc = rows * 4
    restored = packed.view(rows, ic // 64, 4, 16).permute(0, 2, 1, 3).contiguous().reshape(-1, 8).to(torch.int32)
    nibbles = torch.stack(
        (
            restored & 0xF,
            (restored >> 4) & 0xF,
            (restored >> 8) & 0xF,
            (restored >> 12) & 0xF,
        ),
        dim=1,
    )
    return nibbles.reshape(oc, ic)


def convert_to_tinychat_w4x16y16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    *,
    zero_pre_scaled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a float matrix to TinyChat W4A16 and pack it."""
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("nunchaku W4A16 convert expects fp16 or bf16 weights")
    dtype, device = weight.dtype, weight.device
    weight_fp = weight.to(dtype=torch.float32)
    scale_fp = scale.to(dtype=torch.float32, device=device)
    zero_fp = zero.to(dtype=torch.float32, device=device)
    if zero_pre_scaled:
        zero_fp = zero_fp * scale_fp
    oc, ic = weight_fp.shape
    if scale_fp.numel() == 1:
        scale_fp = scale_fp.reshape(1, 1).expand(oc, 1)
        ng = 1
    else:
        ng = int(scale_fp.numel() // oc)
    gs = ic // ng
    if ic != gs * ng:
        raise ValueError(f"ic {ic} != group {gs} * groups {ng}")
    scale_fp = scale_fp.reshape(oc, ng).contiguous()
    if zero_fp.numel() == 1:
        zero_fp = zero_fp.reshape(1, 1).expand(oc, ng)
    zero_fp = zero_fp.reshape(oc, ng).contiguous()
    quantized = weight_fp.view(oc, ng, gs).add(zero_fp.unsqueeze(-1)).div(scale_fp.unsqueeze(-1)).round().view(oc, ic)
    if int(quantized.min()) < 0 or int(quantized.max()) > 15:
        raise ValueError("quantized W4A16 weight should be in [0, 15]")
    packed = pack_w4(quantized.to(torch.int32))
    padded_groups = ceil_num_groups(ic, gs)
    stored_scale = torch.zeros((padded_groups, oc), dtype=dtype, device=device)
    stored_zero = torch.zeros((padded_groups, oc), dtype=dtype, device=device)
    stored_scale[:ng] = scale_fp.t().to(dtype=dtype)
    stored_zero[:ng] = zero_fp.t().to(dtype=dtype).neg()
    return packed, stored_scale, stored_zero


def dequant_tinychat_w4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Invert TinyChat W4A16 packing back to the float matrix that was quantized.

    ``zero`` is the stored (negated, pre-scaled) zero. ``scale`` may have
    padded group rows; only ``in_features / group_size`` rows are real. The
    result is the matrix that was packed, including the adanorm reorder when
    that ran first.
    """
    nibbles = unpack_w4(packed).to(dtype=torch.float32)
    oc, ic = nibbles.shape
    if scale.ndim != 2 or zero.shape != scale.shape or int(scale.shape[1]) != oc:
        raise ValueError(f"scale {tuple(scale.shape)} does not match oc {oc}")
    if group_size < 1 or ic % group_size != 0:
        raise ValueError(f"ic {ic} is not divisible by group {group_size}")
    ng = ic // group_size
    if int(scale.shape[0]) < ng:
        raise ValueError(f"scale has {scale.shape[0]} rows and needs {ng}")
    scale_oc = scale[:ng].t().to(dtype=torch.float32)
    zero_oc = zero[:ng].t().to(dtype=torch.float32).neg()
    return (nibbles.view(oc, ng, group_size) * scale_oc.unsqueeze(-1) - zero_oc.unsqueeze(-1)).reshape(oc, ic)


def convert_to_nunchaku_w4x16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    adanorm_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one W4A16 matrix, including Qwen ``img_mod`` / ``txt_mod``.

    A scale whose leading dimension is 1 is broadcast to ``oc`` first. That is
    the issue 116 assert.
    """
    oc, ic = weight.shape
    scale = expand_w4_scale(scale, oc)
    if scale.shape[1] != 1 or scale.shape[3] != 1:
        raise AssertionError(f"scale axes 1 and 3 should be 1, got {tuple(scale.shape)}")
    ng = int(scale.shape[2])
    if bias is None:
        bias = torch.zeros((oc,), dtype=weight.dtype, device=weight.device)
    if oc % adanorm_splits != 0:
        raise AssertionError(f"output channels {oc} are not divisible by splits {adanorm_splits}")
    if adanorm_splits > 1:
        weight = weight.view(adanorm_splits, oc // adanorm_splits, ic).transpose(0, 1).reshape(oc, ic)
        scale = scale.view(adanorm_splits, oc // adanorm_splits, ng).transpose(0, 1).reshape(oc, 1, ng, 1)
        bias = bias.reshape(adanorm_splits, oc // adanorm_splits).transpose(0, 1)
        delta = [0] * adanorm_splits
        delta[1] = delta[-2] = 1
        bias = bias + torch.tensor(delta, dtype=bias.dtype, device=bias.device)
        bias = bias.reshape(oc)
    zero_point = torch.full_like(scale, 7) if zero is None else zero
    packed, stored_scale, stored_zero = convert_to_tinychat_w4x16y16_linear_weight(
        weight, scale, zero_point, zero_pre_scaled=True
    )
    return packed.view(torch.int32), stored_scale, stored_zero, bias


def convert_to_nunchaku_w4a4_linear(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    smooth: torch.Tensor | None,
    lora: tuple[torch.Tensor, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Pack one SVDQuant W4A4 linear, including its low-rank branch and smooth."""
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("nunchaku W4A4 convert expects fp16 or bf16 weights")
    if scale.ndim != 4:
        raise ValueError("W4A4 scale should be 4D")
    if int(scale.shape[2]) == 1:
        scale_key = "wcscales"
    else:
        scale_key = "wscales"
    oc, ic = weight.shape
    scale = expand_w4_scale(scale, oc)
    ng = int(scale.shape[2])
    gs = ic // ng
    if ic != gs * ng:
        raise ValueError(f"ic {ic} != group {gs} * groups {ng}")
    device, dtype = weight.device, weight.dtype
    # Scale is [oc, 1, groups, 1]. View the weight the same way so each group divides by its scale.
    quantized = (
        weight.to(dtype=torch.float32)
        .view(oc, 1, ng, gs)
        .div(scale.to(dtype=torch.float32))
        .round()
        .view(oc, ic)
    )
    if int(quantized.min()) < -8 or int(quantized.max()) > 7:
        raise ValueError("quantized W4A4 weight should be in [-8, 7]")
    quantized = quantized.view(oc, ic).to(dtype=torch.int32)
    if bias is None:
        bias = torch.zeros((oc, 1), dtype=dtype, device=device)
    else:
        bias = bias.view(-1, 1).to(dtype=dtype)
    if smooth is None:
        smooth = torch.ones((ic, 1), dtype=dtype, device=device)
    else:
        smooth = smooth.view(-1, 1).to(dtype=dtype)
    packer = NunchakuWeightPacker(bits=4)
    packed_weight = packer.pack_weight(packer.pad_weight(quantized))
    packed_scale = packer.pack_scale(packer.pad_scale(scale.to(dtype=dtype), gs), gs if gs < ic else -1)
    packed_bias = packer.pack_scale(packer.pad_scale(bias, -1), -1)
    packed_smooth = packer.pack_scale(packer.pad_scale(smooth, -1), -1)
    state: dict[str, torch.Tensor] = {
        "qweight": packed_weight,
        scale_key: packed_scale,
        "bias": packed_bias,
        "smooth_factor_orig": packed_smooth,
        "smooth_factor": packed_smooth.clone(),
    }
    if lora is not None:
        down, up = lora
        down = packer.pad_lowrank_weight(down.to(dtype=dtype), down=True)
        up = packer.pad_lowrank_weight(up.to(dtype=dtype), down=False)
        state["proj_down"] = packer.pack_lowrank_weight(down, down=True)
        state["proj_up"] = packer.pack_lowrank_weight(up, down=False)
    return state


def packed_lowrank_rank(rank: int) -> int:
    """Smallest nunchaku low-rank tile that can store ``rank`` (a multiple of 16)."""
    if rank < 1:
        raise ValueError(rank)
    return ceil_divide(rank, PACKED_RANK_TILE) * PACKED_RANK_TILE
