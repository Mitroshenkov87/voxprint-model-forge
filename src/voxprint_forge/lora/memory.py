"""Graphics-memory plan for a LoRA run: projected peak per phase, checked before any GPU work.

The plan is arithmetic on the base's shapes and stored sizes, so it costs nothing and runs in
``--dry-run`` too. :func:`check_plan` refuses a run whose projected peak does not fit the budget
(``FORGE_VRAM_CAP_GB`` or the card), and :func:`check_residual` stops a run whose encoders did not
really leave the card before the transformer moves there (CALIB-R2/R3: the text encoder stayed
pinned by an autograd graph and the transformer OOMed on top of it).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from voxprint_forge.lora.errors import LoraStageError

GIB = 2**30
# Allocator slack: rounding to 2 MiB blocks, fragmentation and short-lived buffers. Taken as a share
# of the planned bytes plus a fixed floor, so the plan errs on the high side.
SLACK_SHARE = 0.10
SLACK_FLOOR_GB = 0.5
# Budget headroom under the cap (or the card): what the cap must keep free above the projected peak.
HEADROOM_GB = 1.0
# What may stay on the card after the encoders leave: cached latents and embeddings (a few MB each)
# plus this allowance. More than that means an encoder is still pinned.
RESIDUAL_ALLOWANCE_GB = 0.5


@dataclass(frozen=True)
class MemoryPlan:
    """Projected GiB per phase. ``peak_gb`` is the larger of the encode and the train phase."""

    encoders_gb: float
    encode_activations_gb: float
    transformer_gb: float
    lora_gb: float
    activations_gb: float
    samples_gb: float

    @property
    def encode_gb(self) -> float:
        """Text encoder and VAE on the card while latents and embeddings are computed."""
        return _slack(self.encoders_gb + self.encode_activations_gb)

    @property
    def train_gb(self) -> float:
        """Transformer, LoRA weights, grads and AdamW state, one micro-batch of activations."""
        return _slack(self.transformer_gb + self.lora_gb + self.activations_gb + self.samples_gb)

    @property
    def peak_gb(self) -> float:
        """The projected peak of the run."""
        return max(self.encode_gb, self.train_gb)

    def to_dict(self) -> dict[str, float]:
        """Rounded figures for results and logs."""
        out = {key: round(value, 2) for key, value in asdict(self).items()}
        out.update(encode_gb=round(self.encode_gb, 2), train_gb=round(self.train_gb, 2), peak_gb=round(self.peak_gb, 2))
        return out


def _slack(gb: float) -> float:
    return gb + max(SLACK_FLOOR_GB, SLACK_SHARE * gb)


def lora_params(blocks: int, shapes: tuple[tuple[int, int], ...], rank: int) -> int:
    """Adapter parameters: ``rank * (in + out)`` for each wrapped linear in each block."""
    return blocks * sum(rank * (fan_in + fan_out) for fan_in, fan_out in shapes)


def lora_bytes(params: int, bytes_per: int) -> int:
    """Weights, gradients and the two AdamW moments, all in the adapter dtype."""
    return 4 * params * bytes_per


def budget_gb(cap_gb: float | None, card_gb: float | None) -> float | None:
    """What the run may plan for: the cap (or the card) minus :data:`HEADROOM_GB`. ``None`` without either."""
    have = cap_gb if cap_gb is not None else card_gb
    if have is None:
        return None
    if card_gb is not None:
        have = min(have, card_gb)
    return have - HEADROOM_GB


def check_plan(plan: MemoryPlan, cap_gb: float | None, card_gb: float | None) -> None:
    """Raise ``lora.err.vram`` before any GPU work when the projected peak exceeds the budget."""
    budget = budget_gb(cap_gb, card_gb)
    if budget is not None and plan.peak_gb > budget:
        have = cap_gb if cap_gb is not None else card_gb
        raise LoraStageError("lora.err.vram", have=int(have or 0), need=int(-(-(plan.peak_gb + HEADROOM_GB) // 1)))


def check_residual(residual_gb: float, samples_gb: float, plan: MemoryPlan | None, budget: float | None) -> None:
    """After the encoders leave, before the transformer moves: stop on a pinned encoder or a plan that no longer fits.

    ``residual_gb`` is ``torch.cuda.memory_allocated()`` at that moment. Raises ``lora.err.encoders_stuck``.
    """
    leaked = residual_gb - samples_gb
    over = plan is not None and budget is not None and residual_gb + plan.train_gb - plan.samples_gb > budget
    if leaked > RESIDUAL_ALLOWANCE_GB or over:
        raise LoraStageError("lora.err.encoders_stuck", gb=round(residual_gb, 1))


def vram_cap_gb() -> float | None:
    """``FORGE_VRAM_CAP_GB`` (GiB), a test cap that makes a big card behave like a small one; unset means none."""
    raw = os.environ.get("FORGE_VRAM_CAP_GB", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def card_gb() -> float | None:
    """Total memory of the first CUDA card in GiB, ``None`` without one."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / GIB
    except Exception:  # noqa: BLE001 - detection is best effort
        pass
    return None
