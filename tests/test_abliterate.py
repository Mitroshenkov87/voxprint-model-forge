"""Difference-of-means math on a tiny module. No checkpoint is loaded."""

from __future__ import annotations

import torch
from torch import nn

from voxprint_forge.abliterate import (
    apply_orthogonalization,
    difference_of_means,
    orthogonalize_weight,
    select_direction,
)


def test_orthogonalize_removes_the_direction() -> None:
    direction = torch.tensor([1.0, 0.0, 0.0])
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    updated = orthogonalize_weight(weight, direction)
    assert torch.allclose(updated[0], torch.zeros(2))
    assert torch.allclose(updated[1], weight[1])


def test_direction_and_apply() -> None:
    harmful = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    harmless = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    direction = difference_of_means(harmful, harmless)
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]))
    layer, chosen = select_direction({0: harmful, 1: harmless}, {0: harmless, 1: harmless})
    assert layer == 0
    assert torch.allclose(chosen, direction)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.o_proj.weight.copy_(torch.eye(2))

    block = Block()
    count = apply_orthogonalization(block, direction)
    assert count == 1
    assert torch.allclose(block.o_proj.weight[0], torch.zeros(2))
