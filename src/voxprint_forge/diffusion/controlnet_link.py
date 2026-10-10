"""Link a swapped ControlNet to the pipeline transformer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class _Movable(Protocol):
    """Object that can be moved onto a device."""

    def to(self, device: object) -> object:
        """Return this module on ``device``."""
        ...


class _ControlPipe(Protocol):
    """Pipeline that holds a transformer and a controlnet slot."""

    transformer: object
    controlnet: object


def swap_controlnet(
    pipe: _ControlPipe,
    net: object,
    device: object,
    link: Callable[[object, object], _Movable],
) -> object:
    """Assign ``net`` after linking it to ``pipe.transformer``.

    Swapped nets must share the transformer embedders like the pipeline
    constructor does. ``link`` is ``ZImageControlNetModel.from_transformer``.
    Diffusers raises ``Required modules are None, use from_transformer`` when
    a net loaded on its own is assigned without that link. The BF16 net is
    passed to ``from_pretrained`` and is not swapped through here.
    """
    linked = link(net, pipe.transformer)
    placed = linked.to(device)
    pipe.controlnet = placed
    return placed
