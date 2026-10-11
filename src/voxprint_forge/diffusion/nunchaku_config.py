"""Make Nunchaku honor the Qwen checkpoint's layer count.

``NunchakuQwenImageTransformer2DModel.__init__`` is ``*args, **kwargs``.
Diffusers reads that signature, decides no config key is expected, and builds
the default 60 blocks. A one-block checkpoint then dies in ``_patch_model``
with missing ``transformer_blocks.1.*`` keys. Binding the parent signature
makes ``from_config`` pass ``num_layers`` through before ``_patch_model``.
"""

from __future__ import annotations

import inspect


def bind_qwen_config_signature(cls: type, parent: type) -> None:
    """Replace ``cls.__init__`` so ``from_config`` keeps the parent's keys.

    ``offload`` stays a nunchaku-only argument and is not forwarded. The
    parent constructor is what actually builds ``transformer_blocks``.
    """
    parent_init = parent.__init__
    signature = inspect.signature(parent_init)

    def init(self: object, *args: object, **kwargs: object) -> None:
        self.offload = bool(kwargs.pop("offload", False))
        self.offload_manager = None
        self._is_initialized = False
        parent_init(self, *args, **kwargs)

    init.__signature__ = signature  # type: ignore[attr-defined]
    cls.__init__ = init  # type: ignore[method-assign]


def layers_agree(checkpoint_layers: int, model_layers: int) -> bool:
    """True when the checkpoint and the built model use the same depth."""
    return checkpoint_layers == model_layers
