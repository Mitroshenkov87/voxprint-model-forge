"""Release CUDA memory between quality-eval stages."""

from __future__ import annotations

import gc

import torch


def free_eval_stage(pipe: object | None = None) -> None:
    """Drop ``pipe`` and flush CUDA before the next eval stage.

    Rebind the caller: ``pipe = free_eval_stage(pipe)``. ``del`` on the
    parameter only drops this function's binding. Returning ``None`` is what
    releases the caller's name.

    A call with no pipe still runs ``gc.collect``, ``empty_cache``, and
    ``synchronize``. Part C does that between the BF16, transformer-FP8, and
    encoder-FP8 passes, which reuse one pipeline. Freeing the reference
    pipeline before the next stage is a later change.
    """
    if pipe is not None:
        del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return None
