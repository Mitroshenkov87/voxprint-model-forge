"""Union 2.1 ControlNet config for the diffusers single-file loader.

VideoX-Fun ``config/z_image/z_image_control_2.1_lite.yaml`` sets
``control_layers_places: [0, 10, 20]`` and ``control_refiner_layers_places: [0, 1]``.
The full 2.1 yaml uses every even index from 0 through 28 (15 blocks).
The architecture fields match the diffusers config published for Union 2.1
(``control_in_dim`` 33, ``dim`` 3840). A 3-layer file is the lite checkpoint.
"""

from __future__ import annotations

# config/z_image/z_image_control_2.1_lite.yaml
LITE_PLACES: tuple[int, ...] = (0, 10, 20)
# config/z_image/z_image_control_2.1.yaml
FULL_PLACES: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)
REFINER_PLACES: tuple[int, ...] = (0, 1)


def places_for_count(count: int) -> list[int]:
    """Layer indexes for a checkpoint that contains ``count`` control blocks."""
    if count <= 0:
        return []
    if count == len(LITE_PLACES):
        return list(LITE_PLACES)
    return list(FULL_PLACES[:count])


def union_config(places: list[int]) -> dict[str, object]:
    """Diffusers ``ZImageControlNetModel`` config with an explicit layer map."""
    return {
        "_class_name": "ZImageControlNetModel",
        "_diffusers_version": "0.36.0.dev0",
        "add_control_noise_refiner": "control_noise_refiner",
        "all_f_patch_size": [1],
        "all_patch_size": [2],
        "control_in_dim": 33,
        "control_layers_places": list(places),
        "control_refiner_layers_places": list(REFINER_PLACES),
        "dim": 3840,
        "n_heads": 30,
        "n_kv_heads": 30,
        "n_refiner_layers": 2,
        "norm_eps": 1e-05,
        "qk_norm": True,
    }
