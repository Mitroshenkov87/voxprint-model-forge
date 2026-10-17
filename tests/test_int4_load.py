"""The INT4 arm fails closed unless the nunchaku Z-Image class actually loaded."""

from __future__ import annotations

import pytest

from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.compare.int4 import assert_int4_stack

_MODULE = "nunchaku.models.transformers.transformer_zimage"


def test_a_recorded_nunchaku_version_is_accepted() -> None:
    info = assert_int4_stack("1.3.0", _MODULE)
    assert info == {"nunchaku": "1.3.0", "class_module": _MODULE}


def test_missing_version_or_class_is_exit_2() -> None:
    with pytest.raises(CompareSetup) as missing:
        assert_int4_stack("missing", _MODULE)
    assert missing.value.exit_code == 2
    assert missing.value.code == "compare_int4"
    with pytest.raises(CompareSetup) as wrong:
        assert_int4_stack("1.3.0", "diffusers.models.transformers.transformer_zimage")
    assert wrong.value.exit_code == 2
