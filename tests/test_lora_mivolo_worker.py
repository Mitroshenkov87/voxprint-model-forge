"""The MiVOLO estimator talks JSON lines to a worker in another Python env."""

from __future__ import annotations

import sys

import pytest
from PIL import Image

from voxprint_forge.lora.backends import MivoloEstimator

FAKE = """#!{python}
import json, sys
print(json.dumps({{"ready": True}}), flush=True)
for n, line in enumerate(sys.stdin):
    json.loads(line)
    print(json.dumps({{"n_faces": 1, "ages": [31.5]}} if n == 0 else {{"error": "boom"}}), flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="uses a shebang script as the worker interpreter")
def test_worker_protocol(tmp_path) -> None:
    fake = tmp_path / "fake-python"
    fake.write_text(FAKE.format(python=sys.executable))
    fake.chmod(0o755)
    estimator = MivoloEstimator(tmp_path, tmp_path / "yunet.onnx", "cpu", python=str(fake))
    image = Image.new("RGB", (32, 32))
    reading = estimator.estimate(image)
    assert reading.n_faces == 1 and reading.ages == (31.5,)
    with pytest.raises(RuntimeError):
        estimator.estimate(image)
    estimator.unload()
    assert not estimator.tmp.exists()
