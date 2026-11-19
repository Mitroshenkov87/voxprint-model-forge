"""Copy stdin to stdout with HF, Kaggle, and GitHub tokens removed.

The process reads those values from the environment. It does not print them.
"""

from __future__ import annotations

import os
import sys

NAMES = ("HF_TOKEN", "KAGGLE_API_TOKEN", "GH_TOKEN")


def main() -> None:
    secrets = [os.environ[name] for name in NAMES if len(os.environ.get(name, "")) >= 4]
    for line in sys.stdin:
        for secret in secrets:
            line = line.replace(secret, "***")
        sys.stdout.write(line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
