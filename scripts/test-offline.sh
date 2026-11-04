#!/usr/bin/env bash
# Offline suite: loopback up, no default route. Normal pytest does not touch the network.
exec unshare -rn bash -c 'ip link set lo up 2>/dev/null || true; PYTHONPATH=src "${PYTHON:-python}" -m pytest -q -p no:cacheprovider "$@"' bash "$@"
