"""Tokens and the remote SSH choice live in the OS keyring, never in the repo."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

try:
    import keyring
except ImportError:
    keyring = None

SERVICE = "voxprint-forge"
TOKEN_NAMES = ("HF_TOKEN", "KAGGLE_API_TOKEN", "FORGE_DRIVE_TOKEN")
REMOTE_NAMES = ("ssh_host", "ssh_user", "ssh_key")


class SecretStore(Protocol):
    def get(self, name: str) -> str: ...

    def set(self, name: str, value: str) -> None: ...


class MemoryStore:
    """In-memory stand-in used by tests."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, name: str) -> str:
        return self.data.get(name, "")

    def set(self, name: str, value: str) -> None:
        self.data[name] = value


class KeyringStore:
    def get(self, name: str) -> str:
        if keyring is None:
            return ""
        found = keyring.get_password(SERVICE, name)
        return found or ""

    def set(self, name: str, value: str) -> None:
        if keyring is None:
            raise RuntimeError("keyring is not installed")
        keyring.set_password(SERVICE, name, value)


class FileStore:
    """Mode-0600 file outside the repo, used only when the keyring backend is missing."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, name: str) -> str:
        return self._load().get(name, "")

    def set(self, name: str, value: str) -> None:
        data = self._load()
        data[name] = value
        self._save(data)

    def _load(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return {}
        return {str(key): str(value) for key, value in loaded.items()}

    def _save(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(self.path, 0o600)


def user_config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA", "")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / "voxprint-forge"
    base = os.environ.get("XDG_CONFIG_HOME", "")
    root = Path(base) if base else Path.home() / ".config"
    return root / "voxprint-forge"


def open_store() -> SecretStore:
    if keyring is None:
        return FileStore(user_config_dir() / "secrets.json")
    try:
        keyring.get_keyring()
    except Exception:
        return FileStore(user_config_dir() / "secrets.json")
    return KeyringStore()


def token_env(store: SecretStore, targets: tuple[str, ...]) -> dict[str, str]:
    """Environment for a forge child. Tokens come from the store and are not written to disk."""
    env: dict[str, str] = {"FORGE_UPLOAD_TARGETS": ",".join(targets)}
    for name in TOKEN_NAMES:
        value = store.get(name).strip()
        if value:
            env[name] = value
    return env
