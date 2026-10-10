"""OS keyring only. Tokens are never written to a file or printed."""

from __future__ import annotations

from typing import Protocol

from voxprint_forge.errors import UsageError

try:
    import keyring
except ImportError:
    keyring = None

SERVICE = "voxprint-forge"
TOKEN_NAMES = ("HF_TOKEN", "KAGGLE_API_TOKEN", "FORGE_DRIVE_TOKEN", "FORGE_DRIVE_SERVICE_ACCOUNT")
REMOTE_NAMES = ("ssh_host", "ssh_user", "ssh_key")


class SecretStore(Protocol):
    def get(self, name: str) -> str: ...

    def set(self, name: str, value: str) -> None: ...

    def delete(self, name: str) -> None: ...


class MemoryStore:
    """In-memory stand-in used by tests."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, name: str) -> str:
        return self.data.get(name, "")

    def set(self, name: str, value: str) -> None:
        self.data[name] = value

    def delete(self, name: str) -> None:
        self.data.pop(name, None)


class KeyringStore:
    """Windows Credential Manager or Linux Secret Service via the keyring package."""

    def get(self, name: str) -> str:
        if keyring is None:
            return ""
        found = keyring.get_password(SERVICE, name)
        return found or ""

    def set(self, name: str, value: str) -> None:
        if keyring is None:
            raise UsageError("keyring is not installed", code="keyring")
        keyring.set_password(SERVICE, name, value)

    def delete(self, name: str) -> None:
        if keyring is None:
            return
        try:
            keyring.delete_password(SERVICE, name)
        except Exception:
            return


def open_store() -> SecretStore:
    """Open the OS keyring. A missing backend is an error, not a secrets file."""
    if keyring is None:
        raise UsageError(
            "keyring is not installed. Secrets stay in the OS keyring and are not written to a file.",
            code="keyring",
        )
    try:
        keyring.get_keyring()
    except Exception as exc:
        raise UsageError(
            f"OS keyring is unavailable ({exc}). Secrets are not written to a file.",
            code="keyring",
        ) from None
    return KeyringStore()


def token_env(store: SecretStore, targets: tuple[str, ...]) -> dict[str, str]:
    """Environment for a forge child. Tokens come from the store and are not written to disk."""
    env: dict[str, str] = {"FORGE_UPLOAD_TARGETS": ",".join(targets)}
    for name in TOKEN_NAMES:
        value = store.get(name).strip()
        if value:
            env[name] = value
    return env
