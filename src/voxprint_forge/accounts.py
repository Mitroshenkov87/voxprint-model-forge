"""Connected accounts for Hugging Face, Kaggle, and Google Drive.

Tokens live in the OS keyring. This module does not write a token file and does
not put a token in its return values.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Never

from huggingface_hub import HfApi
from huggingface_hub.utils._oauth_device import poll_device_token, request_device_code

from voxprint_forge.errors import UsageError
from voxprint_forge.keystore import SecretStore, open_store
from voxprint_forge.scrub import scrub
from voxprint_forge.upload import HubRemote

AccountTarget = str
_TARGETS = ("hf", "kaggle", "drive")
_TOKEN_KEY = {"hf": "HF_TOKEN", "kaggle": "KAGGLE_API_TOKEN", "drive": "FORGE_DRIVE_TOKEN"}
_NAME_KEY = {"hf": "account:hf:name", "kaggle": "account:kaggle:name", "drive": "account:drive:name"}
_REFRESH_KEY = "HF_REFRESH"

Probe = Callable[[str], str]
CodeHook = Callable[[str, str], None]


def _unexpected(target: str) -> Never:
    raise UsageError(f"unknown account target {target}", code="usage")


def _check_target(target: str) -> str:
    if target not in _TARGETS:
        _unexpected(target)
    return target


def list_accounts(store: SecretStore) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in _TARGETS:
        token = store.get(_TOKEN_KEY[target]).strip()
        name = store.get(_NAME_KEY[target]).strip()
        rows.append({"target": target, "name": name, "connected": bool(token)})
    return rows


def add_token(store: SecretStore, target: str, token: str) -> dict[str, object]:
    """Store a pasted token. Hugging Face and Kaggle use this; Drive accepts a rclone token JSON."""
    chosen = _check_target(target)
    value = token.strip()
    if not value:
        raise UsageError("token is empty", code="usage")
    if chosen == "drive":
        _drive_token_json(value)
    store.set(_TOKEN_KEY[chosen], value)
    return {"target": chosen, "name": store.get(_NAME_KEY[chosen]).strip(), "connected": True}


def remove_account(store: SecretStore, target: str) -> dict[str, object]:
    chosen = _check_target(target)
    store.delete(_TOKEN_KEY[chosen])
    store.delete(_NAME_KEY[chosen])
    if chosen == "hf":
        store.delete(_REFRESH_KEY)
    return {"target": chosen, "name": "", "connected": False}


def test_account(store: SecretStore, target: str, *, probes: dict[str, Probe] | None = None) -> dict[str, object]:
    chosen = _check_target(target)
    token = store.get(_TOKEN_KEY[chosen]).strip()
    if not token:
        raise UsageError(f"{chosen} is not connected", code="account")
    probe = (probes or default_probes())[chosen]
    name = probe(token)
    store.set(_NAME_KEY[chosen], name)
    return {"target": chosen, "name": name, "ok": True, "connected": True}


def login_browser(
    store: SecretStore,
    target: str,
    *,
    on_code: CodeHook | None = None,
    request: Callable[[], dict[str, object]] | None = None,
    poll: Callable[[dict[str, object]], dict[str, object]] | None = None,
    identify: Probe | None = None,
    authorize: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Browser login. Hugging Face uses the Hub device flow. Drive uses rclone authorize."""
    chosen = _check_target(target)
    if chosen == "hf":
        return _hf_device(store, on_code=on_code, request=request, poll=poll, identify=identify)
    if chosen == "drive":
        return _drive_browser(store, authorize=authorize, identify=identify)
    if chosen == "kaggle":
        raise UsageError("Kaggle login is a KGAT token paste, not a browser flow", code="usage")
    return _unexpected(chosen)


def _hf_device(
    store: SecretStore,
    *,
    on_code: CodeHook | None,
    request: Callable[[], dict[str, object]] | None,
    poll: Callable[[dict[str, object]], dict[str, object]] | None,
    identify: Probe | None,
) -> dict[str, object]:
    info = (request or _request_device)()
    uri = str(info.get("verification_uri_complete") or info.get("verification_uri") or "")
    code = str(info.get("user_code") or "")
    if on_code is not None:
        on_code(uri, code)
    response = (poll or _poll_device)(info)
    token = str(response.get("access_token") or "").strip()
    if not token:
        raise UsageError("Hugging Face device flow returned no access token", code="hf_login")
    refresh = str(response.get("refresh_token") or "").strip()
    name = (identify or _hf_name)(token)
    store.set("HF_TOKEN", token)
    if refresh:
        store.set(_REFRESH_KEY, refresh)
    store.set(_NAME_KEY["hf"], name)
    return {
        "target": "hf",
        "name": name,
        "connected": True,
        "verification_uri": uri,
        "user_code": code,
    }


def _request_device() -> dict[str, object]:
    return dict(request_device_code())


def _poll_device(info: dict[str, object]) -> dict[str, object]:
    return dict(poll_device_token(info))  # type: ignore[arg-type]


def _hf_name(token: str) -> str:
    try:
        who = HfApi(token=token).whoami()
    except Exception as exc:
        raise UsageError(scrub(f"Hugging Face login failed: {exc}"), code="hf_login") from None
    name = str(who.get("name", "")) if isinstance(who, dict) else ""
    if not name:
        raise UsageError("Hugging Face login failed", code="hf_login")
    return name


def _kaggle_name(token: str) -> str:
    return HubRemote(None, token).check_kaggle()


def _drive_name(token: str) -> str:
    """Ask rclone who the token is, using an in-memory remote. The config is not saved."""
    _drive_token_json(token)
    binary = shutil.which("rclone")
    if binary is None:
        return "drive"
    env = os.environ.copy()
    env["RCLONE_CONFIG_FORGE_TYPE"] = "drive"
    env["RCLONE_CONFIG_FORGE_SCOPE"] = "drive"
    env["RCLONE_CONFIG_FORGE_TOKEN"] = token
    env.pop("RCLONE_CONFIG", None)
    try:
        done = subprocess.run(
            [binary, "about", "forge:", "--json"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except OSError:
        return "drive"
    if done.returncode != 0:
        return "drive"
    try:
        payload = json.loads(done.stdout)
    except json.JSONDecodeError:
        return "drive"
    if isinstance(payload, dict):
        for key in ("user", "email", "name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "drive"


def default_probes() -> dict[str, Probe]:
    return {"hf": _hf_name, "kaggle": _kaggle_name, "drive": _drive_name}


def _drive_token_json(value: str) -> dict[str, object]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise UsageError("Drive token must be rclone token JSON", code="usage") from exc
    if not isinstance(payload, dict) or not str(payload.get("access_token") or "").strip():
        raise UsageError("Drive token JSON needs access_token", code="usage")
    return payload


def authorize_drive() -> str:
    """Run ``rclone authorize drive`` and return the token JSON. Nothing is written to a config."""
    binary = shutil.which("rclone")
    if binary is None:
        raise UsageError("rclone is not on PATH", code="drive_missing")
    with tempfile.TemporaryDirectory(prefix="forge-rclone-") as tmp:
        env = os.environ.copy()
        env["RCLONE_CONFIG"] = str(Path(tmp) / "rclone.conf")
        try:
            done = subprocess.run(
                [binary, "authorize", "drive"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        except OSError as exc:
            raise UsageError(f"rclone failed to start: {exc}", code="drive_missing") from exc
    text = f"{done.stdout}\n{done.stderr}"
    if done.returncode != 0:
        raise UsageError(scrub(text[-1500:]), code="drive_login")
    payload = _extract_json_object(text)
    return json.dumps(payload, separators=(",", ":"))


def _extract_json_object(text: str) -> dict[str, object]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise UsageError("rclone authorize did not print a token", code="drive_login")
    return _drive_token_json(text[start : end + 1])


def _drive_browser(
    store: SecretStore,
    *,
    authorize: Callable[[], str] | None,
    identify: Probe | None,
) -> dict[str, object]:
    token = (authorize or authorize_drive)()
    _drive_token_json(token)
    name = (identify or _drive_name)(token)
    store.set("FORGE_DRIVE_TOKEN", token)
    store.set(_NAME_KEY["drive"], name)
    return {"target": "drive", "name": name, "connected": True}


class LiveAccounts:
    """GUI adapter. The window calls these methods and does not touch the keyring itself."""

    def __init__(
        self,
        store: SecretStore,
        *,
        probes: dict[str, Probe] | None = None,
        request: Callable[[], dict[str, object]] | None = None,
        poll: Callable[[dict[str, object]], dict[str, object]] | None = None,
        authorize: Callable[[], str] | None = None,
    ) -> None:
        self.store = store
        self.probes = probes
        self.request = request
        self.poll = poll
        self.authorize = authorize

    def list_accounts(self) -> list[dict[str, object]]:
        return list_accounts(self.store)

    def add_token(self, target: str, token: str) -> dict[str, object]:
        return add_token(self.store, target, token)

    def remove(self, target: str) -> dict[str, object]:
        return remove_account(self.store, target)

    def test(self, target: str) -> dict[str, object]:
        return test_account(self.store, target, probes=self.probes)

    def login_browser(self, target: str, on_code: CodeHook | None = None) -> dict[str, object]:
        identify = None if self.probes is None else self.probes.get(target)
        return login_browser(
            self.store,
            target,
            on_code=on_code,
            request=self.request,
            poll=self.poll,
            identify=identify,
            authorize=self.authorize,
        )


def execute_accounts(
    args: object,
    *,
    store: SecretStore | None = None,
    probes: dict[str, Probe] | None = None,
) -> dict[str, object]:
    """Run ``forge accounts``. ``args`` is the argparse namespace."""
    chosen = store or open_store()
    command = str(getattr(args, "accounts_command", ""))
    target = str(getattr(args, "target", "") or "")
    if command == "list":
        return {"accounts": list_accounts(chosen)}
    if command == "add":
        login = str(getattr(args, "login", "paste") or "paste")
        if login == "browser":
            def _show(uri: str, code: str) -> None:
                print(f"Open {uri}\nEnter code {code}", file=sys.stderr, flush=True)

            return login_browser(chosen, target, on_code=_show)
        return add_token(chosen, target, str(getattr(args, "token", "") or ""))
    if command == "test":
        return test_account(chosen, target, probes=probes)
    if command == "remove":
        return remove_account(chosen, target)
    raise UsageError(f"unknown accounts command {command}", code="usage")

