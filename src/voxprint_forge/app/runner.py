"""Run one ``voxprint-forge-cli lora`` command as a child process (Qt event loop)."""

from __future__ import annotations

from collections.abc import Callable

from voxprint_forge.app.engine import EngineEvent, LineReader, Outcome, cli_command

OnEvent = Callable[[EngineEvent], None]
OnDone = Callable[[Outcome], None]


class ProcessRunner:
    """One job at a time. Pause is the ``PAUSE`` file (``engine.request_pause``); ``stop`` ends the process."""

    def __init__(self) -> None:
        self._process: object | None = None
        self._reader = LineReader()

    @property
    def busy(self) -> bool:
        """True while a job runs."""
        return self._process is not None

    def start(self, args: list[str], on_event: OnEvent, on_done: OnDone) -> None:
        """Start ``voxprint-forge-cli <args>``; events and the outcome arrive on the Qt thread."""
        from PySide6 import QtCore

        if self._process is not None:
            raise RuntimeError("a job is already running")
        command = cli_command() + args
        process = QtCore.QProcess()
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.SeparateChannels)
        self._reader = LineReader()

        def read() -> None:
            chunk = bytes(process.readAllStandardOutput().data()).decode("utf-8", "replace")
            for event in self._reader.feed(chunk):
                on_event(event)

        def finished(code: int, _status: object) -> None:
            read()
            self._process = None
            on_done(self._reader.finish(code))

        process.readyReadStandardOutput.connect(read)
        process.finished.connect(finished)
        self._process = process
        process.start(command[0], command[1:])

    def stop(self) -> None:
        """Stop for good: end the process. Versions saved so far stay in the project."""
        process = self._process
        if process is None:
            return
        process.kill()  # type: ignore[attr-defined]
