"""The dataset gate: dirty does no work, the job is off the UI thread, cancel and deactivate."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from voxprint_forge.ui.check_gate import CancelToken, CheckGate, CheckItem, GateState, ItemResult  # noqa: E402


def _gate(job, items, *, unload=None, on_deactivate=None) -> CheckGate:
    return CheckGate(job, items=lambda: items, unload=unload, on_deactivate=on_deactivate)


def test_mark_dirty_does_not_run_the_job(qtbot) -> None:
    called: list[str] = []

    def job(item: CheckItem, _token: CancelToken) -> ItemResult:
        called.append(item.id)
        return ItemResult(item.id, True)

    gate = _gate(job, [CheckItem("a", "a")])
    qtbot.addWidget(gate)
    gate.mark_dirty()
    QtWidgets.QApplication.processEvents()
    assert called == []
    assert gate.state is GateState.DIRTY
    assert not gate.can_proceed


def test_the_job_and_unload_run_off_the_ui_thread(qtbot) -> None:
    threads: list[int] = []
    unloaded: list[int] = []

    def job(item: CheckItem, _token: CancelToken) -> ItemResult:
        threads.append(threading.get_ident())
        return ItemResult(item.id, True)

    def unload() -> None:
        unloaded.append(threading.get_ident())

    gate = _gate(job, [CheckItem("a", "a"), CheckItem("b", "b", active=False)], unload=unload)
    qtbot.addWidget(gate)
    with qtbot.waitSignal(gate.finished, timeout=5000):
        gate.start()
    assert threads == [threads[0]]
    assert threads[0] != threading.get_ident()
    assert unloaded == [threads[0]]
    assert gate.state is GateState.PASSED
    assert gate.can_proceed
    assert gate.deactivated == ()


def test_cancel_stops_after_the_current_item_and_unloads(qtbot) -> None:
    seen: list[str] = []
    unloaded: list[bool] = []

    def job(item: CheckItem, _token: CancelToken) -> ItemResult:
        seen.append(item.id)
        gate.cancel()
        return ItemResult(item.id, True)

    gate = _gate(job, [CheckItem("a", "a"), CheckItem("b", "b")], unload=lambda: unloaded.append(True))
    qtbot.addWidget(gate)
    with qtbot.waitSignal(gate.finished, timeout=5000):
        gate.start()
    assert seen == ["a"]
    assert unloaded == [True]
    assert gate.state is GateState.CANCELLED
    assert not gate.can_proceed


def test_deactivation_with_a_pass_ends_passed(qtbot) -> None:
    dropped: list[tuple[str, ...]] = []

    def job(item: CheckItem, _token: CancelToken) -> ItemResult:
        return ItemResult(item.id, item.id == "ok", "" if item.id == "ok" else "lora.drop.collage")

    gate = _gate(
        job,
        [CheckItem("ok", "ok"), CheckItem("bad", "bad")],
        on_deactivate=lambda ids: dropped.append(tuple(ids)),
    )
    qtbot.addWidget(gate)
    with qtbot.waitSignal(gate.finished, timeout=5000):
        gate.start()
    assert dropped == [("bad",)]
    assert gate.state is GateState.PASSED
    assert gate.can_proceed
    assert gate.deactivated == ("bad",)
    assert gate.findChild(QtWidgets.QLabel, "forgeCheckMark").text() == "✓"


def test_no_pass_ends_failed_even_after_deactivation(qtbot) -> None:
    gate = _gate(
        lambda item, _token: ItemResult(item.id, False, "lora.drop.blurry"),
        [CheckItem("a", "a"), CheckItem("b", "b")],
        on_deactivate=lambda _ids: None,
    )
    qtbot.addWidget(gate)
    with qtbot.waitSignal(gate.finished, timeout=5000):
        gate.start()
    assert gate.state is GateState.FAILED
    assert not gate.can_proceed
    assert gate.deactivated == ("a", "b")


def test_failed_when_deactivation_is_missing_or_raises(qtbot) -> None:
    def job(item: CheckItem, _token: CancelToken) -> ItemResult:
        return ItemResult(item.id, item.id == "ok")

    items = [CheckItem("ok", "ok"), CheckItem("bad", "bad")]
    missing = _gate(job, items, on_deactivate=None)
    qtbot.addWidget(missing)
    with qtbot.waitSignal(missing.finished, timeout=5000):
        missing.start()
    assert missing.state is GateState.FAILED
    assert missing.deactivated == ()

    def boom(_ids: tuple[str, ...]) -> None:
        raise RuntimeError("no")

    raised = _gate(job, items, on_deactivate=boom)
    qtbot.addWidget(raised)
    with qtbot.waitSignal(raised.finished, timeout=5000):
        raised.start()
    assert raised.state is GateState.FAILED
    assert not raised.can_proceed


def test_a_job_error_unloads_and_fails(qtbot) -> None:
    unloaded: list[bool] = []

    def job(_item: CheckItem, _token: CancelToken) -> ItemResult:
        raise RuntimeError("decode")

    gate = _gate(job, [CheckItem("a", "a")], unload=lambda: unloaded.append(True))
    qtbot.addWidget(gate)
    with qtbot.waitSignal(gate.finished, timeout=5000):
        gate.start()
    assert unloaded == [True]
    assert gate.state is GateState.FAILED


def test_mark_dirty_during_a_run_stays_dirty(qtbot) -> None:
    started = threading.Event()
    release = threading.Event()

    def job(item: CheckItem, _token: CancelToken) -> ItemResult:
        started.set()
        assert release.wait(2)
        return ItemResult(item.id, True)

    gate = _gate(job, [CheckItem("a", "a")])
    qtbot.addWidget(gate)
    gate.start()
    assert started.wait(2)
    worker = gate._worker
    gate.mark_dirty()
    release.set()
    qtbot.waitUntil(worker.isFinished, timeout=5000)
    assert gate.shutdown()
    QtWidgets.QApplication.processEvents()
    assert gate.state is GateState.DIRTY
    assert not gate.can_proceed


def test_check_pressed_while_a_stale_run_ends_starts_a_new_run(qtbot) -> None:
    started = threading.Event()
    release = threading.Event()
    seen: list[str] = []

    def job(item: CheckItem, _token: CancelToken) -> ItemResult:
        seen.append(item.id)
        started.set()
        assert release.wait(2)
        return ItemResult(item.id, True)

    gate = _gate(job, [CheckItem("a", "a")])
    qtbot.addWidget(gate)
    gate.start()
    assert started.wait(2)
    gate.mark_dirty()
    gate.start()
    assert gate.state is GateState.DIRTY
    release.set()
    qtbot.waitUntil(lambda: gate.state is GateState.PASSED, timeout=5000)
    assert seen == ["a", "a"]
    assert gate.shutdown()


def test_shutdown_cancels_and_joins_a_running_check(qtbot) -> None:
    started = threading.Event()
    unloaded: list[bool] = []

    def job(item: CheckItem, token: CancelToken) -> ItemResult:
        started.set()
        while not token.is_cancelled:
            threading.Event().wait(0.01)
        return ItemResult(item.id, True)

    gate = _gate(job, [CheckItem("a", "a"), CheckItem("b", "b")], unload=lambda: unloaded.append(True))
    qtbot.addWidget(gate)
    gate.start()
    assert started.wait(2)
    assert gate.shutdown(5000)
    assert not gate._worker.isRunning()
    assert unloaded == [True]
