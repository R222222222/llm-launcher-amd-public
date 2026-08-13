"""Bounded in-memory replay for one launch session."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any


MAX_CONTROL_EVENTS = 256
MAX_STDOUT_EVENTS = 1_000
MAX_STDOUT_BYTES = 1 << 20
MAX_REPLAY_LINE = 16 << 10


@dataclass(frozen=True)
class EventRecord:
    seq: int
    event: dict[str, Any]


@dataclass(frozen=True)
class EventBatch:
    events: tuple[EventRecord, ...]
    history_gap: bool = False
    closed: bool = False


class ReplaySubscriber:
    def __init__(self, stream: "LaunchEventReplay", after: int):
        self._stream = stream
        self.cursor = after

    def wait_after(self, after: int | None = None, timeout: float | None = None) -> EventBatch:
        if after is not None:
            self.cursor = after
        batch = self._stream.wait_after(self.cursor, timeout)
        if batch.events:
            self.cursor = batch.events[-1].seq
        return batch


class LaunchEventReplay:
    """A strictly ordered stream with independent subscribers/cursors.

    Control and stdout records have separate bounds.  Evicting stdout never
    evicts terminal/control records merely because output is noisy.
    """
    def __init__(self):
        self._condition = threading.Condition()
        self._next_seq = 1
        self._records: dict[int, EventRecord] = {}
        self._control: deque[int] = deque()
        self._stdout: deque[int] = deque()
        self._stdout_bytes = 0
        self._closed = False

    @staticmethod
    def _event_copy(event: dict[str, Any]) -> dict[str, Any]:
        item = dict(event)
        if item.get("type") == "stdout":
            line = str(item.get("line", ""))
            encoded = line.encode("utf-8")[:MAX_REPLAY_LINE]
            item["line"] = encoded.decode("utf-8", errors="ignore")
        return item

    def publish(self, event: dict[str, Any]) -> EventRecord:
        with self._condition:
            if self._closed:
                # Publishing after close is a programmer error, but returning
                # the last record is safer than resurrecting a stream.
                if self._next_seq == 1:
                    raise RuntimeError("launch event replay is closed")
                return max(self._records.values(), key=lambda r: r.seq)
            record = EventRecord(self._next_seq, self._event_copy(event))
            self._next_seq += 1
            self._records[record.seq] = record
            if record.event.get("type") == "stdout":
                self._stdout.append(record.seq)
                self._stdout_bytes += len(record.event.get("line", "").encode("utf-8"))
                while len(self._stdout) > MAX_STDOUT_EVENTS or self._stdout_bytes > MAX_STDOUT_BYTES:
                    old = self._records.pop(self._stdout.popleft(), None)
                    if old is not None:
                        self._stdout_bytes -= len(old.event.get("line", "").encode("utf-8"))
            else:
                self._control.append(record.seq)
                while len(self._control) > MAX_CONTROL_EVENTS:
                    self._records.pop(self._control.popleft(), None)
            self._condition.notify_all()
            return record

    def _ordered_after(self, after: int) -> tuple[EventRecord, ...]:
        return tuple(self._records[seq] for seq in sorted(self._records) if seq > after)

    def history_gap(self, after: int) -> bool:
        with self._condition:
            return self._history_gap_locked(after)

    def _history_gap_locked(self, after: int) -> bool:
        # stdout and control records are retained independently, so merging
        # their surviving records can leave holes in either position.  Check
        # the complete ordered sequence, not just its first record.
        expected = after + 1
        for seq in sorted(self._records):
            if seq <= after:
                continue
            if seq != expected:
                return True
            expected += 1
        return False

    def wait_after(self, after: int, timeout: float | None = None) -> EventBatch:
        if not isinstance(after, int) or isinstance(after, bool) or after < 0:
            raise ValueError("cursor inválido")
        with self._condition:
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                events = self._ordered_after(after)
                if events or self._closed:
                    return EventBatch(events, self._history_gap_locked(after), self._closed)
                if timeout == 0:
                    return EventBatch((), False, self._closed)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return EventBatch((), False, self._closed)
                self._condition.wait(remaining)

    def subscribe(self, after: int = 0) -> ReplaySubscriber:
        if not isinstance(after, int) or isinstance(after, bool) or after < 0:
            raise ValueError("cursor inválido")
        return ReplaySubscriber(self, after)

    def close(self) -> bool:
        with self._condition:
            if self._closed:
                return False
            self._closed = True
            self._condition.notify_all()
            return True

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed


# Short aliases keep the small primitive convenient for focused backend tests.
EventReplay = LaunchEventReplay
LaunchEvents = LaunchEventReplay
