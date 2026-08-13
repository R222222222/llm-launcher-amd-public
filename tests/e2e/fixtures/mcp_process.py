"""Small, network-free MCP process used by the MCP E2E scenario.

The launcher only supervises this process; it does not need to speak MCP for
this test.  The first line is a deliberately stable, PID-bearing sentinel so
the scenario can prove that stdout capture and ``exec`` PID propagation work.
The normal mode never reads stdin (the backend uses ``DEVNULL``); it waits for
SIGTERM/SIGINT.  With ``--exit-code 7`` the sentinel is printed and the
process exits immediately with status 7, exercising the runner's auto-stop
path.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import re


READY_SENTINEL = "LLM_LAUNCHER_E2E_MCP_READY"
READY_SENTINEL_RE = re.compile(r"^LLM_LAUNCHER_E2E_MCP_READY pid=(\d+)$")


def ready_line(pid: int | None = None) -> str:
    """Return the stable stdout sentinel carrying the process PID."""
    return f"{READY_SENTINEL} pid={os.getpid() if pid is None else pid}"


def parse_ready_pid(line: str) -> int:
    """Parse a fixture sentinel, rejecting truncated/non-fixture output."""
    match = READY_SENTINEL_RE.fullmatch(line.strip())
    if match is None:
        raise ValueError(f"invalid MCP fixture sentinel: {line!r}")
    return int(match.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="network-free E2E MCP stdio fixture")
    parser.add_argument("--exit-code", type=int, default=None)
    args = parser.parse_args(argv)

    stop = threading.Event()

    def handle_signal(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    print(ready_line(), flush=True)
    if args.exit_code is not None:
        return args.exit_code

    # The backend deliberately launches children with stdin=DEVNULL.  Waiting
    # on an Event, rather than reading stdin, is deterministic in that mode.
    while not stop.wait(0.1):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
