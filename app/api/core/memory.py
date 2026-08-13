"""Helpers de memória RAM compartilhados pelo backend."""
from __future__ import annotations

import ctypes
import sys

_MIB = 1024 * 1024
_RAM_TOTAL_CACHE: int | None = None


def _ram_mib_pair() -> tuple[int | None, int | None]:
    """Devolve (total, disponível) em MiB sem depender da GPU."""
    global _RAM_TOTAL_CACHE
    try:
        if sys.platform == "win32":
            class _MEMSTAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength",                 ctypes.c_ulong),
                    ("dwMemoryLoad",             ctypes.c_ulong),
                    ("ullTotalPhys",             ctypes.c_ulonglong),
                    ("ullAvailPhys",             ctypes.c_ulonglong),
                    ("ullTotalPageFile",         ctypes.c_ulonglong),
                    ("ullAvailPageFile",         ctypes.c_ulonglong),
                    ("ullTotalVirtual",          ctypes.c_ulonglong),
                    ("ullAvailVirtual",          ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMSTAT()
            stat.dwLength = ctypes.sizeof(_MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total = int(stat.ullTotalPhys // _MIB)
            avail = int(stat.ullAvailPhys // _MIB)
            _RAM_TOTAL_CACHE = total
            return (total or None, avail or None)

        total: int | None = None
        avail: int | None = None
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                if total and avail:
                    break
        if total:
            _RAM_TOTAL_CACHE = total
        return (total, avail)
    except Exception:
        return (_RAM_TOTAL_CACHE or None, None)


def ram_total_mib() -> int | None:
    return _ram_mib_pair()[0]


def ram_avail_mib() -> int | None:
    return _ram_mib_pair()[1]
