"""Small OS measurements. These do not estimate accelerator energy or watts."""

import ctypes
import subprocess
import sys


def memory_usage() -> dict[str, int]:
    """Windows working set and cumulative process peak; not per-model allocation."""
    if sys.platform != "win32":
        return {}

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return {
            "working_set_bytes": counters.WorkingSetSize,
            "peak_process_working_set_bytes": counters.PeakWorkingSetSize,
        }
    return {}


def power_state() -> dict[str, str | int | None]:
    if sys.platform != "win32":
        return {"ac_line": "unknown", "active_power_scheme": "unknown", "battery_percent": None}

    class Status(ctypes.Structure):
        _fields_ = [
            ("ac", ctypes.c_ubyte),
            ("flags", ctypes.c_ubyte),
            ("percent", ctypes.c_ubyte),
            ("reserved", ctypes.c_ubyte),
            ("life", ctypes.c_ulong),
            ("full_life", ctypes.c_ulong),
        ]

    status = Status()
    ac = "unknown"
    known = bool(ctypes.WinDLL("kernel32").GetSystemPowerStatus(ctypes.byref(status)))
    if known:
        ac = {0: "battery", 1: "plugged in"}.get(status.ac, "unknown")
    try:
        completed = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        scheme = completed.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        scheme = "unknown"
    return {
        "ac_line": ac,
        "active_power_scheme": scheme,
        "battery_percent": status.percent if known and status.percent != 255 else None,
    }
