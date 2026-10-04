"""Small OS measurements. These do not estimate accelerator energy or watts."""

import ctypes
import subprocess
import sys


def memory_usage(pid: int | None = None) -> dict[str, int | float]:
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
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = (
        kernel.OpenProcess(0x1000, False, pid) if pid is not None else kernel.GetCurrentProcess()
    )
    if not handle:
        return {}
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    result = {}
    try:
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            result.update(
                working_set_bytes=counters.WorkingSetSize,
                peak_process_working_set_bytes=counters.PeakWorkingSetSize,
            )
        times = [ctypes.c_ulonglong() for _ in range(4)]
        kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 4
        if kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
            result["cpu_seconds"] = (times[2].value + times[3].value) / 10_000_000
        return result
    finally:
        if pid is not None:
            kernel.CloseHandle(handle)


def power_state() -> dict[str, str | int | None]:
    if sys.platform != "win32":
        return {"ac_line": "unknown", "active_power_scheme": "unknown", "battery_percent": None}

    class Status(ctypes.Structure):
        _fields_ = [
            ("ac", ctypes.c_ubyte),
            ("flags", ctypes.c_ubyte),
            ("percent", ctypes.c_ubyte),
            ("saver", ctypes.c_ubyte),
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
        "energy_saver": bool(status.saver) if known else None,
    }
