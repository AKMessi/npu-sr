"""Small OS measurements. These do not estimate accelerator energy or watts."""

import ctypes
import re
import subprocess
import sys


def public_power_scheme(text: str) -> str:
    """Do not serialize a user's custom plan name or GUID into benchmarks."""
    names = {
        "381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced",
        "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
        "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver",
    }
    match = re.search(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", text.lower())
    return names.get(match.group(), "custom") if match else "unknown"


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
        scheme = public_power_scheme(completed.stdout)
    except (OSError, subprocess.SubprocessError):
        scheme = "unknown"
    return {
        "ac_line": ac,
        "active_power_scheme": scheme,
        "battery_percent": status.percent if known and status.percent != 255 else None,
        "energy_saver": bool(status.saver) if known else None,
    }
