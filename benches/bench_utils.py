from __future__ import annotations

import csv
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tomllib
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT_DIR / "ai_bench_suite.yaml").read_text(encoding="utf-8"))
RESULT_SCHEMA_VERSION = 2


def emit(result: Dict[str, Any], args):
    """
    Stampa o salva il dizionario `result`.

    * --json         → pretty-printed su stdout
    * --append-csv   → crea/accoda la riga a path CSV

    Nota importante:
    - se il CSV esiste già, la funzione verifica che l'header sia identico,
      per evitare file con schema misto (molto scomodi da validare/ingestare).
    """
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))

    csv_path = getattr(args, "append_csv", None)
    if not csv_path:
        return

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current_fieldnames = list(result.keys())

    write_header = not path.exists()
    if not write_header:
        with path.open("r", newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != current_fieldnames:
            raise ValueError(
                "CSV schema mismatch for "
                f"{path}: existing header is {existing_header}, current row is {current_fieldnames}. "
                "Delete the CSV or use a file generated with the same schema version."
            )

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=current_fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def _run_text(cmd: list[str], timeout: int = 5) -> str:
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="ignore",
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    if cp.returncode != 0:
        return ""
    return (cp.stdout or cp.stderr or "").strip()



def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out



def _round_gib(num_bytes: int | float | None) -> int | None:
    if num_bytes is None:
        return None
    gib = float(num_bytes) / (1024 ** 3)
    if gib <= 0:
        return None
    return int(math.ceil(gib))


def _round_gib_from_mb(num_mb: int | float | None) -> int | None:
    if num_mb is None:
        return None
    gib = float(num_mb) / 1024.0
    if gib <= 0:
        return None
    return int(math.ceil(gib))



def _parse_size_to_gib(value: str | None) -> float | None:
    if not value:
        return None

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(TB|GB|MB)", value.strip(), re.IGNORECASE)
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2).upper()

    if unit == "TB":
        return round(amount * 1024.0, 1)
    if unit == "GB":
        return round(amount, 1)
    if unit == "MB":
        return round(amount / 1024.0, 1)
    return None


@lru_cache(maxsize=1)
def get_benchmark_version() -> str | None:
    pyproject = ROOT_DIR / "pyproject.toml"
    try:
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        version = data.get("project", {}).get("version")
        return str(version).strip() if version else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_cuda_toolkit_version() -> str | None:
    out = _run_text(["nvcc", "--version"], timeout=5)
    if not out:
        return None

    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)*)", out, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"V([0-9]+(?:\.[0-9]+)+)", out)
    if match:
        return match.group(1)

    return None


@lru_cache(maxsize=1)
def get_nvidia_driver_version() -> str | None:
    out = _run_text(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        timeout=5,
    )
    if not out:
        return None

    values = _dedupe_preserve_order(line.strip() for line in out.splitlines())
    if not values:
        return None
    return " | ".join(values)



def _gpus_from_gputil() -> list[dict[str, Any]]:
    try:
        import GPUtil  # type: ignore

        gpus = GPUtil.getGPUs()
        out: list[dict[str, Any]] = []
        for gpu in gpus:
            out.append(
                {
                    "name": (gpu.name or "").strip(),
                    "vram_gb": _round_gib_from_mb(getattr(gpu, "memoryTotal", None)),
                }
            )
        return [gpu for gpu in out if gpu["name"]]
    except Exception:
        return []



def _gpus_from_nvidia_smi() -> list[dict[str, Any]]:
    out = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if not out:
        return []

    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        name, memory_mb = parts
        try:
            memory_mb_num = float(memory_mb)
        except ValueError:
            memory_mb_num = None
        gpus.append(
            {
                "name": name,
                "vram_gb": _round_gib_from_mb(memory_mb_num),
            }
        )
    return gpus



def _gpus_from_windows() -> list[dict[str, Any]]:
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
    )
    out = _run_text(["powershell", "-NoProfile", "-Command", script], timeout=10)
    if not out:
        return []

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        data = [data]

    gpus: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "").strip()
        adapter_ram = item.get("AdapterRAM")
        try:
            adapter_ram = int(adapter_ram) if adapter_ram is not None else None
        except (TypeError, ValueError):
            adapter_ram = None
        if name:
            gpus.append({"name": name, "vram_gb": _round_gib(adapter_ram)})
    return gpus



def _gpus_from_linux() -> list[dict[str, Any]]:
    out = _run_text(["sh", "-c", "lspci | grep -i 'vga\\|3d\\|display'"], timeout=5)
    if not out:
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        name = line.split(": ", 1)[-1].strip()
        if name:
            gpus.append({"name": name, "vram_gb": None})
    return gpus



def _gpus_from_macos() -> list[dict[str, Any]]:
    out = _run_text(["system_profiler", "SPDisplaysDataType", "-json"], timeout=15)
    if not out:
        return []

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []

    items = data.get("SPDisplaysDataType", [])
    gpus: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        name = (
            item.get("sppci_model")
            or item.get("_name")
            or item.get("spdisplays_vendor")
            or ""
        )
        vram_raw = (
            item.get("spdisplays_vram")
            or item.get("spdisplays_vram_shared")
            or item.get("sppci_vram")
        )
        name = str(name).strip()
        if name:
            gpus.append({"name": name, "vram_gb": _parse_size_to_gib(str(vram_raw))})

    return gpus


@lru_cache(maxsize=1)
def get_gpu_inventory() -> list[dict[str, Any]]:
    gpus = _gpus_from_gputil()
    if gpus:
        return gpus

    gpus = _gpus_from_nvidia_smi()
    if gpus:
        return gpus

    system = platform.system()
    if system == "Windows":
        return _gpus_from_windows()
    if system == "Darwin":
        return _gpus_from_macos()
    if system == "Linux":
        return _gpus_from_linux()
    return []


@lru_cache(maxsize=1)
def get_system_ram_gb() -> float | None:
    system = platform.system()

    try:
        if system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return _round_gib(status.ullTotalPhys)

        if system == "Darwin":
            out = _run_text(["sysctl", "-n", "hw.memsize"], timeout=5)
            return _round_gib(int(out)) if out else None

        if hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            return _round_gib(page_size * phys_pages)
    except Exception:
        return None

    return None


@lru_cache(maxsize=1)
def sysinfo() -> Dict[str, Any]:
    machine = CFG["machine_info"]["machine"] or socket.gethostname() or "Not recognized"

    try:
        import cpuinfo  # type: ignore

        cpu_fallback = cpuinfo.get_cpu_info().get("brand_raw")
    except Exception:
        cpu_fallback = platform.processor()

    gpu_inventory = get_gpu_inventory()
    gpu_names = [gpu.get("name") for gpu in gpu_inventory if gpu.get("name")]
    gpu_fallback = " | ".join(gpu_names) if gpu_names else "Not recognized"

    gpu_vram_values = [
        gpu.get("vram_gb")
        for gpu in gpu_inventory
        if gpu.get("vram_gb") is not None
    ]

    return {
        "machine": machine,
        "CPU": CFG["machine_info"]["cpu"] or cpu_fallback or "Not recognized",
        "GPU": CFG["machine_info"]["gpu"] or gpu_fallback,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "benchmark_version": get_benchmark_version(),
        "schema_version": RESULT_SCHEMA_VERSION,
        "cuda_toolkit_version": get_cuda_toolkit_version(),
        "nvidia_driver_version": get_nvidia_driver_version(),
        "gpu_count": len(gpu_inventory),
        "gpu_vram_gb": max(gpu_vram_values) if gpu_vram_values else None,
        "system_ram_gb": get_system_ram_gb(),
    }
