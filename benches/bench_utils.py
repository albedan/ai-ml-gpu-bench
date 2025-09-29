# bench_utils.py
from __future__ import annotations
import yaml, json, csv, pathlib, subprocess, re
import platform, sys
from typing import Dict
import socket
import GPUtil

CFG = yaml.safe_load(open("ai_bench_suite.yaml"))

def emit(result: Dict, args):
    """
    Stampa o salva il dizionario `result`.

    * --json         → pretty-printed su stdout
    * --append-csv   → crea/accoda la riga a path CSV
    """
    # 1) dump JSON a schermo se richiesto
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))

    # 2) append su CSV se path passato
    csv_path = getattr(args, "append_csv", None)
    if csv_path:
        path = pathlib.Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(result)

def get_gpu_info():
    system = platform.system()

    # 1) GPUtil (priorità)
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            return [f"{g.name} ({int(g.memoryTotal)} MB)" for g in gpus]
    except Exception:
        pass

    # 2) Fallback singolo per OS
    try:
        if system == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                timeout=10
            ).decode(errors="ignore")
            return [ln.strip() for ln in out.splitlines() if ln.strip()]

        elif system == "Linux":
            out = subprocess.check_output(
                ["sh", "-c", "lspci | grep -i 'vga\\|3d'"],
                timeout=5
            ).decode(errors="ignore")
            return [ln.split(": ", 1)[-1].strip() for ln in out.splitlines() if ": " in ln]

        elif system == "Darwin":  # macOS
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                timeout=10
            ).decode(errors="ignore")
            return [m.strip() for m in re.findall(r"Chipset Model:\s*(.+)", out)]
    except Exception:
        pass

    return ["Nessuna GPU trovata"]

def sysinfo():

    machine = CFG["machine_info"]["machine"] or socket.gethostname() or "Not recognized"
    try:
        import cpuinfo
        cpu_fallback = cpuinfo.get_cpu_info().get("brand_raw")
    except Exception:
        cpu_fallback = platform.processor()

    cpu = CFG["machine_info"]["cpu"] or cpu_fallback or "Not recognized"
    gpu = CFG["machine_info"]["gpu"] or get_gpu_info() or "Not recognized"

    return {
        "machine": machine,
        "CPU": cpu,
        "GPU": gpu,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }