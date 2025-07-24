# bench_utils.py
from __future__ import annotations
import yaml, json, csv, pathlib, datetime as dt
import platform, subprocess, re, sys, importlib, os
from typing import Dict

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

def sysinfo():
    return {
        "machine": CFG["machine_info"]["machine"],
        "CPU": CFG["machine_info"]["cpu"],
        "GPU": CFG["machine_info"]["gpu"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }