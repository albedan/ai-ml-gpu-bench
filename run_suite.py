#!/usr/bin/env python
"""
Esegue tutti i test definiti in ai_bench_suite.yaml e fa append nei CSV designati.
"""

import yaml, subprocess, sys, itertools, time, pathlib
import argparse

CFG = yaml.safe_load(open("ai_bench_suite.yaml"))
PY = sys.executable 

def run_cmd(cmd):
    print("→", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode:
        sys.exit(f"❌ comando fallito: {cmd}")

def run_xgboost():
    for rows, gpu in itertools.product(CFG["xgboost"]["rows"], CFG["xgboost"]["gpu"]):
        cmd = [PY, CFG["xgboost"]["script"],
               "--repeats", str(CFG["xgboost"]["repeats"]),
               "--seed", str(CFG["seed"]),
               "--append-csv", CFG["xgboost"]["out_csv"]]
        if rows:  cmd += ["--sample-rows", str(rows)]
        if gpu:   cmd += ["--gpu"]
        cmd += ["--json"]    # facoltativo, per debug
        run_cmd(cmd)

def run_ollama():
    for model, gpu in itertools.product(CFG["ollama"]["models"], CFG["ollama"]["gpu"]):
        cmd = [PY, CFG["ollama"]["script"],
               "--model", model,
               "--prompt", CFG["ollama"]["prompt"],
               "--repeats", str(CFG["ollama"]["repeats"]),
               "--seed", str(CFG["seed"]),
               "--append-csv", CFG["ollama"]["out_csv"]]
        if gpu:
            cmd += ["--gpu"]
        cmd += ["--json"]
        run_cmd(cmd)

def main():
    pathlib.Path("results").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Run benchmark suite: XGBoost, Ollama, or both"
    )
    parser.add_argument(
        "--suite",
        choices=["xgboost", "ollama", "both"],
        default="both",
        help="Which benchmark suite to run (default: both)",
    )
    args = parser.parse_args()

    start_time = time.time()

    if args.suite in ("xgboost", "both"):
        print("Starting XGBoost benchmarks...")
        run_xgboost()
        print("XGBoost benchmarks completed.\n")

    if args.suite in ("ollama", "both"):
        print("Starting Ollama benchmarks...")
        run_ollama()
        print("Ollama benchmarks completed.\n")

    total = time.time() - start_time
    mins, secs = divmod(total, 60)
    print(f"\n✅ Total benchmark time: {int(mins)}m {secs:.2f}s")

if __name__ == "__main__":
    main()
