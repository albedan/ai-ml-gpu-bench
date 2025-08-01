#!/usr/bin/env python
"""
ollama_bench.py – benchmark locale di modelli Ollama (latency & throughput).

Esempio:
    python ollama_bench.py -m llama3 -p "Ciao, spiegami la relatività" -n 5 --gpu
"""
from __future__ import annotations

from bench_utils import emit, sysinfo

import argparse, time, statistics, json, os, sys
from pathlib import Path
import requests
from tabulate import tabulate
import datetime as dt

HOST = "http://localhost:11434"

def load_prompt(path_or_text: str) -> str:
    p = Path(path_or_text)
    return p.read_text() if p.exists() else path_or_text

def call_ollama(model: str, prompt: str, use_gpu: bool, seed: int):
    """Restituisce: wall_time, total_ns, eval_ns, eval_tokens"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "seed": seed,
            "temperature": 0.0,
            "num_gpu": -1 if use_gpu else 0,
        },
    }

    start = time.perf_counter()
    r = requests.post(f"{HOST}/api/generate", json=payload, timeout=1800)
    wall = time.perf_counter() - start
    r.raise_for_status()
    data = r.json()

    tot_ns  = data.get("total_duration", 0)
    eval_ns = data.get("eval_duration", 0)
    tokens  = data.get("eval_count", 0)
    txt     = data.get("response", "")
    return wall, tot_ns, eval_ns, tokens, len(txt)

def ns2s(ns: int) -> float:
    return ns / 1e9

# ────────────────────────────── CLI ───────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="llama3", help="nome modello Ollama")
    ap.add_argument("-p", "--prompt", required=True,
                    help="testo o file da usare come prompt")
    ap.add_argument("-n", "--repeats", type=int, default=3)
    ap.add_argument("--gpu", action="store_true",
                    help="lascia usare la GPU (default: CPU only)")
    ap.add_argument("--seed", type=int, default=42, help="seed riproducibile")
    ap.add_argument("--json", action="store_true",
                help="stampa il risultato in JSON al termine")
    ap.add_argument("--append-csv",
                help="percorso di un CSV su cui accodare la riga di output")    
    ap.add_argument("--run-id", required=False,
                help="identificativo unico della sessione benchmark")
    args = ap.parse_args()

    run_id = args.run_id or "dev"

    prompt = load_prompt(args.prompt)
    mode   = "GPU" if args.gpu else "CPU"

    w_times, e_times, t_times, sizes = [], [], [], []
    print(f"• Benchmarking {args.model} – {mode} – seed {args.seed}")
    for i in range(1, args.repeats + 1):
        print(f"  Run {i}/{args.repeats} … ", end="", flush=True)
        wall, tot_ns, eval_ns, tok, out_len = call_ollama(
            args.model, prompt, args.gpu, args.seed
        )
        w_times.append(wall)
        e_times.append(ns2s(eval_ns))
        t_times.append(tok / ns2s(eval_ns) if eval_ns else 0)
        sizes.append(out_len)
        print(f"{wall:,.2f}s  (tokens/s {t_times[-1]:.1f})")

    # riepilogo
    def min_med_max(arr):
        return min(arr), statistics.median(arr), max(arr)

    wall_min, wall_med, wall_max = min_med_max(w_times)
    tok_min,  tok_med,  tok_max  = min_med_max(t_times)

    rows = [
        ["Wall latency [s]", f"{wall_min:,.2f}", f"{wall_med:,.2f}", f"{wall_max:,.2f}"],
        ["Tokens/s",          f"{tok_min:,.1f}",  f"{tok_med:,.1f}",  f"{tok_max:,.1f}"],
    ]
    print("\n" + tabulate(rows,
        headers=["Metric", "min", "average", "max"],
        tablefmt="github"))
    
    result = {"run_id": run_id}
    result.update(sysinfo())
    result.update({           # costruisci il dict con tutte le metriche
        "bench":"ollama",
        "model": args.model,
        "gpu": args.gpu,
        "wall_min_s":   f"{wall_min:.2f}",
        "wall_med_s":   f"{wall_med:.2f}",
        "wall_max_s":   f"{wall_max:.2f}",
        "tok_min_s":    f"{tok_min:.2f}",
        "tok_med_s":    f"{tok_med:.2f}",
        "tok_max_s":    f"{tok_max:.2f}",
        "seed": args.seed,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "reference": False,
    })
    result.update(sysinfo())
    emit(result, args)

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        sys.exit("❌ Ollama apparently not running on http://localhost:11434")
