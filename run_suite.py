#!/usr/bin/env python
"""
Esegue tutti i test definiti in ai_bench_suite.yaml e fa append nei CSV designati.
"""

import yaml, subprocess, sys, itertools, time, pathlib, argparse
import webbrowser, pathlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
import nbformat
import secrets

from benches import encrypt_upload

CFG = yaml.safe_load(open("ai_bench_suite.yaml"))
PY = sys.executable 

def run_cmd(cmd):
    print("→", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode:
        print(f"⚠️  command failed (exit {res.returncode}): {' '.join(cmd)}")
        return False
    return True

def run_xgboost(run_id):
    any_ok = False
    for rows, gpu in itertools.product(CFG["xgboost"]["rows"], CFG["xgboost"]["gpu_used"]):
        cmd = [PY, CFG["xgboost"]["script"],
               "--repeats", str(CFG["xgboost"]["repeats"]),
               "--seed", str(CFG["seed"]),
               "--run-id", run_id,
               "--append-csv", CFG["xgboost"]["out_csv"]]
        if rows:  cmd += ["--sample-rows", str(rows)]
        if gpu:   cmd += ["--gpu"]
        cmd += ["--json"]    # facoltativo, per debug
        ok = run_cmd(cmd)
        any_ok = any_ok or ok
    if not any_ok:
        print("⚠️  All XGBoost commands failed (see messages above).")

def run_ollama(run_id):
    models = CFG["ollama"]["models"]
    if not models:
        # (d) Nessun modello: gestione esplicita e pulita
        print("ℹ️  No Ollama models listed in YAML; skipping Ollama suite.")
        return
    any_ok = False
    for model, gpu in itertools.product(models, CFG["ollama"]["gpu_used"]):
        cmd = [PY, CFG["ollama"]["script"],
               "--model", model,
               "--prompt", CFG["ollama"]["prompt"],
               "--repeats", str(CFG["ollama"]["repeats"]),
               "--seed", str(CFG["seed"]),
               "--run-id", run_id,
               "--append-csv", CFG["ollama"]["out_csv"]]
        if gpu:
            cmd += ["--gpu"]
        cmd += ["--json"]
        ok = run_cmd(cmd)
        any_ok = any_ok or ok
    if not any_ok:
        print("⚠️  All Ollama commands failed (see messages above).")

def handle_upload(args):
    """Cifra i CSV e li carica su Filebin (se non disabilitato)."""
    if args.no_upload_results or not CFG["upload"]["enabled_default"]:
        print("🔒  Upload disabled")
        return

    pub = encrypt_upload.load_pub(CFG["upload"]["public_key"])
    bin_id = CFG["upload"]["filebin_bin"]

    print("\n📦  Preparing upload")
    print(f"   • Destination  : https://filebin.net/{bin_id}/")
    print(f"   • Files to be uploaded: {', '.join(Path(p).name for p in CFG['upload']['targets'])}")

    for csv in CFG["upload"]["targets"]:
        path = pathlib.Path(csv)
        if not path.exists():
            print(f"⚠️  {csv} not found, next")
            continue

        enc_path = encrypt_upload.encrypt(pub, path)
        url = encrypt_upload.upload_to_filebin(enc_path, bin_id)
        print(f"📤  {enc_path.name} uploaded → {url}")
        try:
            enc_path.unlink()
        except OSError as e:
            print(f"⚠️  Impossible to delete {enc_path.name}: {e}")

def gen_run_id():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3).upper()   # 6 hex → 3 bytes ≈ 5-6 char leggibili
    return f"{ts}_{suffix[:5]}"

def inject_header_cell(notebook_path: str, run_id: str,
                       duration_s: float, suite_chosen: str):
    """
    Aggiunge o sostituisce la prima cella Markdown con un riepilogo della run.
    """
    nb = nbformat.read(notebook_path, as_version=4)

    human = str(timedelta(seconds=int(duration_s)))
    tests = {
        "xgboost": "🏋️ XGBoost",
        "ollama":  "🧠 Ollama",
        "both":    "🏋️ XGBoost  +  🧠 Ollama"
    }[suite_chosen]

    md = (
        f"## ✅ Benchmark completed\n\n"
        f"* **run_id**: `{run_id}`\n"
        f"* **Total duration**: {human}\n"
        f"* **Executed tests**: {tests}\n"
        f"* **Bars with thick border** ⇒ results of *this* run\n"
        f"* **Bars with thin border** ⇒ official references\n\n"
        "Thank you for running the suite! 🚀"
    )

    header = nbformat.v4.new_markdown_cell(md)

    if nb.cells and nb.cells[0].cell_type == "markdown" and "Benchmark completato" in nb.cells[0].source:
        nb.cells[0] = header          # sovrascrivi eventuale cella di run precedente
    else:
        nb.cells.insert(0, header)    # inserisci in testa

    nbformat.write(nb, notebook_path)

def render_notebook():
    nb_file  = "bench_results_analysis_altair.ipynb"
    html_out = pathlib.Path(nb_file).with_suffix(".html")

    print("📚  Eseguo notebook e genero HTML…")
    ok = run_cmd([            # <-- qui usiamo la helper già esistente
        "jupyter", "nbconvert",
        "--to", "html",
        "--execute",
        "--no-input",        # nessuna cella di codice nel risultato
        nb_file
    ])

    if ok:
        print(f"✅  HTML creato: {html_out}")
        webbrowser.open_new_tab(html_out.resolve().as_uri())
    else:
        print("⚠️  Failed to generate HTML report; you can execute the notebook manually.")

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
    parser.add_argument("--no-upload-results", action="store_true",
                        help="Do not encrypt / do not send results")

    args = parser.parse_args()

    run_id = gen_run_id()
    print(f"🔑  RUN ID: {run_id}")

    start_time = time.time()

    if args.suite in ("xgboost", "both"):
        print("Starting XGBoost benchmarks...")
        run_xgboost(run_id)
        print("XGBoost benchmarks completed.\n")

    if args.suite in ("ollama", "both"):
        print("Starting Ollama benchmarks...")
        run_ollama(run_id)
        print("Ollama benchmarks completed.\n")

    total = time.time() - start_time
    mins, secs = divmod(total, 60)
    handle_upload(args)

    # ↔  ---> CREAZIONE DELLA CELLA HEADER
    inject_header_cell(
        "bench_results_analysis_altair.ipynb",
        run_id=run_id,
        duration_s=total,
        suite_chosen=args.suite
    )

    print(f"\n✅ Total benchmark time: {int(mins)}m {secs:.2f}s")
    render_notebook()

if __name__ == "__main__":
    main()
