#!/usr/bin/env python
"""
xgb_bench.py – benchmark XGBoost (training + inference) riproducibile.

Esempio:
    python xgb_bench.py -n 5 --gpu --sample-rows 2_000_000 --seed 42
"""
from __future__ import annotations

from bench_utils import emit, sysinfo

import argparse, pathlib, time, urllib.request, os, statistics
from typing import Tuple
import datetime as dt
import hashlib
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ────────── configurazione ────────────────────────────────────────────────────
HIGGS_URL = (
    "https://huggingface.co/datasets/albedan/higgs_parquet/resolve/main/HIGGS.parquet"
)
DATA_DIR = pathlib.Path("./data")
PARQUET_FILE = DATA_DIR / "HIGGS.parquet"
DEFAULT_SEED = 42
EXPECTED_SHA256_PARQUET = '8ca8941fcc1848930bf8a1890035bc2750c063975e75bd6803c249b235edced7'

# ────────── dataset helper ────────────────────────────────────────────────────
def download_dataset() -> pathlib.Path:
    DATA_DIR.mkdir(exist_ok=True)
    if not PARQUET_FILE.exists():
        print(f"⬇️  Downloading HIGGS… ({HIGGS_URL})")
        urllib.request.urlretrieve(HIGGS_URL, PARQUET_FILE)
    else:
        print("✅ Dataset already present.")
    return PARQUET_FILE

def download_dataset() -> pathlib.Path:
    DATA_DIR.mkdir(exist_ok=True)

    if not PARQUET_FILE.exists():
        pbar = [None]

        def reporthook(count, block_size, total_size):
            if pbar[0] is None:
                pbar[0] = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading"
                )
            downloaded = count * block_size
            if downloaded <= total_size:
                pbar[0].n = downloaded
                pbar[0].refresh()
            else:
                pbar[0].n = total_size
                pbar[0].refresh()

        print(f"⬇️  Downloading HIGGS… ({HIGGS_URL})")
        urllib.request.urlretrieve(HIGGS_URL, PARQUET_FILE, reporthook)
        if pbar[0]:
            pbar[0].close()
    else:
        print("✅ Dataset already present.")

    return PARQUET_FILE

def load_higgs(rows: int | None, seed: int) -> Tuple[np.ndarray, ...]:
    """Legge il CSV in RAM con pandas; restituisce NumPy array + split stabile."""

    # --- BLOCCO MINIMALE DI VERIFICA/REDOWNLOAD --------------------------------
    def _calc_digest(p: pathlib.Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):  # 4MB
                h.update(chunk)
        return h.hexdigest()

    existed_before = PARQUET_FILE.exists()
    csv_path = download_dataset()

    calc = _calc_digest(csv_path)
    if calc.lower() != EXPECTED_SHA256_PARQUET:
        if existed_before:
            print("⚠️  Digest mismatch: elimino e riscarico…")
            PARQUET_FILE.unlink(missing_ok=True)
            download_dataset()
            calc = _calc_digest(PARQUET_FILE)
            if calc.lower() != EXPECTED_SHA256_PARQUET:
                raise RuntimeError(
                    f"SHA256 non corrisponde dopo il redownload: {calc}"
                )
        else:
            raise RuntimeError(f"SHA256 non corrisponde: {calc}")
    else:
        print(f"✅ Digest ok ({calc})")

    print(f"📥 Loading {'all' if rows is None else f'{rows:_}'} rows…")
    df = pd.read_parquet(csv_path, engine='fastparquet')[:rows]
    X = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0].to_numpy(np.int8)

    np.random.seed(seed)
    return train_test_split(X, y, test_size=0.2, random_state=seed)


# ────────── training + inference ──────────────────────────────────────────────
def run_training(
    X_train,
    y_train,
    X_test,
    y_test,
    use_gpu: bool,
    seed: int,
    booster_params: dict | None = None,
):
    booster_params = booster_params or {}
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=os.cpu_count(),
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        seed=seed,
        random_state=seed,
    )
    params.update(booster_params)

    if use_gpu:
        import cupy as cp

        cp.random.seed(seed)
        X_train = cp.asarray(X_train)
        X_test = cp.asarray(X_test)
        y_train = cp.asarray(y_train)

    model = xgb.XGBClassifier(**params)

    # ─ training ─
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # ─ inference ─
    t1 = time.perf_counter()
    proba = model.predict_proba(X_test, validate_features=False)[:, 1]
    infer_time = time.perf_counter() - t1

    if use_gpu:  # CuPy → NumPy per la metrica
        import cupy as cp

        proba = cp.asnumpy(proba)

    auc = roc_auc_score(y_test, proba)
    return train_time, infer_time, auc


# ────────── main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--repeats", type=int, default=3, help="iterazioni")
    ap.add_argument("--gpu", action="store_true", help="usa la GPU (richiede CuPy)")
    ap.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="righe da leggere (default: tutte ~11 M)",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed globale")
    ap.add_argument("--json", action="store_true",
                help="stampa il risultato in JSON al termine")
    ap.add_argument("--append-csv",
                help="percorso di un CSV su cui accodare la riga di output")
    ap.add_argument("--run-id", required=False,
                help="identificativo unico della sessione benchmark")
    args = ap.parse_args()

    run_id = args.run_id or "dev"
    try:
        X_train, X_test, y_train, y_test = load_higgs(args.sample_rows, args.seed)
    except MemoryError as e:
        print(f"⚠️  HIGGS load failed (out of memory): {e}. Skipping this configuration.")
        return 0
    except Exception as e:
        print(f"⚠️  HIGGS load failed: {e}. Skipping this configuration.")
        return 0
    
    train_times, infer_times, aucs = [], [], []
    for i in range(1, args.repeats + 1):
        print(f"\n🏃 Run {i}/{args.repeats} …")
        try:
            t_train, t_inf, auc = run_training(
                X_train, y_train, X_test, y_test, args.gpu, args.seed
            )
        except Exception as e:
            print(f"⚠️  XGBoost run failed (GPU={args.gpu}): {e}. Skipping this run.")
            continue
        train_times.append(t_train)
        infer_times.append(t_inf)
        aucs.append(auc)
        print(f"⏱️  Train {t_train:,.2f} s  |  Infer {t_inf:,.2f} s  |  AUC {auc:.4f}")

    # ─ summary ─
    def stats(arr):
        return (
            statistics.median(arr),
            min(arr),
            max(arr),
            statistics.stdev(arr) if len(arr) > 1 else 0.0,
        )

    if not train_times:
        print("⚠️  No successful XGBoost runs; not writing CSV for this configuration.")
        return 0
    
    t_median, t_min, t_max, t_std = stats(train_times)
    i_median, i_min, i_max, i_std = stats(infer_times)

    print("\n========= RISULTATI =========")
    print(f"[Training]  mediana {t_median:,.2f} s | min {t_min:,.2f} | max {t_max:,.2f} | σ {t_std:,.2f}")
    print(f"[Inferenza] mediana {i_median:,.2f} s | min {i_min:,.2f} | max {i_max:,.2f} | σ {i_std:,.2f}")
    print(f"AUC med    : {statistics.median(aucs):.4f}")
    print(f"Seed usato : {args.seed}")
    print("=============================")

    result = {"run_id": run_id}
    result.update(sysinfo())
    result.update({           # costruisci il dict con tutte le metriche
        "bench":"xgboost",
        "version":xgb.__version__,
        "dataset_rows": args.sample_rows if args.sample_rows else "full",
        "gpu_used": args.gpu,
        "train_median_s": f"{t_median:.2f}",
        "infer_median_s": f"{i_median:.2f}",
        "auc": rf"{statistics.median(aucs):.5f}",
        "seed": args.seed,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "reference": False,
    })
    emit(result, args)

if __name__ == "__main__":
    main()
