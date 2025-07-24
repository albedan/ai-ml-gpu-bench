# AI & ML GPU Bench Suite for Python 

Benchmark riproducibili per:

- **XGBoost** (train & inferenza su dataset HIGGS)
- **Ollama LLMs** (latency & throughput per token)

Il tutto è orchestrato da un singolo YAML e da `run_suite.py`, così puoi lanciare un’intera batteria di test con un solo comando.

---

## Requisiti

Assicurati di aver installato almeno le componenti must tra le seguenti

| Requisito                  | Perché serve                                       | Come installare                                                                                                   | Necessario? |
|----------------------------|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------|
| **Python ≥ 3.13**          | Runtime per gli script                             | <https://www.python.org/>                                                                                         | **Must**    |
| **uv 0.8.x**                     | Gestore pacchetti & lock-file super-veloce         | <https://docs.astral.sh/uv/getting-started/installation/>                                                         | **Must**    |
| **CUDA ≥ 12.x**            | Benchmark GPU (XGBoost + CuPy, Ollama)             | Driver NVIDIA + <https://developer.nvidia.com/cuda-downloads>                                                     | **Optional**<br><sub>(solo se nel YAML è selezionata una GPU)</sub> |
| **Ollama** (in esecuzione su http://localhost:11434) | Benchmark LLM via REST API                         | <https://ollama.com/download>                                                                                     | **Optional**<br><sub>(solo se si vogliono testare gli LLM)</sub> |

---

## Setup ambiente (con uv)

```bash
# 1. Crea l’ambiente virtuale
uv venv .venv
source .venv/bin/activate

# 2. Installa le dipendenze
uv sync
```

---

## Configurazione: `ai_bench_suite.yaml`

Tutti i parametri di benchmark vivono qui:

```yaml
machine_info:
  machine: "PC_AL_2025"     # Scegli il tuo nome host sintetico
  cpu:    "AMD Ryzen 5 9600X" # Nome commerciale CPU
  gpu:    "Nvidia RTX 5060 16GB" # Nome GPU
```

- **Aggiorna** questi tre campi in base alla tua macchina.
- **Commenta / decommenta** le voci nella sezione ``ollama`` per includere o escludere LLM che non hai scaricato.
- Per gli LLM che hai lasciato non commentati, verifica di averli disponibili con ``ollama list`` da terminale. Puoi installarli con ``ollama pull [nome_llm]``.
- Ogni combinazione elencata in `rows` × `gpu` (per XGBoost) e `models` × `gpu` (per Ollama) viene provata automaticamente.

> Il seed globale è impostato a `42`, modificabile via YAML.

---

## Esecuzione

### Tutta la suite

```bash
uv run run_suite.py
```

Lo script legge lo YAML, lancia gli script indicati e appende i risultati nei CSV dichiarati.

### Singoli benchmark

```bash
# XGBoost, CPU, 1 milione di righe, 5 ripetizioni
uv run benches/xgb_bench.py --sample-rows 1000000 -n 5 --append-csv results/xgb.csv

# Ollama, modello gemma3:12b, GPU, prompt da file
uv run benches/ollama_bench.py -m gemma3:12b --gpu -p Spiegami la relatività --append-csv results/ollama.csv
```

Ogni script accetta `--json` per stampare il risultato in chiaro oltre a salvare su CSV.

---

## Output

- **CSV**: una riga per benchmark con tutte le metriche e metadati macchina.
- **Notebook** (`bench_results_analysis.ipynb`): esplora i CSV e genera grafici per analizzare le performance. Sono presenti alcuni risultati di riferimento.

---

## FAQ

### 𝐐  "Ollama non parte su GPU"

- Assicurati di avere CUDA 12 e un driver >= 535.
- Alcuni modelli non hanno build quantizzate GPU: lascia commented nelle `models` o lancia solo `--cpu`.

### 𝐐  "ImportError: cannot import cupy"

- Installa la build corretta per la tua versione CUDA: `uv pip install cupy-cuda12x`.

### 𝐐  "Out Of Memory su XGBoost GPU"

- Riduci `rows` nello YAML.

---

## Contribuire

TBD!
