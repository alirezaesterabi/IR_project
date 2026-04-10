# Rerun Pipeline

This document is the canonical rerun guide for the retrieval pipeline.

The pipeline is the same each time:

1. preprocess raw JSONL into the canonical processed corpus path
2. rebuild lexical indices
3. build dense embeddings and Chroma
4. export BM25 and dense runs
5. fuse with RRF
6. evaluate all available runs

The only thing that should change between reruns is the raw input file used at the start.

## Raw Input Options

Use one of these raw JSONL files:

- `data/raw_data/sample_10k.json` for fast reruns and demos
- `data/raw_data/sample_100k.json` for larger local validation runs
- `data/raw_data/targets.nested.json` for full-scale runs

## Important Behavior

This workflow intentionally reuses the same downstream locations.

That means each rerun overwrites the current canonical artifacts:

- processed corpus: `data/json_format_data/subset/`
- lexical models: `models/`
- dense caches: `models/doc_embeddings_*` and `models/doc_ids_*`
- dense store: `chroma_db/`
- run files: `results/runs/`
- evaluation outputs: `results/evaluation/`

Do not run multiple dataset variants in parallel if you want to preserve earlier outputs.

The processed-corpus folder name is intentionally general. Even if the raw input is 10K, 100K, or full, the canonical processed output should still live under `data/json_format_data/subset/`.

## Environment Setup

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`requirements.txt` already installs the supported local pipeline dependencies plus notebook support.

Optional extras that are **not** required for this rerun guide:

- `requirements-tests.txt` for running the test suite
- `requirements-rag.txt` for experimental RAG components

If you want the evaluation notebook to appear as a named kernel in Jupyter, register the current virtual environment once:

```bash
python -m ipykernel install --user --name ir_project_venv --display-name "Python (IR Project .venv)"
```

## Run Profiles

Choose one profile before starting the pipeline. The later steps use the same canonical downstream locations, but the raw input and dense run tag should match the profile you picked.

### 10K Preprocess

```python
RAW_INPUT = Path("data/raw_data/sample_10k.json")
RUN_TAG = "10K_YYYYMMDD"
DENSE_LIMIT = 10000
```

### 100K Preprocess

```python
RAW_INPUT = Path("data/raw_data/sample_100k.json")
RUN_TAG = "100K_YYYYMMDD"
DENSE_LIMIT = 100000
```

### Full Preprocess

```python
RAW_INPUT = Path("data/raw_data/targets.nested.json")
RUN_TAG = "FULL_YYYYMMDD"
DENSE_LIMIT = None
```

## Step 1: Preprocess

Use the profile that matches the corpus you want to process.

### 10K Dense Build

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from src.preprocessing.pipeline import run_pipeline

RAW_INPUT = Path("data/raw_data/sample_10k.json")

run_pipeline(
    input_path=RAW_INPUT,
    output_dir=Path("data/json_format_data/subset"),
    max_records=None,
    show_progress=True,
)
PY
```

### 100K Dense Build

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from src.preprocessing.pipeline import run_pipeline

RAW_INPUT = Path("data/raw_data/sample_100k.json")

run_pipeline(
    input_path=RAW_INPUT,
    output_dir=Path("data/json_format_data/subset"),
    max_records=None,
    show_progress=True,
)
PY
```

### Full Dense Build

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from src.preprocessing.pipeline import run_pipeline

RAW_INPUT = Path("data/raw_data/targets.nested.json")

run_pipeline(
    input_path=RAW_INPUT,
    output_dir=Path("data/json_format_data/subset"),
    max_records=None,
    show_progress=True,
)
PY
```

Expected outputs:

- `data/json_format_data/subset/documents.jsonl`
- `data/json_format_data/subset/stats.json`

Check `stats.json` after each rerun to confirm which raw file produced the current corpus.

## Step 2: Build Lexical Indices

This command is the same for `10K`, `100K`, and `full` because the processed corpus always lives in the same canonical location.

```bash
.venv/bin/python scripts/build_index.py --docs data/json_format_data/subset/documents.jsonl
```

Expected outputs:

- `models/bm25/index.pkl`
- `models/bm25/doc_ids.json`
- `models/tfidf/vectorizer.pkl`
- `models/tfidf/matrix.npz`
- `models/tfidf/doc_ids.json`
- `models/identifier/index.pkl`

## Step 3: Build Dense Embeddings

Use the `RUN_TAG` and `DENSE_LIMIT` that match your chosen profile.

### 10K Dense Export

```bash
.venv/bin/python scripts/build_dense_embeddings.py \
  --docs data/json_format_data/subset/documents.jsonl \
  --limit 10000 \
  --model bge-m3 \
  --chroma \
  --run-tag 10K_20260409
```

### 100K Dense Export

```bash
.venv/bin/python scripts/build_dense_embeddings.py \
  --docs data/json_format_data/subset/documents.jsonl \
  --limit 100000 \
  --model bge-m3 \
  --chroma \
  --run-tag 100K_20260409
```

### Full Dense Export

```bash
.venv/bin/python scripts/build_dense_embeddings.py \
  --docs data/json_format_data/subset/documents.jsonl \
  --model bge-m3 \
  --chroma \
  --run-tag FULL_20260409
```

Expected outputs:

- `models/doc_embeddings_bge_m3_<RUN_TAG>.npy`
- `models/doc_ids_bge_m3_<RUN_TAG>.json`
- Chroma collection `opensanctions_bge_m3_<RUN_TAG>`

## Step 4: Export BM25 Run

This command is the same for `10K`, `100K`, and `full`.

```bash
.venv/bin/python scripts/export_bm25_run.py
```

Expected output:

- `results/runs/bm25.csv`

## Step 5: Export Dense Run

Use the same `RUN_TAG` you used when building dense embeddings.

### 10K

```bash
.venv/bin/python scripts/export_dense_run.py --model bge-m3 --run-tag 10K_20260409
```

### 100K

```bash
.venv/bin/python scripts/export_dense_run.py --model bge-m3 --run-tag 100K_20260409
```

### Full

```bash
.venv/bin/python scripts/export_dense_run.py --model bge-m3 --run-tag FULL_20260409
```

Expected output:

- `results/runs/dense_bge_m3.csv`

## Step 6: Fuse With RRF

This command is the same for `10K`, `100K`, and `full` because it always reads the canonical run files written in the previous steps.

```bash
.venv/bin/python -m src.fusion.rrf \
  --bm25 results/runs/bm25.csv \
  --dense results/runs/dense_bge_m3.csv \
  --output results/runs/rrf_bge_m3.csv
```

Expected output:

- `results/runs/rrf_bge_m3.csv`

## Step 7: Evaluate In The Notebook

This notebook is the primary evaluation step for the rerun pipeline:

- `notebooks/05_evaluation_types_1_6.ipynb`

It expects the canonical run files produced in the earlier steps:

- `results/runs/bm25.csv`
- `results/runs/dense_bge_m3.csv`
- `results/runs/rrf_bge_m3.csv`

This step is also the same for `10K`, `100K`, and `full`. The notebook reads whichever canonical run files were most recently generated.

Open the notebook from the repository root and run all cells:

```bash
. .venv/bin/activate
jupyter lab notebooks/05_evaluation_types_1_6.ipynb
```

In most cases, starting Jupyter from the activated `.venv` is enough for interactive use.

If you prefer a non-interactive execution that still writes outputs into the notebook file itself:

```bash
.venv/bin/python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=ir_project_venv \
  notebooks/05_evaluation_types_1_6.ipynb
```

The `nbconvert` command above assumes you registered the `.venv` once as `ir_project_venv` during environment setup.

Expected outputs:

- rendered outputs inside `notebooks/05_evaluation_types_1_6.ipynb`
- `results/evaluation/types_1_6_bm25_by_type.csv`
- `results/evaluation/types_1_6_bm25_overall.csv`
- `results/evaluation/types_1_6_bm25_per_query.csv`
- `results/evaluation/types_1_6_dense_by_type.csv`
- `results/evaluation/types_1_6_rrf_by_type.csv`
- `results/evaluation/types_1_6_rrf_overall.csv`
- `results/evaluation/types_1_6_rrf_per_query.csv`
- `results/evaluation/types_1_6_bm25_vs_rrf_by_type.csv`
- `results/evaluation/types_1_6_bm25_vs_rrf_overall.csv`

Script alternative:

```bash
.venv/bin/python scripts/evaluate_runs.py
```

## Notes

- The current query and qrels files under `data/queries/` and `data/qrels/` are reused across reruns.
- Smaller corpora can lead to shallow or empty coverage for some query types. This is expected and should be interpreted as a dataset-size limitation, not necessarily a pipeline failure.
- The evaluation helpers now handle empty-run subsets without crashing, so very small reruns can still produce reports.
