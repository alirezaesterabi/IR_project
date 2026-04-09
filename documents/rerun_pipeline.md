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

- processed corpus: `data/json_format_data/subset_100k/`
- lexical models: `models/`
- dense caches: `models/doc_embeddings_*` and `models/doc_ids_*`
- dense store: `chroma_db/`
- run files: `results/runs/`
- evaluation outputs: `results/evaluation/`

Do not run multiple dataset variants in parallel if you want to preserve earlier outputs.

## Environment Setup

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you want to run the evaluation notebook locally and Jupyter is not already available in the environment, install it once:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name ir_project_venv --display-name "Python (IR Project .venv)"
```

## Step 1: Preprocess

Pick the raw input you want and replace the `RAW_INPUT` path below.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from src.preprocessing.pipeline import run_pipeline

RAW_INPUT = Path("data/raw_data/sample_10k.json")

run_pipeline(
    input_path=RAW_INPUT,
    output_dir=Path("data/json_format_data/subset_100k"),
    max_records=None,
    show_progress=True,
)
PY
```

Expected outputs:

- `data/json_format_data/subset_100k/documents.jsonl`
- `data/json_format_data/subset_100k/stats.json`

Check `stats.json` after each rerun to confirm which raw file produced the current corpus.

## Step 2: Build Lexical Indices

```bash
.venv/bin/python scripts/build_index.py --docs data/json_format_data/subset_100k/documents.jsonl
```

Expected outputs:

- `models/bm25/index.pkl`
- `models/bm25/doc_ids.json`
- `models/tfidf/vectorizer.pkl`
- `models/tfidf/matrix.npz`
- `models/tfidf/doc_ids.json`
- `models/identifier/index.pkl`

## Step 3: Build Dense Embeddings

Choose a run tag that matches the corpus you are building.

Examples:

- `10K_YYYYMMDD`
- `100K_YYYYMMDD`
- `FULL_YYYYMMDD`

For a 10K rerun:

```bash
.venv/bin/python scripts/build_dense_embeddings.py \
  --docs data/json_format_data/subset_100k/documents.jsonl \
  --limit 10000 \
  --model minilm \
  --chroma \
  --run-tag 10K_20260409
```

For a 100K rerun:

```bash
.venv/bin/python scripts/build_dense_embeddings.py \
  --docs data/json_format_data/subset_100k/documents.jsonl \
  --limit 100000 \
  --model minilm \
  --chroma \
  --run-tag 100K_20260409
```

Expected outputs:

- `models/doc_embeddings_minilm_<RUN_TAG>.npy`
- `models/doc_ids_minilm_<RUN_TAG>.json`
- Chroma collection `opensanctions_minilm_<RUN_TAG>`

## Step 4: Export BM25 Run

```bash
.venv/bin/python scripts/export_bm25_run.py
```

Expected output:

- `results/runs/bm25.csv`

## Step 5: Export Dense Run

Use the same run tag you used when building dense embeddings.

```bash
.venv/bin/python scripts/export_dense_run.py --model minilm --run-tag 10K_20260409
```

Expected output:

- `results/runs/dense_minilm.csv`

## Step 6: Fuse With RRF

```bash
.venv/bin/python -m src.fusion.rrf \
  --bm25 results/runs/bm25.csv \
  --dense results/runs/dense_minilm.csv \
  --output results/runs/rrf_minilm.csv
```

Expected output:

- `results/runs/rrf_minilm.csv`

## Step 7: Evaluate In The Notebook

This notebook is the primary evaluation step for the rerun pipeline:

- `notebooks/05_evaluation_types_1_6.ipynb`

It expects the canonical run files produced in the earlier steps:

- `results/runs/bm25.csv`
- `results/runs/dense_minilm.csv`
- `results/runs/rrf_minilm.csv`

Open the notebook from the repository root and run all cells:

```bash
. .venv/bin/activate
jupyter lab notebooks/05_evaluation_types_1_6.ipynb
```

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

Expected outputs:

- rendered outputs inside `notebooks/05_evaluation_types_1_6.ipynb`
- `results/evaluation/overall_summary.csv`
- `results/evaluation/by_type_bm25.csv`
- `results/evaluation/by_type_dense_minilm.csv`
- `results/evaluation/by_type_rrf_minilm.csv`
- `results/evaluation/per_query_*.csv`
- `results/evaluation/coverage_*.csv`
- `results/evaluation/comparison_bm25_vs_rrf_minilm.csv`

Script alternative:

```bash
.venv/bin/python scripts/evaluate_runs.py
```

## Notes

- The current query and qrels files under `data/queries/` and `data/qrels/` are reused across reruns.
- Smaller corpora can lead to shallow or empty coverage for some query types. This is expected and should be interpreted as a dataset-size limitation, not necessarily a pipeline failure.
- The evaluation helpers now handle empty-run subsets without crashing, so very small reruns can still produce reports.
- `sample_targets.json` and `toy_example` remain in the repository for older learning materials, but they are not the main rerun path documented here.
