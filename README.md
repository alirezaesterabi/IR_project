# Information Retrieval Project — OpenSanctions Entity Search

An end-to-end Information Retrieval system built on the [OpenSanctions](https://www.opensanctions.org/) dataset (~1.24 M sanctioned entities), developed as part of the QMUL MSc programme.

---

## Project Overview

| Phase | Status | Description |
| ----- | ----------- | ----------- |
| 1 | Done | Data exploration and schema analysis |
| 2 | Done | Preprocessing pipeline (parser → flattener → text processor) |
| 3 | Done | Query set development (50 queries across 7 types) |
| 4 | Implemented | Classical indices, dense embedding build, Chroma retrieval, and RRF fusion |
| 5 | In Progress | Evaluation notebooks/utilities and reproducible run export |
| 6 | Pending | BM25F multi-field extension |

---

## Repository Structure

```text
IR_project/
├── data/
│   ├── json_format_data/                 # Processed corpora (subset/full)
│   ├── qrels/                            # qrels_type_1.json ... qrels_type_6.json
│   ├── queries/                          # queries_type_1.json ... queries_type_6.json
│   ├── raw_data/                         # OpenSanctions source data + sample
│   └── toy_example/                      # Small teaching corpus
├── src/
│   ├── preprocessing/                    # Parser, document builder, embedding_text pipeline
│   ├── retrieval/
│   │   ├── classical_ir.py              # BM25, TF-IDF, Identifier retrievers
│   │   ├── dense_config.py              # Dense model registry + RUN_TAG naming
│   │   └── dense_retriever.py           # Query-time SentenceTransformer + Chroma retriever
│   ├── fusion/
│   │   └── rrf.py                       # Reciprocal Rank Fusion
│   └── evaluation/
│       └── utils.py                     # ranx helpers for per-type and overall metrics
├── scripts/
│   ├── build_index.py                    # Build BM25 / TF-IDF / Identifier indices
│   ├── build_dense_embeddings.py         # Build dense embeddings + optional Chroma upsert
│   ├── export_bm25_run.py                # Export BM25 run CSV for Types 1-6
│   ├── export_dense_run.py               # Export dense run CSV for MiniLM / BGE-M3
│   └── evaluate_runs.py                  # Evaluate BM25, dense, and fused runs
├── notebooks/                            # Exploration, pooling, and analysis notebooks
├── tests/                                # Unit tests for retrieval, fusion, and evaluation
├── requirements.txt
└── .gitignore
```

---

## Data

The full OpenSanctions dataset (~3.7 GB) is **not tracked** in this repository.
See [`data/DATA_SOURCE.md`](data/DATA_SOURCE.md) for download instructions.

**What is tracked:**

- `data/raw_data/sample_targets.json` — 100-record JSONL sample (used by all learning notebooks)
- `data/toy_example/documents.jsonl` — flattened and preprocessed version of the 100-record sample
- `data/evaluation/queries_part_a.xlsx` — 50 auto-generated evaluation queries

---

## Learning Modules

The `learning/` folder contains two self-contained modules that teach IR concepts using the real OpenSanctions toy corpus as a running example.

### Module 1 — Text Processing & Indexing

| Notebook | Topics |
| -------- | ------ |
| `01_preprocessing.ipynb` | Tokenisation, normalisation, stop-word removal, stemming, lemmatisation, Zipf's Law, Heap's Law |
| `02_indexing_boolean_retrieval.ipynb` | Term-document matrix, inverted index, BSBI, Boolean AND/OR/NOT, positional index, phrase queries, VByte compression, TF-IDF/BM25 preview |

### Module 2 — Ranked Retrieval

| Notebook | Topics |
| -------- | ------ |
| `01_ranked_retrieval.ipynb` | Boolean → VSM → TF-IDF → BM25 → BM25F (covers all Week 3 lab exercises + theory.md worked examples) |

Both modules use `data/toy_example/documents.jsonl` as the shared running example and reference theory files with full mathematical derivations.

---

## Quick Start

```bash
# 1. Clone and set up
git clone <repo-url>
cd IR_project
python -m venv env && source env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run the preprocessing pipeline on the sample (no full dataset needed)
python -m src.preprocessing.pipeline \
    --input data/raw_data/sample_targets.json \
    --output data/toy_example/documents.jsonl

# 3. Open the learning notebooks
jupyter lab learning/modules/01_text_processing_indexing_retrieval/01_preprocessing.ipynb
```

---

## Dense And Hybrid Workflow

The dense pipeline is model-agnostic: both `minilm` and `bge-m3` are defined in
`src/retrieval/dense_config.py`, and the same export / evaluation flow works for
either model as long as the correct `RUN_TAG` is used.

```bash
# 1. Build the lexical baseline
python scripts/build_index.py --model bm25

# 2. Build dense embeddings and upsert them into ChromaDB
python scripts/build_dense_embeddings.py \
    --model minilm \
    --chroma \
    --run-tag FULL_20260407

# 3. Export comparable run files
python scripts/export_bm25_run.py
python scripts/export_dense_run.py --model minilm --run-tag FULL_20260407

# 4. Fuse BM25 + dense with RRF
python -m src.fusion.rrf \
    --bm25 results/runs/bm25.csv \
    --dense results/runs/dense_minilm.csv \
    --output results/runs/rrf_minilm.csv

# 5. Evaluate all available runs
python scripts/evaluate_runs.py
```

To evaluate `bge-m3`, build the matching Chroma collection and run:

```bash
python scripts/export_dense_run.py --model bge-m3 --run-tag FULL_20260407
python -m src.fusion.rrf \
    --bm25 results/runs/bm25.csv \
    --dense results/runs/dense_bge_m3.csv \
    --output results/runs/rrf_bge_m3.csv
python scripts/evaluate_runs.py
```

---

## Requirements

Key dependencies (see `requirements.txt` for pinned versions):

- `spacy` + `en_core_web_sm` — lemmatisation
- `nltk` — stop words, Porter stemmer
- `scikit-learn` — `CountVectorizer` (learning modules)
- `pandas`, `numpy`, `matplotlib` — data handling and visualisation
- `rank-bm25` — BM25 index (Phase 4)
- `openpyxl` — query set export
