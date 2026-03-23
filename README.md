# Information Retrieval Project — OpenSanctions Entity Search

An end-to-end Information Retrieval system built on the [OpenSanctions](https://www.opensanctions.org/) dataset (~1.24 M sanctioned entities), developed as part of the QMUL MSc programme.

---

## Project Overview

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | Done | Data exploration and schema analysis |
| 2 | Done | Preprocessing pipeline (parser → flattener → text processor) |
| 3 | Done | Query set development (50 queries across 7 types) |
| 4 | Pending | Indexing and retrieval (BM25 via `rank-bm25`) |
| 5 | Pending | Evaluation (Precision, Recall, F1) |
| 6 | Pending | BM25F multi-field extension |

---

## Repository Structure

```
IR_project/
│
├── data/
│   ├── DATA_SOURCE.md                    # Data provenance and download instructions
│   ├── raw_data/
│   │   └── sample_targets.json           # 100-record sample (used by learning notebooks)
│   ├── toy_example/
│   │   ├── documents.jsonl               # Flattened toy corpus (100 records, ~96 KB)
│   │   └── README.md                     # Schema and loading instructions
│   └── evaluation/
│       ├── queries_part_a.xlsx           # Auto-generated queries (types 1, 2, 5, 6)
│       └── queries_part_b_template.xlsx  # Template for human-curated queries (types 3, 4, 7)
│
├── src/preprocessing/                    # Phase 2 preprocessing pipeline
│   ├── parser.py                         # Streaming JSONL parser
│   ├── document_builder.py               # Nested JSON flattener → text_blob + tokens
│   ├── text_processing.py                # Normalise, stop-word removal, lemmatisation
│   └── pipeline.py                       # Orchestrates all steps (CLI entry point)
│
├── notebooks/
│   ├── 01_json_exploration.ipynb         # Phase 1: raw data exploration
│   ├── 02_preprocessing_validation.ipynb # Phase 2: single-row pipeline walkthrough
│   └── 03_query_generation.ipynb         # Phase 3: query set generation (50 queries × 4 types)
│
├── learning/
│   ├── modules/
│   │   ├── 01_text_processing_indexing_retrieval/
│   │   │   ├── theory.md                 # Full Module 1 theory (preprocessing → Boolean IR)
│   │   │   ├── 01_preprocessing.ipynb    # Preprocessing concepts on toy corpus
│   │   │   └── 02_indexing_boolean_retrieval.ipynb  # Indexing, BSBI, Boolean, TF-IDF preview
│   │   └── 02_ranked_retrieval/
│   │       ├── theory.md                 # Full Module 2 theory (TF-IDF → BM25 → LM)
│   │       └── 01_ranked_retrieval.ipynb # Boolean · VSM · TF-IDF · BM25 · BM25F (Week 3 lab)
│   └── raw_materials/                    # Lecture slides and original lab notebooks
│
├── documents/
│   ├── implementation_phases.md          # Detailed phase specifications
│   └── integrated_learning_project_plan.md
│
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
|----------|--------|
| `01_preprocessing.ipynb` | Tokenisation, normalisation, stop-word removal, stemming, lemmatisation, Zipf's Law, Heap's Law |
| `02_indexing_boolean_retrieval.ipynb` | Term-document matrix, inverted index, BSBI, Boolean AND/OR/NOT, positional index, phrase queries, VByte compression, TF-IDF/BM25 preview |

### Module 2 — Ranked Retrieval

| Notebook | Topics |
|----------|--------|
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

## Requirements

Key dependencies (see `requirements.txt` for pinned versions):

- `spacy` + `en_core_web_sm` — lemmatisation
- `nltk` — stop words, Porter stemmer
- `scikit-learn` — `CountVectorizer` (learning modules)
- `pandas`, `numpy`, `matplotlib` — data handling and visualisation
- `rank-bm25` — BM25 index (Phase 4)
- `openpyxl` — query set export
