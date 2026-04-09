# Information Retrieval Project — OpenSanctions Entity Search

An end-to-end retrieval system for sanctions entity search over the
[OpenSanctions](https://www.opensanctions.org/) dataset. The supported local
workflow covers preprocessing, classical retrieval, dense retrieval, RRF fusion,
and notebook-based evaluation.

## Project Overview

| Phase | Status | Description |
| ----- | ------ | ----------- |
| 1 | Done | Data exploration and schema analysis |
| 2 | Done | Preprocessing pipeline and document building |
| 3 | Done | Query and qrels preparation for Types 1-6 |
| 4 | Implemented | BM25, TF-IDF, identifier retrieval, dense retrieval, and RRF |
| 5 | Implemented | Evaluation notebook plus CSV exports |
| 6 | Optional | Experimental RAG components |

## Supported Workflows

- Local reruns of the non-RAG pipeline using the canonical runbook in `docs/rerun_pipeline.md`
- Notebook-based evaluation via `notebooks/05_evaluation_types_1_6.ipynb`
- Learning materials under `learning/`
- Optional test and RAG installs via dedicated requirement files

Colab is not a supported workflow for this repository.

## Repository Structure

```text
IR_project/
├── data/
│   ├── raw_data/                   # Raw OpenSanctions inputs and local samples
│   ├── json_format_data/           # Canonical processed corpora
│   ├── queries/                    # Query JSON files
│   └── qrels/                      # Qrels JSON files
├── docs/
│   ├── rerun_pipeline.md               # Canonical rerun guide
│   ├── presentation_outline.md         # Presentation source
│   ├── presentation_slides_export.html # Presentation export
│   └── assignment_2_brief.pdf          # Assignment brief
├── notebooks/                          # Analysis and evaluation notebooks
├── scripts/                            # Pipeline entrypoints and export helpers
├── src/                                # Preprocessing, retrieval, fusion, evaluation, rag
├── tests/                              # Unit tests
├── requirements.txt                    # Core local pipeline + notebook dependencies
├── requirements-tests.txt              # Optional test dependencies
└── requirements-rag.txt                # Optional RAG dependencies
```

## Data

The full OpenSanctions dataset is not tracked in Git. See `data/DATA_SOURCE.md`
for download instructions and file placement.

### Supported raw inputs

- `data/raw_data/sample_10k.json` for quick local reruns
- `data/raw_data/sample_100k.json` for larger local validation runs
- `data/raw_data/targets.nested.json` for full-scale runs

### Canonical processed output

All reruns should write the shared processed corpus to:

- `data/json_format_data/subset/`

That keeps downstream indices, runs, and evaluation paths consistent across 10K,
100K, and full workflows.

## Quick Start

Use a Python `3.11+` virtual environment from the repository root.

```bash
git clone <repo-url>
cd IR_project
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Then follow `docs/rerun_pipeline.md` for the exact 10K, 100K, or full rerun
commands.

If you want the notebook to appear as its own kernel in Jupyter:

```bash
python -m ipykernel install --user --name ir_project_venv --display-name "Python (IR Project .venv)"
```

## Dependency Files

- `requirements.txt`: supported base environment for the local retrieval pipeline and notebooks
- `requirements-tests.txt`: optional test-only additions
- `requirements-rag.txt`: optional RAG additions

The base install intentionally covers notebooks as well as the CLI pipeline.
Tests and RAG are kept separate so the default environment stays focused on the
supported retrieval workflow.

## Runtime Notes

- `spaCy` requires the `en_core_web_sm` model. Install it explicitly with `python -m spacy download en_core_web_sm`.
- `nltk` stopword/tokenizer data is bootstrapped automatically on first use if missing.
- Dense retrieval can use CPU or CUDA depending on the installed PyTorch build.
- The evaluation notebook writes rendered output into `notebooks/05_evaluation_types_1_6.ipynb` and CSV summaries into `results/evaluation/`.

## Learning Modules

The `learning/` folder contains self-contained teaching materials built around
the first 100 rows of `data/raw_data/sample_10k.json`.

## Optional Components

- RAG code lives under `src/rag/`, is not part of the main rerun workflow, and
  is being kept for a later cleanup/review pass.
- Tests live under `tests/` and can be installed separately when needed.

## Canonical Runbook

For the exact step-by-step pipeline, use:

- `docs/rerun_pipeline.md`
