# Project task list

Task-level status for [implementation_phases.md](./implementation_phases.md). Update this file as work completes.

**Status legend:** `Done` · `In progress` · `Not started` · `Verify locally` (expected on disk but gitignored or not in repo)

---

## Phase 1: Data acquisition & exploration

| ID | Parent | Task | Status | Notes |
|----|--------|------|--------|-------|
| P1-01 | §1 | Download `targets.nested.json` (~3.66 GB) from OpenSanctions | Verify locally | Listed in `.gitignore` at `data/raw_data/targets.nested.json` |
| P1-02 | §1 | Download `targets.simple.csv` (~436 MB) as backup | Not started | Optional backup per plan |
| P1-03 | §1 | Store raw data under project data folder | In progress | Plan: `data/raw/`; repo uses `data/raw_data/` — align naming |
| P1-04 | §2 | Create exploratory notebook (`notebooks/01_data_exploration.ipynb`) | In progress | Equivalent: `notebooks/01_json_exploration.ipynb` |
| P1-05 | §2 | Understand data structure, entity types, fields | In progress | Covered partly in exploration notebook / `DATA_SOURCE.md` |
| P1-06 | §2 | Analyse field distributions (name, alias, previousName, description, …) | In progress | As per notebook completion |
| P1-07 | §2 | Identify data quality issues | In progress | As per notebook completion |
| P1-08 | §2 | Check language diversity and transliteration challenges | In progress | As per notebook completion |
| P1-09 | §3 | Document which fields exist and their coverage | In progress | `data/DATA_SOURCE.md` covers key fields; expand if needed |
| P1-10 | §3 | Document data quality issues (missing values, inconsistencies) | In progress | Tie to exploration outputs |
| P1-11 | §3 | Complexity assessment (nested structures, multiple identifiers) | In progress | Tie to exploration outputs |

---

## Phase 2: Preprocessing pipeline

| ID | Parent | Task | Status | Notes |
|----|--------|------|--------|-------|
| P2-01 | §4 | JSON streaming for large file without loading whole file | Done | `src/preprocessing/parser.py` |
| P2-02 | §4 | Use `json` and `pathlib` | Done | As implemented in parser |
| P2-03 | §4 | Parse line by line (line-delimited JSON) | Done | `parser.py` |
| P2-04 | §4 | Create `src/preprocessing/parser.py` | Done | |
| P2-05 | §5 | Lowercase normalization | Done | `text_processing.py` |
| P2-06 | §5 | Stopword removal via NLTK | Done | `text_processing.py` |
| P2-07 | §5 | Lemmatization via spaCy | Done | `text_processing.py` |
| P2-08 | §5 | Handle transliterated names (Arabic, Russian, Chinese) | In progress | Confirm coverage in code / tests |
| P2-09 | §5 | Create `src/preprocessing/text_processing.py` | Done | |
| P2-10 | §6 | Flatten nested JSON into searchable text blobs | Done | `document_builder.py` / `pipeline.py` |
| P2-11 | §6 | Concatenate name, alias, previousName, description, summary, programId | Done | As per document builder design |
| P2-12 | §6 | Preserve identifier fields separately (imoNumber, mmsi) | Done | As per document builder |
| P2-13 | §6 | Store metadata (schema, country, programId) for filtering | Done | As per document builder |
| P2-14 | §6 | Create flattening module (`indexer.py` in plan) | Done | Implemented as `document_builder.py` (+ `pipeline.py`), not `indexer.py` |
| P2-15 | §7 | Extract first 100K entities for development | Not started | No `data/processed/subset_100k/` in repo |
| P2-16 | §7 | Validate pipeline correctness on subset | In progress | `notebooks/02_preprocessing_validation.ipynb` |
| P2-17 | §7 | Measure processing speed | In progress | As per validation notebook |
| P2-18 | §7 | Store subset under `data/processed/subset_100k/` | Not started | |

---

## Phase 3: Query set development

| ID | Parent | Task | Status | Notes |
|----|--------|------|--------|-------|
| P3-01 | §8 | Create query brainstorming notebook | In progress | `notebooks/03_query_generation.ipynb` (plan: `query_generation.ipynb`) |
| P3-02 | §8 | Type 1: exact identifier queries (2–3) | In progress | Part of query workbook / Excel |
| P3-03 | §8 | Type 2: name / alias queries (2–3) | In progress | |
| P3-04 | §8 | Type 3: semantic / descriptive queries (2–3) | In progress | |
| P3-05 | §8 | Type 4: relational / graph queries (2–3) | In progress | |
| P3-06 | §8 | Type 5: cross-dataset deduplication queries (2–3) | In progress | |
| P3-07 | §8 | Type 6: jurisdiction / filter queries (2–3) | In progress | |
| P3-08 | §8 | Type 7: RAG summarisation queries (2–3) | In progress | |
| P3-09 | §8 | Ensure 20+ candidates; realistic operational needs | In progress | |
| P3-10 | §9 | Excel columns: query_id, query_type, query_text, expected_difficulty, notes | In progress | `queries_part_a.xlsx`, template for part B |
| P3-11 | §9 | Save draft as `data/evaluation/queries_draft.xlsx` | In progress | Current naming differs; align or copy to planned name |
| P3-12 | §10 | Share workbook with domain expert | Not started | |
| P3-13 | §10 | Validate queries realistic and operationally relevant | Not started | |
| P3-14 | §10 | Expert annotates difficulty and expected results | Not started | |
| P3-15 | §10 | Save reviewed file as `data/evaluation/queries_reviewed.xlsx` | Not started | |
| P3-16 | §11 | Create `scripts/excel_to_json.py` | Done | |
| P3-17 | §11 | Convert reviewed Excel → `data/evaluation/queries.json` | Not started | Run script after review; file not in repo yet |
| P3-18 | §11 | JSON structure matches plan (`queries` array with required fields) | Not started | Validated when `queries.json` exists |

---

## Phase 4: Ground truth construction

| ID | Parent | Task | Status | Notes |
|----|--------|------|--------|-------|
| P4-01 | §12 | Type 1: IMO → entities with matching `imoNumber` | Not started | Part of automatic qrels |
| P4-02 | §12 | Type 2: name/alias match with fuzzy threshold | Not started | |
| P4-03 | §12 | Type 5: same entity across schema / canonical id | Not started | |
| P4-04 | §12 | Type 6: filter by programId and schema | Not started | |
| P4-05 | §12 | Create `scripts/generate_ground_truth.py` | Not started | |
| P4-06 | §12 | Output `data/evaluation/qrels.json` (TREC-style) | Not started | |
| P4-07 | §13 | Define pooling for Types 3 and 4 (no objective GT) | Not started | |
| P4-08 | §13 | Methodology: BM25, TF-IDF, dense → top-20 each → pool ≤60 | Not started | Depends on Phase 5 systems |
| P4-09 | §13 | Manual judgement scale 0 / 1 / 2 | Not started | |
| P4-10 | §13 | Create `notebooks/pooling_workflow.ipynb` | Not started | |
| P4-11 | §13 | Treat documents outside pool as non-relevant | Not started | |

---

## Phase 5: Indexing and retrieval

| ID | Parent | Task | Status | Notes |
|----|--------|------|--------|-------|
| P5-01 | §14 | BM25 with `rank-bm25` on preprocessed blobs | Not started | |
| P5-02 | §14 | Test BM25 on 100K subset first | Not started | |
| P5-03 | §14 | Tune BM25 k1, b | Not started | |
| P5-04 | §14 | TF-IDF + cosine baseline (`scikit-learn`) | Not started | |
| P5-05 | §14 | Fuzzy / name expansion with `rapidfuzz` | Not started | |
| P5-06 | §14 | Transliteration-aware expansion | Not started | |
| P5-07 | §14 | Create `src/retrieval/classical_ir.py` | Not started | |
| P5-08 | §14 | Save BM25 index under `models/bm25_index/` | Not started | |
| P5-09 | §15 | Sentence embeddings all-MiniLM-L6-v2 (384-d) | Not started | |
| P5-10 | §15 | Encode blobs; store in ChromaDB | Not started | |
| P5-11 | §15 | Index 1.3M or 100K; cosine similarity query | Not started | |
| P5-12 | §15 | Optional: faiss-cpu comparison | Not started | |
| P5-13 | §15 | Create `src/retrieval/dense_retrieval.py` | Not started | |
| P5-14 | §15 | Save vector store under `models/chromadb/` | Not started | |
| P5-15 | §16 | Run classical and dense systems on query set | Not started | |
| P5-16 | §16 | Produce ranked lists per query | Not started | |
| P5-17 | §16 | Store TREC-style runs in `results/retrieval_runs/` | Not started | |

---

## Phase 6: Evaluation framework

| ID | Parent | Task | Status | Notes |
|----|--------|------|--------|-------|
| P6-01 | §17 | Integrate `ranx` for metrics | Not started | Dependency in `requirements.txt` |
| P6-02 | §17 | Precision@1 (Types 1–2 focus) | Not started | |
| P6-03 | §17 | Recall@10 (primary) | Not started | |
| P6-04 | §17 | Recall@20 | Not started | |
| P6-05 | §17 | MAP | Not started | |
| P6-06 | §17 | nDCG@10 for graded (Types 3–4) | Not started | |
| P6-07 | §17 | Create `src/evaluation/metrics.py` | Not started | |
| P6-08 | §18 | Merge top-20 from BM25, TF-IDF, dense for pooling | Not started | |
| P6-09 | §18 | Pool ≤60 unique docs per query | Not started | |
| P6-10 | §18 | Expert judgement session (0 / 1 / 2) | Not started | |
| P6-11 | §18 | Update `qrels.json` with pooled judgements | Not started | |
| P6-12 | §19 | Evaluate on 100K subset | Not started | |
| P6-13 | §19 | Baseline per query type; note failing queries | Not started | |
| P6-14 | §19 | Write `results/metrics/baseline_100k.json` | Not started | |
| P6-15 | §19 | Create `notebooks/02_classical_ir_experiments.ipynb` | Not started | Avoid clash with existing `02_preprocessing_validation.ipynb` naming |

---

## Phase 7: Fusion and RAG

| ID | Parent | Task | Status | Notes |
|----|--------|------|--------|-------|
| P7-01 | §20 | Implement RRF | Not started | |
| P7-02 | §20 | RRF formula with k=60 (BM25 + dense ranks) | Not started | |
| P7-03 | §20 | Optional cross-encoder reranking | Not started | |
| P7-04 | §20 | Create `src/retrieval/fusion.py` | Not started | |
| P7-05 | §21 | Type 7 pipeline: top-k fused → context | Not started | |
| P7-06 | §21 | flan-t5-base via `transformers` | Not started | |
| P7-07 | §21 | Prompted summarisation from retrieved records | Not started | |
| P7-08 | §21 | LangChain orchestration | Not started | |
| P7-09 | §21 | Ground answers in retrieved context | Not started | |
| P7-10 | §21 | Create `src/rag/generator.py` | Not started | |
| P7-11 | §22 | RAGAS faithfulness | Not started | |
| P7-12 | §22 | RAGAS answer relevance | Not started | |
| P7-13 | §22 | Expert qualitative review | Not started | |
| P7-14 | §22 | Create `src/evaluation/ragas_eval.py` | Not started | |

---

## Phase 8: Scale and optimise

| ID | Parent | Task | Status | Notes |
|----|--------|------|--------|-------|
| P8-01 | §23 | Run preprocessing on full `targets.nested.json` | Not started | |
| P8-02 | §23 | Full BM25 index (~1.3M) | Not started | |
| P8-03 | §23 | Full ChromaDB / embeddings | Not started | |
| P8-04 | §23 | Record indexing time and storage | Not started | |
| P8-05 | §23 | Store artefacts under `models/` | Not started | |
| P8-06 | §24 | BM25 grid: k1 ∈ {1.2, 1.5, 2.0}, b ∈ {0.5, 0.75, 1.0} | Not started | |
| P8-07 | §24 | RRF k sensitivity | Not started | |
| P8-08 | §24 | Compare embedding models | Not started | |
| P8-09 | §24 | Validation / CV; write `results/hyperparameters.json` | Not started | |
| P8-10 | §25 | Full evaluation on all ~20 queries, types 1–7 | Not started | |
| P8-11 | §25 | All planned metrics | Not started | |
| P8-12 | §25 | Compare BM25, TF-IDF, dense, hybrid (RRF) | Not started | |
| P8-13 | §25 | `results/metrics/final_evaluation.json` | Not started | |
| P8-14 | §25 | Plots in `results/plots/` | Not started | |
| P8-15 | §25 | `notebooks/04_evaluation_analysis.ipynb` | Not started | |

---

## Phase 9: Report

| ID | Parent | Task | Status | Notes |
|----|--------|------|--------|-------|
| P9-01 | §26 | Analyse what worked (e.g. BM25 vs dense) | Not started | |
| P9-02 | §26 | Analyse what failed (relational, transliteration, …) | Not started | |
| P9-03 | §26 | Limitations and failure cases | Not started | |
| P9-04 | §26 | Comparison to related work (BERT, ColBERT, …) | Not started | |
| P9-05 | §27 | Report: Introduction | Not started | |
| P9-06 | §27 | Report: System architecture | Not started | |
| P9-07 | §27 | Report: Implementation details | Not started | |
| P9-08 | §27 | Report: Evaluation setup | Not started | |
| P9-09 | §27 | Report: Results and analysis | Not started | |
| P9-10 | §27 | Report: Discussion and future work | Not started | |
| P9-11 | §27 | Report: Conclusion | Not started | |
| P9-12 | §27 | Figures, tables, example queries | Not started | |
| P9-13 | §27 | Submit report + code repository | Not started | |

---

## Key deliverables checklist (from implementation_phases.md)

| ID | Deliverable | Status | Notes |
|----|-------------|--------|-------|
| D-01 | `data/raw/targets.nested.json` (or project-equivalent path) downloaded | Verify locally | Repo path: `data/raw_data/` + gitignore |
| D-02 | `notebooks/01_data_exploration.ipynb` completed | In progress | See `01_json_exploration.ipynb` |
| D-03 | Preprocessing pipeline `src/preprocessing/` | Done | |
| D-04 | `data/evaluation/queries.json` (≈20 queries) | Not started | |
| D-05 | `data/evaluation/qrels.json` | Not started | |
| D-06 | `src/retrieval/classical_ir.py` | Not started | |
| D-07 | `src/retrieval/dense_retrieval.py` | Not started | |
| D-08 | `src/evaluation/` framework | Not started | |
| D-09 | `src/retrieval/fusion.py` | Not started | |
| D-10 | `src/rag/generator.py` | Not started | |
| D-11 | Full evaluation on 1.3M dataset | Not started | |
| D-12 | Results analysed and visualised | Not started | |
| D-13 | Assignment 2 report submitted | Not started | |

---

## Phase 10: Presentation

| ID | Parent | Task | Status | Notes |
|----|--------|------|--------|-------|
| P10-01 | Presentation | Content generation (talk outline, key results, demo script) | Not started | |
| P10-02 | Presentation | Presentation adding (build slides / deck in chosen tool) | Not started | |

---

## How to maintain this file

1. After each milestone, change **Status** for the affected rows (and add **Notes** if paths or owners differ from the plan).
2. Keep **Verify locally** until the artefact is confirmed on disk or committed.
3. When the implementation doc is updated, add or adjust rows here to stay aligned.

Last reviewed: 2026-03-28 (initial creation; statuses reflect repository snapshot at that date).
