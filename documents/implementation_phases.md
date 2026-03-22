# Implementation Phases

## Overview
This document outlines the step-by-step implementation plan for the IR Search Engine project (Assignment 2, Weeks 7-11).

---

## Phase 1: Data Acquisition & Exploration

### 1. Download OpenSanctions Data
- Download `targets.nested.json` (3.66 GB) from OpenSanctions
- Download `targets.simple.csv` (435.95 MB) as backup
- Store in `data/raw/`

### 2. Exploratory Notebook
- Create `notebooks/01_data_exploration.ipynb`
- Understand data structure, entity types, fields
- Analyze field distributions (name, alias, previousName, description, etc.)
- Identify data quality issues
- Check language diversity and transliteration challenges

### 3. Document Findings
- What fields exist and their coverage
- Data quality issues (missing values, inconsistencies)
- Complexity assessment (nested structures, multiple identifiers)

---

## Phase 2: Preprocessing Pipeline (Week 1 - Kieren)

### 4. Build Streaming Parser
- Implement JSON streaming to handle 3.66GB without memory overflow
- Use `json` and `pathlib` libraries
- Parse line by line to avoid loading entire file
- Create `src/preprocessing/parser.py`

### 5. Text Preprocessing
- Lowercase normalization
- Stopword removal via `nltk`
- Lemmatization via `spaCy`
- Handle transliterated names (Arabic, Russian, Chinese)
- Create `src/preprocessing/text_processing.py`

### 6. Document Flattening
- Flatten nested JSON into searchable text blobs
- Concatenate: `name`, `alias`, `previousName`, `description`, `summary`, `programId`
- Preserve identifier fields separately (`imoNumber`, `mmsi`)
- Store metadata (`schema`, `country`, `programId`) for filtering
- Create `src/preprocessing/indexer.py`

### 7. Test on 100K Subset
- Extract first 100K entities for development
- Validate pipeline correctness
- Measure processing speed
- Store in `data/processed/subset_100k/`

---

## Phase 3: Query Set Development (Week 1 - Marek, Alireza)

### 8. Query Brainstorming Notebook
- Create `notebooks/query_generation.ipynb`
- Generate 20+ candidate queries across all 7 types:
  - **Type 1: Exact Identifier** - IMO, MMSI, call sign (2-3 queries)
    - Example: `IM09296822`
  - **Type 2: Name/Alias** - Current name, alias, historical name with variations (2-3 queries)
    - Example: `Angelica Schulte`
  - **Type 3: Semantic/Descriptive** - Natural language, no exact name match (2-3 queries)
    - Example: `Russian crude oil tanker evading sanctions Baltic Sea`
  - **Type 4: Relational/Graph** - Links between entities (2-3 queries)
    - Example: `What vessels is Ketee Co Ltd linked to?`
  - **Type 5: Cross-Dataset Deduplication** - Same entity across sources (2-3 queries)
    - Example: `SAGITTA sanctions all sources`
  - **Type 6: Jurisdiction/Filter** - Scoped searches (2-3 queries)
    - Example: `OFAC sanctioned vessels Russian oil 2025`
  - **Type 7: RAG Summarisation** - Narrative generation (2-3 queries)
    - Example: `Summarise all sanctions on the SAGITTA vessel`
- Ensure queries reflect realistic operational needs

### 9. Export to Excel
- Create Excel file with columns:
  - `query_id` (e.g., Q001, Q002, ...)
  - `query_type` (1-7)
  - `query_text` (the actual query string)
  - `expected_difficulty` (easy/medium/hard)
  - `notes` (any contextual information)
- Save as `data/evaluation/queries_draft.xlsx`

### 10. Expert Review
- Share Excel file with domain expert (financial crime, sanctions risk)
- Validate queries are realistic and operationally relevant
- Expert annotates difficulty and expected results
- Save reviewed version as `data/evaluation/queries_reviewed.xlsx`

### 11. Import Reviewed Queries
- Create script `scripts/excel_to_json.py`
- Convert `queries_reviewed.xlsx` → `queries.json`
- JSON format:
```json
{
  "queries": [
    {
      "query_id": "Q001",
      "query_type": 1,
      "query_text": "IM09296822",
      "expected_difficulty": "easy",
      "notes": "Direct IMO lookup"
    }
  ]
}
```

---

## Phase 4: Ground Truth Construction (Week 1-2)

### 12. Automatic Ground Truth (Types 1, 2, 5, 6)
- **Type 1 (Exact Identifier)**: Direct field match in dataset
  - Query has IMO number → extract entity with matching `imoNumber`
  - No human judgement required
- **Type 2 (Name/Alias)**: Field match with variations
  - Query matches `name`, `alias`, or `previousName` (exact or fuzzy)
  - Define as relevant if Levenshtein distance < threshold
- **Type 5 (Cross-Dataset)**: Same entity across `schema` field
  - Group by canonical identifier
- **Type 6 (Jurisdiction)**: Filter by `programId` and `schema`
  - Extract subset matching jurisdiction

- Create `scripts/generate_ground_truth.py`
- Output: `data/evaluation/qrels.json` (TREC format)
```json
{
  "Q001": {
    "entity_id_123": 1,
    "entity_id_456": 1
  }
}
```

### 13. Pooling Preparation (Types 3, 4)
- **Type 3 (Semantic)** and **Type 4 (Relational)**: No objective ground truth
- Set up pooling methodology:
  - Run all retrieval systems (BM25, TF-IDF, dense) independently
  - Merge top-20 results from each system
  - Create pool of up to 60 unique candidate documents per query
  - Manual judgement on 3-point scale: 0 (not relevant), 1 (somewhat relevant), 2 (highly relevant)
- Create `notebooks/pooling_workflow.ipynb`
- Documents outside pool assumed non-relevant

---

## Phase 5: Indexing & Retrieval (Week 1-2)

### 14. Classical IR Indexing (Alireza - Week 1)
- **BM25**: Primary ranker
  - Use `rank-bm25` library
  - Index preprocessed text blobs
  - Test on 100K subset first
  - Tune parameters: k1, b
- **TF-IDF with Cosine Normalization**: Baseline
  - Use `scikit-learn`
  - Serves as comparative baseline
- **Fuzzy Matching**: Name variant expansion
  - Use `rapidfuzz` for pre-retrieval name expansion
  - Handle transliterations (Arabic, Russian, Chinese)
- Create `src/retrieval/classical_ir.py`
- Save indices in `models/bm25_index/`

### 15. Dense Retrieval Indexing (Marek - Week 2)
- **Sentence Embeddings**: all-MiniLM-L6-v2 (384-dim)
  - Use `sentence-transformers`
  - Encode all preprocessed text blobs
  - Store in ChromaDB vector database
- **ChromaDB**: Vector storage & nearest-neighbour search
  - Index all 1.3M entities (or 100K for initial testing)
  - Query via cosine similarity
- **Alternative**: faiss-cpu for comparison
- Create `src/retrieval/dense_retrieval.py`
- Save database in `models/chromadb/`

### 16. Test on Queries
- Run both systems independently against query set
- Generate ranked lists for each query
- Store results in `results/retrieval_runs/`
- Format: TREC run format (query_id, entity_id, rank, score)

---

## Phase 6: Evaluation Framework (Week 2)

### 17. Implement Metrics
- Use `ranx` library for standard IR metrics
- **Precision@1**: For exact identifier queries (Types 1, 2)
- **Recall@10**: Primary metric (regulatory cost focus)
- **Recall@20**: Secondary for pooling depth
- **MAP (Mean Average Precision)**: Aggregate across query types
- **nDCG@10**: For graded relevance judgements (Types 3, 4)
- Create `src/evaluation/metrics.py`

### 18. Pooling for Types 3, 4
- Merge top-20 from BM25, TF-IDF, dense retrieval
- Create pool of unique documents (max 60 per query)
- Manual judgement session:
  - Domain expert reviews each document
  - 3-point scale: 0 (not relevant), 1 (somewhat), 2 (highly)
- Update `qrels.json` with judgements
- Documents outside pool = 0 (assumed non-relevant)

### 19. Initial Evaluation
- Run evaluation on 100K subset
- Measure baseline performance for each query type
- Identify weaknesses (which queries fail?)
- Generate evaluation report: `results/metrics/baseline_100k.json`
- Create `notebooks/02_classical_ir_experiments.ipynb`

---

## Phase 7: Fusion & RAG (Week 3)

### 20. RRF Fusion (Marek)
- Implement Reciprocal Rank Fusion (RRF)
- Formula: RRF(d) = 1/(k + rank_BM25) + 1/(k + rank_dense), k=60
- Merge ranked lists from classical IR and dense retrieval
- Avoids score normalization issues
- Optional: Add cross-encoder reranking as second stage
- Create `src/retrieval/fusion.py`

### 21. RAG Layer (Marek)
- Implement for Type 7 (RAG Summarisation) queries
- Pipeline:
  1. Retrieve top-k fused entities (e.g., k=10)
  2. Pass to `flan-t5-base` via `transformers`
  3. Prompt: "Summarise the sanctions on [entity name] based on the following records: [context]"
  4. Generate narrative summary
- Use `langchain` for pipeline orchestration
- Constrain generation to retrieved context (prevent hallucination)
- Create `src/rag/generator.py`

### 22. RAGAS Evaluation (Marek)
- Evaluate RAG outputs on two dimensions:
  - **Faithfulness**: Is summary grounded in retrieved documents?
  - **Answer Relevance**: Does it address the query?
- Qualitative review by domain expert
- Create `src/evaluation/ragas_eval.py`

---

## Phase 8: Scale & Optimize (Week 3)

### 23. Scale to Full 1.3M Dataset (Kieren)
- Run preprocessing pipeline on full `targets.nested.json`
- Build full-scale indices:
  - BM25 on all 1.3M entities
  - ChromaDB with all 1.3M embeddings
- Measure indexing time and storage requirements
- Store in `models/`

### 24. Hyperparameter Tuning
- **BM25 parameters**: Grid search over k1, b
  - k1: [1.2, 1.5, 2.0]
  - b: [0.5, 0.75, 1.0]
- **Fusion weight**: Test RRF with different k values
- **Embedding model**: Compare all-MiniLM-L6-v2 vs alternatives
- Use validation set or cross-validation
- Document best parameters in `results/hyperparameters.json`

### 25. Final Evaluation
- Run full evaluation on all 20 queries
- All query types (1-7)
- All metrics (Precision@1, Recall@10, Recall@20, MAP, nDCG@10)
- Compare:
  - BM25 only
  - TF-IDF only
  - Dense retrieval only
  - Hybrid (RRF fusion)
- Generate final results: `results/metrics/final_evaluation.json`
- Create visualizations: `results/plots/`
- Create `notebooks/04_evaluation_analysis.ipynb`

---

## Phase 9: Report (Week 4)

### 26. Results Analysis
- What worked well? (e.g., BM25 for exact matches, dense for semantic)
- What didn't work? (e.g., relational queries, name transliterations)
- System limitations and failure cases
- Comparison to related work (BERT, ColBERT, etc.)

### 27. Write Assignment 2 Report
- Follow TREC methodology structure
- Sections:
  1. Introduction
  2. System Architecture (expand from Assignment 1)
  3. Implementation Details
  4. Evaluation Setup
  5. Results & Analysis
  6. Discussion & Future Work
  7. Conclusion
- Include figures, tables, example queries
- Submit report + code repository

---

## Key Deliverables Checklist

- [ ] `data/raw/targets.nested.json` downloaded
- [ ] `notebooks/01_data_exploration.ipynb` completed
- [ ] Preprocessing pipeline (`src/preprocessing/`) implemented
- [ ] `data/evaluation/queries.json` finalized (20 queries)
- [ ] `data/evaluation/qrels.json` ground truth constructed
- [ ] Classical IR system (`src/retrieval/classical_ir.py`) implemented
- [ ] Dense retrieval system (`src/retrieval/dense_retrieval.py`) implemented
- [ ] Evaluation framework (`src/evaluation/`) implemented
- [ ] RRF fusion (`src/retrieval/fusion.py`) implemented
- [ ] RAG layer (`src/rag/generator.py`) implemented
- [ ] Full evaluation completed on 1.3M dataset
- [ ] Results analyzed and visualized
- [ ] Assignment 2 report written and submitted

---

## Notes

- **Week 1 focus**: Data pipeline + Classical IR on 100K subset
- **Week 2 focus**: Dense retrieval + Evaluation framework + Pooling
- **Week 3 focus**: Fusion + RAG + Scale to 1.3M + Final evaluation
- **Week 4 focus**: Report writing and optimization

- **Team coordination**: Use git branches for parallel development
- **Code review**: Peer review before merging to main
- **Documentation**: Document all design decisions in notebooks
