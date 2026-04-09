# Project Structure

## Overview

This document outlines the recommended directory structure for the IR Search Engine project for sanctions entity search over the OpenSanctions Default dataset.

## Directory Structure

```text
IR_project/
├── README.md                          # Project overview and setup instructions
├── .gitignore                         # Ignore large artefacts, local models, results
├── requirements.txt                   # Python dependencies
├── documents/
│   ├── assignment_1.pdf              # Assignment specification
│   └── project_structure.md          # This file
├── data/
│   ├── raw_data/                     # OpenSanctions source data + sample
│   ├── json_format_data/             # Processed corpora (subset/full)
│   ├── queries/                      # queries_type_1.json ... queries_type_6.json
│   ├── qrels/                        # qrels_type_1.json ... qrels_type_6.json
│   ├── pooling/                      # Pooling spreadsheets / manual judging inputs
│   └── toy_example/                  # Small teaching corpus
├── src/
│   ├── preprocessing/
│   │   ├── parser.py                 # Streaming parser
│   │   ├── document_builder.py       # Flatten structured OpenSanctions records
│   │   ├── embedding_text.py         # Dense-friendly text representation
│   │   ├── text_processing.py        # Tokenisation / lemmatisation
│   │   └── pipeline.py               # End-to-end preprocessing entrypoint
│   ├── retrieval/
│   │   ├── classical_ir.py          # BM25, TF-IDF, Identifier retrievers
│   │   ├── dense_config.py          # Dense model registry + RUN_TAG helpers
│   │   └── dense_retriever.py       # Query-time SentenceTransformer + Chroma retriever
│   ├── fusion/
│   │   └── rrf.py                   # Reciprocal Rank Fusion
│   └── evaluation/
│       └── utils.py                 # ranx loaders, metrics, comparison tables
├── scripts/
│   ├── build_index.py               # Build BM25 / TF-IDF / Identifier indices
│   ├── build_dense_embeddings.py    # Build dense embeddings + optional Chroma upsert
│   ├── encode_dense_minilm.py       # Convenience wrapper for MiniLM encoding
│   ├── encode_dense_bge_m3.py       # Convenience wrapper for BGE-M3 encoding
│   ├── export_bm25_run.py           # Export lexical run CSV for Types 1-6
│   ├── export_dense_run.py          # Export dense run CSV for MiniLM / BGE-M3
│   └── evaluate_runs.py             # Evaluate BM25, dense, and fused runs
├── notebooks/                        # Exploration, pooling, and analysis notebooks
├── tests/                            # Unit tests for dense config, fusion, evaluation, etc.
├── models/                           # Saved lexical / dense artefacts (gitignored)
├── chroma_db/                        # Persistent Chroma collections (gitignored)
└── results/                          # Exported runs and evaluation CSVs (gitignored)
```

## Design Principles

### 1. Modular by Component

- Each major system component has its own directory
- Clear separation between Classical IR, Dense Retrieval, RAG, and Evaluation
- Enables parallel development by team members

### 2. Team Collaboration

- **Kieren**: Data Ingestion & Preprocessing, Classical IR evaluation
- **Alireza**: Classical IR implementation, Evaluation framework
- **Marek**: Dense Retrieval, Fusion layer, RAG, Evaluation (RAGAS)

### 3. Data Management

- Large data files excluded from git (see .gitignore)
- Raw data downloaded via script (not committed)
- Processed data cached locally
- Use `scripts/download_data.sh` to fetch OpenSanctions dataset

### 4. Reproducibility

- Dense model configuration is centralized in `src/retrieval/dense_config.py`
- Query-time dense retrieval is centralized in `src/retrieval/dense_retriever.py`
- Scripts exist for building indices, exporting runs, and evaluating them
- `requirements.txt` defines the shared dependency set

### 5. Testing

- Unit tests for each component
- Integration tests for the full pipeline
- Evaluation framework for retrieval quality

## Key Technologies

### Data Processing

- **json, pathlib**: JSON streaming to avoid memory overflow
- **nltk**: Stopword removal
- **spaCy**: Lemmatization
- **pandas, numpy**: Data handling

### Classical IR

- **rank-bm25**: BM25 ranking
- **scikit-learn**: TF-IDF with cosine normalization
- **rapidfuzz**: Fuzzy matching for name variants

### Dense Retrieval

- **sentence-transformers**: `all-MiniLM-L6-v2` and `BAAI/bge-m3`
- **chromadb**: Vector storage and nearest-neighbour search
- Dense runs are exported with a model-agnostic CSV contract for downstream RRF and evaluation

### RAG

- **transformers**: flan-t5-base for text generation
- **langchain**: RAG pipeline orchestration

### Evaluation

- **ranx**: Precision@K, Recall@K, MAP, nDCG computation

## Query Types

1. **Exact Identifier**: IMO, MMSI, call sign, registration number
2. **Name/Alias**: Current name, alias, or historical name with variations
3. **Semantic/Descriptive**: Natural language queries without exact matches
4. **Relational/Graph**: Links between entities (ownership, sanctions graph)
5. **Cross-Dataset Deduplication**: Same entity across multiple sources
6. **Jurisdiction/Filter**: Scoped searches (authority, programme ID, country)
7. **RAG Summarisation**: Narrative generation from multiple linked records

## Evaluation Metrics

- **Primary**: Recall@10 (regulatory cost of missing sanctioned entities)
- **Secondary**: Precision@1, MAP, nDCG@10
- **RAG-specific**: RAGAS (faithfulness, answer relevance)

## Timeline (Weeks 7-11)

### Week 1

- Blob construction, lemmatization pipeline (Kieren)
- BM25, TF-IDF, fuzzy matching on 100K subset (Alireza)

### Week 2

- Evaluation framework, BM25 tuning (Alireza)
- Sentence embeddings, ChromaDB setup (Marek)

### Week 3

- Scale to full 1.3M dataset (Kieren)
- RRF fusion, hybrid evaluation (Kieren, Marek)
- Pooling ground truth types 3,4; RAGAS (Marek)
- flan-t5-base, prompt design, LangChain (Marek)

## Notes

- This is Assignment 1 (design phase). Implementation happens in Assignment 2.
- Libraries and tools listed are the team's current best assessment and may change.
- Week 4 is mainly for report writing and optimisation.
