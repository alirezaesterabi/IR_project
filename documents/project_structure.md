# Project Structure

## Overview
This document outlines the recommended directory structure for the IR Search Engine project for sanctions entity search over the OpenSanctions Default dataset.

## Directory Structure

```
IR_project/
├── README.md                          # Project overview and setup instructions
├── .gitignore                         # Ignore data files, models, venv, etc.
├── requirements.txt                   # Python dependencies
├── documents/                         # Project documentation
│   ├── assignment_1.pdf              # Assignment specification
│   └── project_structure.md          # This file
├── data/                             # Data directory (gitignored)
│   ├── raw/                          # Original OpenSanctions data
│   │   ├── targets.nested.json       # 3.66 GB nested JSON format
│   │   └── targets.simple.csv        # 435.95 MB simple CSV format
│   ├── processed/                    # Preprocessed data
│   │   ├── flattened_documents/      # Flattened entity documents
│   │   └── metadata/                 # Schema, country, programId metadata
│   └── evaluation/                   # Test queries and ground truth
│       ├── queries.json              # 20 test queries (7 types)
│       ├── qrels.json                # Relevance judgements
│       └── pooling_results/          # Pooling method outputs
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                     # Configuration settings
│   ├── preprocessing/                # Data ingestion & preprocessing (Kieren)
│   │   ├── __init__.py
│   │   ├── parser.py                 # JSON streaming parser (json, pathlib)
│   │   ├── text_processing.py       # Lemmatization (spaCy), stopwords (nltk)
│   │   └── indexer.py               # Document flattening and blob construction
│   ├── retrieval/                    # Retrieval components
│   │   ├── __init__.py
│   │   ├── classical_ir.py          # BM25 (rank-bm25), TF-IDF (scikit-learn), fuzzy (rapidfuzz) - Alireza
│   │   ├── dense_retrieval.py       # Sentence embeddings (sentence-transformers), ChromaDB - Marek
│   │   └── fusion.py                # RRF rank fusion (numpy) - Marek
│   ├── rag/                          # RAG layer (Marek)
│   │   ├── __init__.py
│   │   └── generator.py             # flan-t5-base summarization (transformers, langchain)
│   ├── evaluation/                   # Evaluation framework (Marek, Alireza)
│   │   ├── __init__.py
│   │   ├── metrics.py               # Precision@K, Recall@K, MAP, nDCG (ranx)
│   │   └── ragas_eval.py            # RAGAS evaluation for RAG outputs
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       └── helpers.py               # Common helper functions
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb    # Initial data analysis
│   ├── 02_classical_ir_experiments.ipynb  # BM25, TF-IDF tuning
│   ├── 03_dense_retrieval_experiments.ipynb  # Sentence embeddings exploration
│   └── 04_evaluation_analysis.ipynb # Results visualization
├── scripts/                          # Standalone scripts
│   ├── download_data.sh             # Download OpenSanctions dataset
│   ├── build_index.py               # Build search indices
│   └── run_evaluation.py            # Run evaluation pipeline
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_retrieval.py
│   └── test_evaluation.py
├── models/                           # Saved models (gitignored)
│   ├── bm25_index/                  # Serialized BM25 index
│   └── chromadb/                    # ChromaDB vector database
└── results/                          # Evaluation results (gitignored)
    ├── metrics/                      # Evaluation metrics (JSON/CSV)
    └── plots/                        # Visualizations (PNG/PDF)
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
- All experiments documented in notebooks
- Scripts for building indices and running evaluation
- `requirements.txt` for dependency management
- Configuration centralized in `src/config.py`

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
- **sentence-transformers**: all-MiniLM-L6-v2 embeddings (384-dim)
- **chromadb**: Vector storage & nearest-neighbour search
- **faiss-cpu**: Alternative dense search backend

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
- Week 4 is mainly for report writing and optimization.
