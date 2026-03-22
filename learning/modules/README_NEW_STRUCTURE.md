# IR Learning Modules - Comprehensive Structure

**Based on: comprehensive_ir_learning_plan.md**

This README maps out the NEW module structure aligned with the 10-week comprehensive learning plan.

---

## 🎯 Structure Overview

Each week follows this pattern:

```
Week_N_[Topic]/
├── theory.md           # Theory + Math + Slides content
├── toy_data/          # Small curated dataset for examples
│   ├── documents.json
│   ├── queries.txt
│   └── relevance.json (for evaluation weeks)
└── example.ipynb      # Theory in action + Lab integration
```

---

## 📚 Module Organization

### **Phase 1: Foundations (Weeks 1-3)**

#### Week 1a: Text Processing & Indexing Basics
**Folder:** `week_1a_text_processing/`
- **theory.md**:
  - Tokenization (Unicode, language-specific rules)
  - Normalization (case folding, accents)
  - Stemming (Porter algorithm - all 5 steps)
  - Lemmatization (WordNet, spaCy)
  - Morphological analysis
  - Statistical laws: Zipf's Law, Heap's Law
  - Manning Ch 2

- **example.ipynb**:
  - Implement tokenizer from scratch
  - Porter stemmer step-by-step
  - Compare stemming vs lemmatization
  - Visualize Zipf's Law
  - Calculate Heap's Law parameters
  - Lab 1 integration: Python text processing

- **toy_data**:
  - 10 simple documents (CS topics)
  - Mix of simple/complex words for stemming practice

---

#### Week 1b: Inverted Index Construction
**Folder:** `week_1b_inverted_index/`
- **theory.md**:
  - Term-document incidence matrix
  - Inverted index structure (dictionary + postings)
  - BSBI (Blocked Sort-Based Indexing)
  - SPIMI (Single-Pass In-Memory Indexing)
  - MapReduce for distributed indexing
  - Manning Ch 1, Ch 4

- **example.ipynb**:
  - Build inverted index from scratch (in-memory)
  - Implement BSBI step-by-step
  - Implement SPIMI
  - Compare memory usage
  - Time complexity analysis
  - Lab 1 integration: Indexing exercises

- **toy_data**:
  - Same 10 documents as Week 1a
  - Track index construction metrics

---

#### Week 2a: Boolean Retrieval & Query Processing
**Folder:** `week_2a_boolean_retrieval/`
- **theory.md**:
  - Boolean queries (AND, OR, NOT)
  - Postings list intersection (merge algorithm)
  - Query optimization (term ordering)
  - Skip pointers (√n optimal spacing proof)
  - Complexity analysis: O(x+y)
  - Manning Ch 1

- **example.ipynb**:
  - Implement postings list intersection
  - Add skip pointers
  - Benchmark with/without skip pointers
  - Query optimization examples
  - Boolean query evaluation
  - Lab integration: Query processing exercises

- **toy_data**:
  - 10 documents + 5-10 Boolean queries
  - Manual verification of results

---

#### Week 2b: Phrase Queries & Positional Index
**Folder:** `week_2b_phrase_queries/`
- **theory.md**:
  - Positional index structure
  - Biword index
  - Positional intersection algorithm
  - Space-time tradeoffs
  - Manning Ch 2 (sections 2.4)

- **example.ipynb**:
  - Build positional index
  - Implement phrase query processing
  - Compare biword vs positional index
  - Proximity queries
  - Lab integration: Phrase search

- **toy_data**:
  - Documents with repeated phrases
  - Phrase queries to test

---

#### Week 3: Index Compression
**Folder:** `week_3_compression/`
- **theory.md**:
  - Dictionary compression (blocked storage, front coding)
  - Postings compression (gap encoding)
  - Variable byte encoding
  - Gamma codes (derivation)
  - Delta codes
  - Rice/Golomb codes
  - PForDelta
  - Information theory foundations (entropy, Kraft inequality)
  - Manning Ch 5

- **example.ipynb**:
  - Implement 5 compression schemes
  - Compression ratio analysis
  - Speed vs compression tradeoff
  - Derive optimal parameters
  - Lab integration: Compression experiments

- **toy_data**:
  - Postings lists from previous weeks
  - Benchmark compression ratios

---

### **Phase 2: Ranking & Scoring (Weeks 4-6)**

#### Week 4a: TF-IDF & Vector Space Model
**Folder:** `week_4a_tfidf_vsm/`
- **theory.md**:
  - Term frequency variants (raw, log, normalized)
  - Document frequency & IDF
  - Why log in TF? Why log in IDF?
  - TF-IDF formula derivation
  - SMART notation (all variants)
  - Vector space geometry
  - Dot product, Cosine similarity
  - Length normalization
  - Proof: Why cosine > Euclidean
  - Manning Ch 6 (sections 6.2-6.4)

- **example.ipynb**:
  - TF-IDF calculation by hand (verify with code)
  - Implement 18 SMART variants (6 TF × 3 IDF)
  - Cosine similarity examples
  - 3D visualization of vector space
  - Ranking documents by relevance
  - Lab 2 integration: Classical IR models

- **toy_data**:
  - 10 documents + 5 queries
  - Small enough to calculate TF-IDF by hand

---

#### Week 4b: BM25 Deep Dive
**Folder:** `week_4b_bm25/`
- **theory.md**:
  - BM25 formula complete breakdown
  - Every parameter explained (k₁, b)
  - Derivation from 2-Poisson model
  - Robertson-Sparck Jones weights
  - Probabilistic relevance framework
  - What happens when k₁→0, k₁→∞, b=0, b=1?
  - Prove: BM25 reduces to TF-IDF under certain settings
  - Manning Ch 11

- **example.ipynb**:
  - BM25 implementation from scratch
  - Parameter sensitivity analysis (k₁: 0.5-3.0, b: 0-1)
  - Visualize saturation curves
  - Compare BM25 vs TF-IDF
  - Real-world parameter tuning
  - Lab 2 integration: BM25 exercises

- **toy_data**:
  - Documents with varying lengths
  - Test length normalization effect

---

#### Week 5: BM25F & Language Models
**Folder:** `week_5_bm25f_lm/`
- **theory.md**:
  - BM25F (multi-field extension)
  - Per-field weights and length normalization
  - Query likelihood language models
  - Unigram model
  - Smoothing (Jelinek-Mercer, Dirichlet)
  - Why smoothing is necessary
  - KL divergence between query and document
  - Manning Ch 11, Ch 12

- **example.ipynb**:
  - BM25F implementation (title, body, metadata)
  - Language model implementation
  - Different smoothing methods
  - Compare BM25 vs BM25F vs LM
  - Lab integration: Multi-field ranking

- **toy_data**:
  - Documents with title + body fields
  - Entity-like structure (similar to OpenSanctions)

---

#### Week 6: Query Processing at Scale
**Folder:** `week_6_query_optimization/`
- **theory.md**:
  - Document-at-a-time (DAAT)
  - Term-at-a-time (TAAT)
  - Max-score pruning
  - WAND (Weak AND) algorithm
  - Block-Max WAND
  - Impact ordering
  - Complexity analysis
  - Manning Ch 7

- **example.ipynb**:
  - DAAT vs TAAT implementation
  - WAND with skip pointers
  - Benchmark: queries/second
  - When does WAND help?
  - Lab integration: Efficient retrieval

- **toy_data**:
  - Larger dataset (100-1000 documents)
  - Performance testing

---

### **Phase 3: Modern Approaches (Weeks 7-9)**

#### Week 7: Learning to Rank
**Folder:** `week_7_learning_to_rank/`
- **theory.md**:
  - Pointwise (regression/classification)
  - Pairwise (RankNet loss derivation)
  - Listwise (ListNet, LambdaMART)
  - Feature engineering
  - Gradient derivation for pairwise loss
  - Direct NDCG optimization
  - Papers: Burges et al. (2005, 2010)

- **example.ipynb**:
  - RankNet from scratch (with backprop)
  - Feature engineering (15+ features)
  - LambdaMART with LightGBM
  - Feature importance analysis
  - Compare classical vs learned
  - Lab integration: ML for ranking

- **toy_data**:
  - Documents with multiple features
  - Relevance labels (0-2 scale)

---

#### Week 8: Dense Retrieval & Embeddings
**Folder:** `week_8_dense_retrieval/`
- **theory.md**:
  - Dense vector representations
  - Bi-encoder architecture
  - Contrastive learning (DPR)
  - In-batch negatives
  - ColBERT (late interaction)
  - ANCE (hard negative mining)
  - Loss function derivations
  - Papers: Karpukhin et al. (2020), Khattab & Zaharia (2020)

- **example.ipynb**:
  - Sentence transformers usage
  - Fine-tune embeddings (if data available)
  - Semantic search examples
  - Compare sparse vs dense
  - Lab integration: Neural retrieval

- **toy_data**:
  - Documents requiring semantic understanding
  - Synonym/paraphrase queries

---

#### Week 9a: Vector Search (FAISS)
**Folder:** `week_9a_vector_search/`
- **theory.md**:
  - Exact k-NN (O(N) problem)
  - HNSW (Hierarchical Navigable Small World)
  - IVF (Inverted File Index for vectors)
  - Product Quantization (PQ)
  - MIPS vs k-NN
  - Quantization theory
  - Papers: Malkov & Yashunin (2018), Johnson et al. (2019)

- **example.ipynb**:
  - FAISS index types (Flat, IVF, HNSW, PQ)
  - Benchmark: accuracy vs speed
  - Parameter tuning (nprobe, M, efSearch)
  - GPU acceleration
  - Lab integration: Scalable search

- **toy_data**:
  - 1K-10K document embeddings
  - Latency testing

---

#### Week 9b: Hybrid Retrieval & Reranking
**Folder:** `week_9b_hybrid_reranking/`
- **theory.md**:
  - Sparse + Dense fusion
  - RRF (Reciprocal Rank Fusion)
  - Linear combination
  - Two-stage retrieval
  - Cross-encoder reranking
  - MonoBERT, DuoBERT
  - Why cross-encoder > bi-encoder
  - Papers: Nogueira & Cho (2019)

- **example.ipynb**:
  - BM25 + Dense hybrid
  - RRF implementation
  - Cross-encoder reranking
  - Complete pipeline: Retrieve → Rerank
  - Ablation study
  - Lab integration: Hybrid systems

- **toy_data**:
  - Queries benefiting from semantic + lexical

---

#### Week 10: RAG (Retrieval-Augmented Generation)
**Folder:** `week_10_rag/`
- **theory.md**:
  - RAG architecture
  - Retriever component
  - Prompt engineering for LLMs
  - Query rewriting
  - HyDE (Hypothetical Document Embeddings)
  - Multi-hop retrieval
  - Self-RAG
  - Papers: Lewis et al. (2020), Asai et al. (2023)

- **example.ipynb**:
  - Basic RAG: BM25 + Flan-T5
  - Advanced RAG: Hybrid + GPT-4
  - Query rewriting examples
  - HyDE implementation
  - Answer evaluation
  - Lab integration: Question answering

- **toy_data**:
  - Documents for Q&A
  - Questions requiring synthesis

---

### **Phase 4: Evaluation & Advanced (Weeks 11-12)**

#### Week 11: Evaluation
**Folder:** `week_11_evaluation/`
- **theory.md**:
  - Precision@K, Recall@K, F1@K
  - MAP (Mean Average Precision)
  - NDCG (derivation: why log discount?)
  - MRR (Mean Reciprocal Rank)
  - Statistical significance (t-test, Wilcoxon)
  - Bootstrap resampling
  - Manning Ch 8

- **example.ipynb**:
  - Implement all metrics by hand
  - Statistical testing examples
  - Create test collection
  - Inter-annotator agreement
  - Lab 3 integration: Evaluation exercises

- **toy_data**:
  - Queries with relevance judgments
  - Ground truth labels

---

#### Week 12: Advanced Topics (Entity Search, Multilingual, Efficiency)
**Folder:** `week_12_advanced/`
- **theory.md**:
  - Entity-oriented retrieval
  - Entity linking
  - Multilingual IR
  - Query expansion (Rocchio)
  - Caching strategies
  - Distributed IR

- **example.ipynb**:
  - Entity recognition + linking
  - Fuzzy name matching
  - Cross-lingual search
  - Query expansion with PRF
  - Lab integration: Advanced techniques

- **toy_data**:
  - Entity-like documents (OpenSanctions structure)
  - Multilingual text

---

## 📅 Timeline Summary

| Week | Topic | Duration |
|------|-------|----------|
| 1a | Text Processing & Indexing Basics | 3-4 days |
| 1b | Inverted Index Construction | 3-4 days |
| 2a | Boolean Retrieval | 3-4 days |
| 2b | Phrase Queries & Positional Index | 2-3 days |
| 3 | Index Compression | 4-5 days |
| 4a | TF-IDF & Vector Space Model | 3-4 days |
| 4b | BM25 Deep Dive | 4-5 days |
| 5 | BM25F & Language Models | 3-4 days |
| 6 | Query Processing at Scale | 3-4 days |
| 7 | Learning to Rank | 4-5 days |
| 8 | Dense Retrieval | 4-5 days |
| 9a | Vector Search (FAISS) | 2-3 days |
| 9b | Hybrid & Reranking | 3-4 days |
| 10 | RAG | 3-4 days |
| 11 | Evaluation | 3-4 days |
| 12 | Advanced Topics | 3-4 days |

**Total: ~10-12 weeks for complete mastery**

---

## 🎯 Minimum Path (For Quick Project Completion)

If you need to finish quickly, follow these essential weeks:
1. **Week 1b**: Inverted Index
2. **Week 4a**: TF-IDF & VSM
3. **Week 4b**: BM25
4. **Week 8**: Dense Retrieval
5. **Week 9b**: Hybrid & Reranking
6. **Week 10**: RAG
7. **Week 11**: Evaluation

**Total: ~4-5 weeks minimum**

---

## 📝 Usage

For each week:
1. Read `theory.md` → Understand concepts + math
2. Derive formulas by hand
3. Open `example.ipynb` → Run code, see theory in action
4. Modify examples, experiment with parameters
5. Do exercises
6. Apply to OpenSanctions (your project)

---

## 🔄 Old vs New Structure

**Old (4 modules):**
- Module 1: Indexing & TF-IDF (5 days)
- Module 2: Classical IR (8 days)
- Module 3: Evaluation (5 days)
- Module 4: Dense Retrieval (3 days)

**New (12 weeks):**
- Much more granular
- Each week focused on one topic
- Progressive difficulty
- Theory → Practice → Application

---

## ✅ Next Steps

We'll start with **Week 1a: Text Processing & Indexing Basics**:
1. Create theory.md
2. Prepare toy dataset (10 simple documents)
3. Build example.ipynb with Lab 1 integration

Ready to begin? 🚀
