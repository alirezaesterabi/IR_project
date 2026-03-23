# Comprehensive IR Learning Plan
**Beyond Slides: A Deep Understanding Approach**

---

## Philosophy: The Three-Layer Learning Model

To truly understand Information Retrieval, you need three interconnected layers:

```
┌─────────────────────────────────────────┐
│  Layer 3: IMPLEMENTATION & EXPERIMENTS  │  ← What does it do in practice?
├─────────────────────────────────────────┤
│  Layer 2: MATHEMATICAL FOUNDATIONS      │  ← Why does it work?
├─────────────────────────────────────────┤
│  Layer 1: CONCEPTUAL UNDERSTANDING      │  ← What problem are we solving?
└─────────────────────────────────────────┘
```

Most courses focus on Layer 1 + some Layer 3. **Your advantage** is going deep into Layer 2 (the math) and understanding the "why" behind everything.

---

## The Complete IR Learning Journey

### Phase 1: Foundations (Weeks 1-3) - "The Building Blocks"

#### 1.1 Text Processing & Indexing [Week 1]
**Core Question:** *How do we turn unstructured text into searchable data structures?*

**Must Learn:**
- **Tokenization**: Not just "split by space" - understand:
  - Unicode normalization (NFC, NFD, NFKC, NFKD)
  - Language-specific rules (German compounds, Arabic joining, Chinese segmentation)
  - Case folding (Turkish i problem, German ß)
  - Numbers, dates, URLs, hashtags handling

- **Morphological Analysis**:
  - Porter Stemmer: Learn all 5 steps + implement from scratch
  - Snowball stemmers: Language-specific improvements
  - Lemmatization: WordNet, spaCy, neural approaches
  - **Derive by hand**: Stem 20 words using Porter algorithm before using libraries

- **Inverted Index Construction**:
  - In-memory: Simple dictionary + postings
  - Blocked Sort-Based Indexing (BSBI): For medium collections
  - Single-Pass In-Memory Indexing (SPIMI): Dynamic memory management
  - MapReduce indexing: Distributed systems
  - **Implement**: All four approaches on toy datasets (100, 1K, 10K docs)

**Mathematical Focus:**
- Heap's Law: M = kT^b (vocabulary growth)
  - Derive parameters k and b from your data
  - Understand when it breaks down
- Zipf's Law: cf_i ∝ 1/i (frequency distribution)
  - Plot log-log graph, measure fit
  - Implications for compression and caching

**Resources:**
- Manning Ch 1-2, 4
- Papers:
  - "Fast Inverted Index Construction" (Heinz & Zobel 2003)
  - "MapReduce for Index Construction" (Dean & Ghemawat 2004)
  - Porter (1980) original stemming paper

**Deliverables:**
1. Hand-written stemming of 20 words with Porter algorithm
2. Implementation of BSBI and SPIMI from scratch
3. Analysis: Zipf & Heap's law on 3 different corpora
4. Performance comparison: time/memory tradeoffs

---

#### 1.2 Boolean Retrieval & Query Processing [Week 1-2]
**Core Question:** *How do we efficiently answer queries using the index?*

**Must Learn:**
- **Boolean Queries**:
  - AND, OR, NOT operations on postings lists
  - Query optimization: term ordering by df(t)
  - Skip pointers: When to use, implementation, space-time tradeoff

- **Postings List Intersection**:
  - Baseline: O(x + y) merge algorithm
  - With skip pointers: Expected vs worst case
  - Adaptive skip distances
  - **Derive**: Optimal skip pointer distance (√n analysis)

- **Phrase Queries**:
  - Biword index: Storage overhead
  - Positional index: Space vs flexibility tradeoff
  - Positional intersection algorithm
  - **Implement**: Phrase query "information retrieval systems" on OpenSanctions

**Mathematical Focus:**
- Skip pointer optimization:
  - Prove why √n is optimal spacing
  - Derive expected number of comparisons
- Query complexity analysis:
  - Best case, worst case, average case
  - Impact of term statistics

**Advanced Topics (Optional but Recommended):**
- Wildcard queries (k-gram indexes, permuterm)
- Spell correction (edit distance, k-gram overlap)
- Soundex and phonetic matching

**Resources:**
- Manning Ch 1-3
- Papers:
  - "Skip Pointers: Trading Space for Time" (Moffat & Zobel 1996)

**Deliverables:**
1. Proof of optimal skip pointer distance
2. Implementation with skip pointers
3. Benchmark: Boolean queries with/without skip pointers on OpenSanctions
4. Phrase query implementation with positional index

---

#### 1.3 Index Compression [Week 2]
**Core Question:** *How do we store indexes efficiently?*

**Must Learn:**
- **Dictionary Compression**:
  - Dictionary-as-string with pointer table
  - Blocked storage (4 terms per block)
  - Front coding (prefix elimination)

- **Postings Compression**:
  - Gap encoding: Store differences, not absolute values
  - Variable byte encoding: Self-delimiting codes
  - Gamma codes: Unary length + binary offset
  - Delta codes: Improvement over gamma
  - Rice/Golomb codes: Optimal for specific distributions
  - PForDelta: Modern approach (patched frame-of-reference)

**Mathematical Focus:**
- **Information Theory Foundations**:
  - Entropy: H = -Σ p(x) log p(x)
  - Derive optimal code length for Zipfian distribution
  - Kraft inequality for prefix codes

- **Compression Ratio Analysis**:
  - Expected bits per posting: E[bits] for each scheme
  - Trade-off curves: compression ratio vs decompression speed

**Deep Dive Exercise:**
- **Derive from scratch**: Why gamma code uses unary length + binary offset
- **Implement**: All 5 compression schemes (VB, gamma, delta, Rice, PForDelta)
- **Benchmark**: On OpenSanctions postings lists

**Resources:**
- Manning Ch 5
- Papers:
  - "Inverted Index Compression" (Witten et al. 1999)
  - "PForDelta: Fast Integer Compression" (Zukowski et al. 2006)
- Textbook: Cover & Thomas "Elements of Information Theory" (Ch 5 on Huffman codes)

**Deliverables:**
1. Hand calculation: Compress sample postings list with each method
2. Full implementation of 5 compression schemes
3. Analysis: Compression ratio vs speed tradeoff on OpenSanctions
4. Mathematical derivation: Optimal Rice parameter for Zipfian distribution

---

### Phase 2: Ranking & Scoring (Weeks 3-5) - "The Core of IR"

#### 2.1 Vector Space Model & TF-IDF [Week 3]
**Core Question:** *How do we measure relevance beyond Boolean matching?*

**Must Learn:**
- **TF-IDF Foundations**:
  - Term frequency variants: raw, log, normalized, binary
  - Document frequency: Why IDF = log(N/df) not 1/df?
  - Different IDF formulations: log((N+1)/(df+1)), log(N/df) + 1
  - SMART notation: ddd.qqq schemes (ltc.lnc, etc.)

**Mathematical Deep Dive:**
- **Derive from first principles**:
  - Why logarithm in TF? Information theory justification
  - Why logarithm in IDF? Discrimination value
  - Alternative derivation: IDF from probability theory

- **Vector Space Geometry**:
  - Documents as vectors in |V|-dimensional space
  - Dot product: q·d = Σ w_t,q × w_t,d
  - Cosine similarity: cos(θ) = (q·d)/(|q||d|)
  - **Prove**: Why cosine is better than Euclidean distance
  - Length normalization: Why divide by |d|?

**Implementation Exercises:**
1. Calculate TF-IDF by hand for 10 documents, 5 terms
2. Implement 6 different TF schemes × 3 IDF schemes = 18 variants
3. Compare SMART notation schemes on OpenSanctions test queries
4. Geometric visualization: 3D vector space for 3-term vocabulary

**Resources:**
- Manning Ch 6
- Papers:
  - Salton & Buckley (1988) "Term Weighting Approaches in Automatic Text Retrieval"
  - Singhal et al. (1996) "Pivoted Document Length Normalization"

**Deliverables:**
1. Complete TF-IDF calculation (by hand) for sample corpus
2. Implementation of all SMART variants
3. Empirical comparison on OpenSanctions: which scheme works best?
4. Mathematical proof: cosine vs Euclidean distance

---

#### 2.2 Classical Ranking Models [Week 3-4]
**Core Question:** *How do different models balance term frequency and document length?*

**Must Learn:**

**BM25 (Okapi) - THE MOST IMPORTANT MODEL**
- Formula breakdown:
  ```
  score(q,d) = Σ IDF(t) × (f(t,d) × (k₁ + 1)) / (f(t,d) + k₁ × (1 - b + b × |d|/avgdl))
  ```

- **Derive every component**:
  - Why (k₁ + 1) in numerator? Saturation effect
  - Why k₁ in denominator? Controls TF saturation speed
  - Why b parameter? Length normalization strength
  - What happens when b=0? (no length norm)
  - What happens when b=1? (full length norm)
  - What happens when k₁→∞? (linear TF)
  - What happens when k₁→0? (binary TF)

- **Mathematical Foundation**:
  - BM25 comes from probabilistic relevance framework
  - Robertson-Sparck Jones weights
  - 2-Poisson model for term occurrence
  - **Study**: Robertson & Walker (1994) paper - understand the full derivation

**Language Models:**
- Query likelihood model: P(q|d)
  - Unigram language model
  - Maximum likelihood estimation
  - Smoothing: Jelinek-Mercer, Dirichlet
  - **Derive**: Why smoothing is necessary (zero probability problem)

**Divergence Models:**
- KL divergence: D(q||d)
- Cross-entropy
- Connection to language models

**Advanced: BM25F (Multi-field BM25)**
- Different weights for title, body, anchor text
- Per-field length normalization
- Implementation for OpenSanctions (name, address, aliases)

**Deep Mathematical Exercise:**
- **Prove**: BM25 reduces to TF-IDF under certain parameter settings
- **Derive**: Relationship between BM25 and 2-Poisson model
- **Implement**: BM25 from scratch (no libraries) with configurable k₁, b

**Resources:**
- Manning Ch 11 (Probabilistic IR)
- Papers (CRITICAL - READ ALL):
  - Robertson & Walker (1994) "Okapi at TREC-3" - Original BM25
  - Robertson et al. (2004) "BM25F: Extensions to BM25"
  - Zhai & Lafferty (2004) "A Study of Smoothing Methods for Language Models"
  - Ponte & Croft (1998) "A Language Modeling Approach to IR"

**Deliverables:**
1. Hand calculation: BM25 score for query on 5 documents
2. Parameter sensitivity analysis: vary k₁ from 0.5 to 3.0, b from 0 to 1
3. Implementation: Pure BM25 + BM25F for OpenSanctions
4. Theoretical derivation: BM25 from 2-Poisson model (10-page write-up)
5. Comparison: BM25 vs TF-IDF vs Language Models on test queries

---

#### 2.3 Query Processing at Scale [Week 4]
**Core Question:** *How do we rank millions of documents efficiently?*

**Must Learn:**
- **Document-at-a-time (DAAT)**:
  - Merge postings lists, accumulate scores
  - Heap for top-K retrieval

- **Term-at-a-time (TAAT)**:
  - Process each term completely
  - Accumulator array for all documents
  - More cache-friendly

- **Optimization Techniques**:
  - **Max-score**: Upper bounds on term contributions
  - **WAND** (Weak AND): Skip non-competitive documents
  - **Block-Max WAND**: Better pruning with block-level statistics
  - Impact ordering: Sort postings by impact scores

**Mathematical Focus:**
- Prove correctness of max-score pruning
- Analyze WAND skip distance in practice
- Expected number of score computations: DAAT vs TAAT

**Resources:**
- Manning Ch 7
- Papers:
  - Turtle & Flood (1995) "Query Evaluation: Strategies and Optimizations"
  - Broder et al. (2003) "Efficient Query Evaluation using a Two-Level Retrieval Process"
  - Ding & Suel (2011) "Faster Top-k Document Retrieval using Block-Max Indexes"

**Deliverables:**
1. Implementation: DAAT vs TAAT on OpenSanctions
2. WAND implementation with skip pointers
3. Benchmark: queries/second for different optimizations
4. Analysis: When is WAND better than exhaustive scoring?

---

### Phase 3: Modern Approaches (Weeks 5-7) - "Beyond Classical IR"

#### 3.1 Learning to Rank [Week 5]
**Core Question:** *Can we learn optimal ranking functions from data?*

**Must Learn:**

**Pointwise Approaches:**
- Treat ranking as regression/classification
- Features: BM25 score, document length, PageRank, query-doc term overlap
- Models: Linear regression, logistic regression, neural networks

**Pairwise Approaches:**
- RankNet: Neural network with pairwise loss
- Loss function: Cross-entropy on pairs
- **Derive**: Gradient of pairwise loss

**Listwise Approaches:**
- ListNet: Probability distribution over permutations
- LambdaMART: Gradient boosting for ranking
- Direct optimization of ranking metrics (NDCG)

**Mathematical Depth:**
- **Derive**: Why pairwise loss is better than pointwise
- **Implement**: RankNet from scratch (including backprop)
- **Study**: LambdaMART - how does it optimize NDCG directly?

**Resources:**
- Papers (MUST READ):
  - Burges et al. (2005) "Learning to Rank using Gradient Descent" (RankNet)
  - Cao et al. (2007) "Learning to Rank: From Pairwise to Listwise" (ListNet)
  - Burges (2010) "From RankNet to LambdaRank to LambdaMART"
  - Liu (2009) "Learning to Rank for Information Retrieval" - Survey paper

**Deliverables:**
1. Feature engineering for OpenSanctions (15+ features)
2. RankNet implementation from scratch
3. LambdaMART using LightGBM/XGBoost
4. Comparison: Classical (BM25) vs Learning to Rank
5. Feature importance analysis: What matters most?

---

#### 3.2 Dense Retrieval & Embeddings [Week 6]
**Core Question:** *Can we move beyond exact term matching to semantic similarity?*

**Must Learn:**

**Dense Retrieval Fundamentals:**
- Embedding documents and queries in dense vector space (d=768 typical)
- Semantic similarity: cos(embed(q), embed(d))
- Bi-encoder architecture: Separate encoders for queries and documents

**Key Models:**
- **DPR (Dense Passage Retrieval)**:
  - BERT-based encoders
  - Contrastive learning: positive vs negative passages
  - In-batch negatives

- **ColBERT**:
  - Late interaction: MaxSim over token embeddings
  - Better than single [CLS] representation
  - Compression: 128-dim per token

- **ANCE (Approximate Nearest Neighbor Negative Contrastive Learning)**:
  - Hard negative mining
  - Asynchronous index updates

**Vector Search:**
- Exact k-NN: Brute force O(N) - too slow
- **HNSW** (Hierarchical Navigable Small World):
  - Multi-layer graph structure
  - Probabilistic skip lists analogy
  - Log complexity for search

- **IVF (Inverted File Index)**:
  - Cluster vectors, search nearby clusters
  - Product Quantization (PQ) for compression

- **FAISS library**: Facebook's vector search
  - Different index types: Flat, IVF, HNSW, PQ
  - GPU acceleration

**Mathematical Deep Dive:**
- **Contrastive Loss**:
  ```
  L = -log(exp(sim(q,d+)) / (exp(sim(q,d+)) + Σ exp(sim(q,d-))))
  ```
  - Why does this work? Connection to softmax
  - In-batch negatives: Computational efficiency

- **Maximum Inner Product Search (MIPS)**:
  - Difference from k-NN
  - Transform MIPS to k-NN

- **Quantization Theory**:
  - Vector quantization: Lloyd's algorithm (k-means)
  - Product quantization: Divide space into subspaces
  - Compute bounds on approximation error

**Resources:**
- Papers (ESSENTIAL):
  - Karpukhin et al. (2020) "Dense Passage Retrieval" (DPR)
  - Khattab & Zaharia (2020) "ColBERT: Efficient and Effective Passage Search"
  - Xiong et al. (2021) "Approximate Nearest Neighbor Negative Contrastive Learning" (ANCE)
  - Malkov & Yashunin (2018) "Efficient and Robust HNSW"
  - Johnson et al. (2019) "Billion-scale similarity search with FAISS"

**Deliverables:**
1. Fine-tune sentence-transformers on domain-specific data (if available)
2. Dense retrieval implementation: DPR-style model
3. Vector index comparison: FAISS flat vs IVF vs HNSW
4. Analysis: Dense vs sparse (BM25) vs hybrid on OpenSanctions
5. Latency benchmarking: exact vs approximate search

---

#### 3.3 Hybrid Retrieval & Reranking [Week 6-7]
**Core Question:** *How do we combine the best of sparse and dense methods?*

**Must Learn:**

**Hybrid Retrieval:**
- Sparse (BM25) + Dense (embeddings) combination
- **Fusion Strategies**:
  - Linear combination: α×score_sparse + (1-α)×score_dense
  - Reciprocal Rank Fusion (RRF): 1/(k + rank)
  - CombSUM, CombMNZ variants

**Two-Stage Retrieval:**
1. **Stage 1 - Retrieval**: Fast candidate generation (sparse or dense)
2. **Stage 2 - Reranking**: Expensive but accurate (cross-encoders)

**Cross-Encoder Reranking:**
- Feed [CLS] query [SEP] document [SEP] to BERT
- Classification: relevant / not relevant
- Much more accurate than bi-encoders (but slow)
- Can only rerank top-K (K=100 typical)

**Models:**
- **MonoBERT**: BERT for reranking
- **DuoBERT**: BERT for pairwise comparison
- **T5 for ranking**: "Query: X Document: Y Relevant:"

**Advanced: Late Interaction Models:**
- ColBERTv2: Residual compression
- SPLADE: Sparse + dense in one model (learned sparse representations)

**Mathematical Analysis:**
- **Derive**: Why RRF works (rank-based fusion is robust)
- **Prove**: Cross-encoder > bi-encoder (more interaction)
- Analyze speed vs quality tradeoff

**Resources:**
- Papers:
  - Nogueira & Cho (2019) "Passage Re-ranking with BERT" (MonoBERT)
  - Nogueira et al. (2019) "Document Ranking with DuoBERT"
  - Formal et al. (2021) "SPLADE: Sparse Lexical and Expansion Model"
  - Santhanam et al. (2021) "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"

**Deliverables:**
1. Hybrid retrieval: BM25 + Dense with RRF
2. Cross-encoder reranking on top-100
3. Complete pipeline: Retrieve (hybrid) → Rerank (cross-encoder) → Results
4. Ablation study: Each component's contribution
5. End-to-end evaluation on OpenSanctions queries

---

#### 3.4 Retrieval-Augmented Generation (RAG) [Week 7]
**Core Question:** *How do we use retrieval to improve LLM responses?*

**Must Learn:**

**RAG Architecture:**
```
Query → Retriever → [top-K docs] → LLM (with docs as context) → Answer
```

**Key Components:**
1. **Retriever**: BM25 / Dense / Hybrid (what you built above)
2. **Prompt Engineering**: How to format retrieved docs for LLM
3. **LLM**: GPT, Flan-T5, LLaMA for generation
4. **Post-processing**: Citation, fact-checking

**Advanced RAG Techniques:**
- **Query Rewriting**: Expand query before retrieval
- **Hypothetical Document Embeddings (HyDE)**: Generate fake doc, embed, search
- **Multi-hop Retrieval**: Iterative retrieval for complex questions
- **Self-RAG**: LLM decides when to retrieve

**Evaluation:**
- Answer quality: BLEU, ROUGE, BERTScore
- Faithfulness: Does answer match retrieved docs?
- Relevance: Are retrieved docs actually useful?

**Resources:**
- Papers:
  - Lewis et al. (2020) "RAG: Retrieval-Augmented Generation"
  - Gao et al. (2023) "REALM: Retrieval-Augmented Language Model Pre-Training"
  - Asai et al. (2023) "Self-RAG: Learning to Retrieve, Generate and Critique"
  - Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE)

**Deliverables:**
1. Basic RAG: BM25 retrieval + Flan-T5 generation
2. Advanced RAG: Hybrid retrieval + query rewriting + GPT-4
3. Evaluation framework: Answer quality metrics
4. OpenSanctions application: Sanctions entity Q&A system

---

### Phase 4: Evaluation & Experimentation (Week 8) - "How do we know what works?"

#### 4.1 Evaluation Metrics [Week 8]
**Core Question:** *How do we measure retrieval quality?*

**Must Master:**

**Binary Relevance Metrics:**
- Precision@K: What fraction of top-K are relevant?
- Recall@K: What fraction of relevant docs are in top-K?
- F1@K: Harmonic mean
- **Derive by hand**: For sample ranking with 10 docs, 3 relevant

**Ranked Metrics:**
- **Mean Average Precision (MAP)**:
  ```
  MAP = (1/|Q|) Σ_q (1/|R_q|) Σ_{k=1}^K P@k × rel(k)
  ```
  - Calculate by hand for 3 queries

- **Normalized Discounted Cumulative Gain (NDCG)**:
  ```
  DCG@K = Σ_{i=1}^K (2^{rel_i} - 1) / log_2(i + 1)
  NDCG@K = DCG@K / IDCG@K
  ```
  - **Derive**: Why log discount? Information theory view
  - **Derive**: Why 2^rel? Exponential gain vs linear

- **Mean Reciprocal Rank (MRR)**:
  ```
  MRR = (1/|Q|) Σ_q 1/rank_q
  ```
  - When to use? First relevant result matters most

**Statistical Significance:**
- Paired t-test: Is system A better than B?
- Wilcoxon signed-rank test: Non-parametric alternative
- Bootstrap resampling: Confidence intervals
- **Practice**: Run significance tests on your experiments

**Resources:**
- Manning Ch 8
- Papers:
  - Järvelin & Kekäläinen (2002) "Cumulated Gain-Based Evaluation" (NDCG)
  - Buckley & Voorhees (2004) "Retrieval Evaluation with Incomplete Information"
- Book: Buttcher et al. "Information Retrieval: Implementing and Evaluating Search Engines" (Ch 8)

**Deliverables:**
1. Hand calculation: All metrics for sample ranking
2. Implement evaluation library from scratch
3. Statistical significance testing for OpenSanctions experiments
4. Analysis: Which metric best correlates with user satisfaction?

---

#### 4.2 Test Collections & Benchmarks [Week 8]
**Core Question:** *How do we create ground truth and compare systems?*

**Must Learn:**

**Standard Test Collections:**
- **MS MARCO**: 8.8M passages, 1M queries (sparse labels)
- **TREC Deep Learning**: ~200K passages per query (dense labels)
- **Natural Questions**: Wikipedia, 307K questions
- **BEIR**: 18 datasets for zero-shot evaluation

**Creating Your Own Test Set:**
- Query generation: Manual vs synthetic (LLM-generated)
- Relevance judgments: Binary vs graded (0-3 scale)
- Pooling method: Judge only top-K from multiple systems
- Inter-annotator agreement: Kappa statistic

**Cranfield Paradigm:**
1. Document collection
2. Queries with information needs
3. Relevance judgments (ground truth)
4. Metrics to compare systems

**OpenSanctions Test Set:**
- Create 50-100 queries covering different entity types
- Relevance judgments: 3 levels (exact match, partial, not relevant)
- Coverage: Person, organization, vessel, aircraft queries

**Resources:**
- Papers:
  - Voorhees & Harman (2005) "TREC: Experiment and Evaluation in Information Retrieval"
  - Thakur et al. (2021) "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation"
  - Nguyen et al. (2016) "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset"

**Deliverables:**
1. OpenSanctions test collection: 50-100 queries with relevance judgments
2. Annotation guidelines document
3. Inter-annotator agreement analysis (if multiple annotators)
4. Benchmark: All your systems on this test set

---

### Phase 5: Advanced Topics (Weeks 9-10) - "Specialized Knowledge"

#### 5.1 Query Understanding [Week 9]
**Core Question:** *How do we understand what the user really wants?*

**Topics:**
- **Query Expansion**: Add related terms (synonyms, co-occurring terms)
  - Pseudo-relevance feedback (PRF): Use top-K docs to expand
  - Rocchio algorithm: α×q + β×(relevant) - γ×(non-relevant)
  - **Derive**: Optimal α, β, γ values

- **Query Rewriting**:
  - Spell correction: Edit distance, language models
  - Query reformulation: Templates, seq2seq models
  - Session-based: Learn from user behavior

- **Intent Classification**:
  - Navigational: "facebook login"
  - Informational: "how does BM25 work"
  - Transactional: "buy iphone 15"

**Resources:**
- Manning Ch 9
- Papers:
  - Xu & Croft (1996) "Query Expansion Using Local and Global Document Analysis"
  - Ahmad et al. (2020) "Context-Aware Query Suggestion"

---

#### 5.2 Multilingual & Cross-Lingual IR [Week 9]
**Core Question:** *How do we search across languages?*

**Topics:**
- Dictionary-based translation
- Machine translation for queries/documents
- Cross-lingual embeddings: Aligned vector spaces
- Multilingual models: mBERT, XLM-R

**OpenSanctions Relevance:**
- Entities have names in multiple languages
- Addresses in native scripts
- Cross-lingual entity matching

---

#### 5.3 Entity Search & Knowledge Graphs [Week 10]
**Core Question:** *How do we search for entities specifically?*

**Topics:**
- **Entity Recognition & Linking**:
  - Named Entity Recognition (NER): BERT-based taggers
  - Entity Linking: Connect mentions to knowledge base

- **Entity-Oriented Retrieval**:
  - BM25F with entity fields
  - Entity embeddings: TransE, RotatE
  - Graph-based ranking: PageRank on entity graph

- **OpenSanctions Specific**:
  - Entity resolution: Same person, different names
  - Relationship queries: "Show entities connected to X"
  - Fuzzy matching: Name variants, transliterations

**Resources:**
- Papers:
  - Balog (2018) "Entity-Oriented Search" - Full book!
  - Hasibi et al. (2017) "DBpedia-Entity: A Test Collection for Entity Search"

---

#### 5.4 Efficiency & Scalability [Week 10]
**Core Question:** *How do large-scale systems actually work?*

**Topics:**
- **Caching**:
  - Query result cache: LRU, LFU policies
  - Posting list cache: Which terms to cache?
  - Optimal cache size: Cost-benefit analysis

- **Distributed IR**:
  - Document partitioning vs term partitioning
  - Query routing
  - Result merging from multiple shards

- **Latency Budgets**:
  - Early termination strategies
  - Time-bounded search: WAND with deadlines

**Resources:**
- Manning Ch 7
- Papers:
  - Cambazoglu & Baeza-Yates (2011) "Scalability Challenges in Web Search Engines"
  - Moffat et al. (2006) "Self-Adjusting Indexes for Dynamic Text Collections"

---

## Your Personalized Plan for OpenSanctions Project

### Minimum Viable Learning Path (3-4 weeks before project implementation)
Focus on what you NEED for the project:

**Week 1: Core Foundations**
- Day 1-2: Tokenization, stemming (Porter), inverted index (BSBI)
- Day 3-4: Boolean retrieval, skip pointers
- Day 5: Index compression (VB coding, gamma codes)
- **Deliverable**: Working inverted index on 1K OpenSanctions entities

**Week 2: Ranking Models**
- Day 1-3: TF-IDF (all variants), SMART notation
- Day 4-7: BM25 - deep dive, derive from scratch, implement
- **Deliverable**: BM25 implementation, parameter tuning on test queries

**Week 3: Evaluation + Modern Approaches**
- Day 1-2: Evaluation metrics (MAP, NDCG, MRR)
- Day 3-4: Dense retrieval (sentence-transformers)
- Day 5-7: Hybrid retrieval (BM25 + dense + RRF)
- **Deliverable**: Complete hybrid system with evaluation

**Week 4: RAG + Polish**
- Day 1-3: RAG implementation (retrieval + Flan-T5)
- Day 4-5: Create test collection (50 queries)
- Day 6-7: Full evaluation, comparison, write-up
- **Deliverable**: Complete project ready for submission

### Maximum Learning Path (10 weeks for deep mastery)
Follow the full plan above - you'll understand IR better than 95% of practitioners.

---

## Key Resources Summary

### Essential Books:
1. **Manning, Raghavan, Schütze** - "Introduction to Information Retrieval" (Your main textbook)
2. **Büttcher, Clarke, Cormack** - "Information Retrieval: Implementing and Evaluating Search Engines" (More implementation-focused)
3. **Balog** - "Entity-Oriented Search" (For entity-specific IR)
4. **Croft, Metzler, Strohman** - "Search Engines: Information Retrieval in Practice" (Alternative view)

### Must-Read Papers (Top 20):
1. Robertson & Walker (1994) - "Okapi at TREC-3" [BM25 origin]
2. Salton & Buckley (1988) - "Term Weighting Approaches" [TF-IDF]
3. Karpukhin et al. (2020) - "Dense Passage Retrieval" [DPR]
4. Khattab & Zaharia (2020) - "ColBERT" [Late interaction]
5. Lewis et al. (2020) - "RAG" [Retrieval-augmented generation]
6. Burges et al. (2005) - "Learning to Rank using Gradient Descent" [RankNet]
7. Burges (2010) - "From RankNet to LambdaRank to LambdaMART" [Learning to rank survey]
8. Zhai & Lafferty (2004) - "Smoothing Methods for Language Models"
9. Ponte & Croft (1998) - "Language Modeling Approach to IR"
10. Robertson et al. (2004) - "BM25F Extensions"
11. Moffat & Zobel (1996) - "Self-Indexing Inverted Files"
12. Witten et al. (1999) - "Managing Gigabytes" [Compression]
13. Broder et al. (2003) - "Efficient Query Evaluation"
14. Malkov & Yashunin (2018) - "Efficient and Robust HNSW"
15. Johnson et al. (2019) - "Billion-scale similarity search with FAISS"
16. Thakur et al. (2021) - "BEIR Benchmark"
17. Asai et al. (2023) - "Self-RAG"
18. Formal et al. (2021) - "SPLADE"
19. Järvelin & Kekäläinen (2002) - "NDCG"
20. Voorhees & Harman (2005) - "TREC Overview"

### Online Courses (Optional Supplement):
- Stanford CS276: Information Retrieval and Web Search
- CMU 11-442/11-642: Search Engines
- University of Amsterdam: Information Retrieval 1 & 2

---

## Weekly Deliverables Checklist

Track your progress with concrete outputs:

### Week 1:
- [ ] Inverted index implementation (BSBI + SPIMI)
- [ ] Porter stemmer by hand (20 words)
- [ ] Skip pointer implementation
- [ ] Zipf/Heap's law analysis on 3 corpora

### Week 2:
- [ ] TF-IDF calculation by hand (10 docs)
- [ ] SMART notation implementation (6 variants)
- [ ] 5 compression schemes implemented
- [ ] Compression ratio analysis

### Week 3:
- [ ] BM25 derivation write-up (10 pages)
- [ ] BM25 implementation from scratch
- [ ] Parameter sensitivity analysis (k₁, b)
- [ ] Language model implementation

### Week 4:
- [ ] WAND implementation
- [ ] DAAT vs TAAT comparison
- [ ] Query processing benchmarks
- [ ] Optimization analysis

### Week 5:
- [ ] RankNet from scratch
- [ ] LambdaMART with LightGBM
- [ ] Feature engineering (15+ features)
- [ ] Learning to rank evaluation

### Week 6:
- [ ] Dense retrieval (DPR-style)
- [ ] FAISS index comparison
- [ ] Dense vs sparse analysis
- [ ] Vector search benchmarks

### Week 7:
- [ ] Hybrid retrieval (BM25 + dense + RRF)
- [ ] Cross-encoder reranking
- [ ] RAG implementation
- [ ] Complete pipeline

### Week 8:
- [ ] Evaluation metrics by hand
- [ ] Test collection creation (50-100 queries)
- [ ] Statistical significance testing
- [ ] Full system comparison

### Week 9-10:
- [ ] Query expansion (Rocchio)
- [ ] Entity-specific features
- [ ] Multilingual handling
- [ ] Final optimization

---

## The "Math First, Then Code" Principle

For every major component, follow this workflow:

1. **Understand the problem** (1 hour):
   - What user need does this solve?
   - What are the limitations of simpler approaches?

2. **Study the math** (3-4 hours):
   - Read the original paper
   - Derive formulas by hand
   - Work through examples manually
   - Understand parameter meanings

3. **Implement from scratch** (4-6 hours):
   - No libraries for core algorithm
   - Write clean, documented code
   - Add assertions and tests

4. **Validate** (2 hours):
   - Compare with reference implementation
   - Test on known examples
   - Edge case testing

5. **Optimize** (2-3 hours):
   - Profile code
   - Add efficient data structures
   - Consider libraries now

6. **Apply to project** (2-3 hours):
   - Integration with OpenSanctions
   - Parameter tuning
   - Evaluation

**Total per major component: ~15-20 hours**

---

## Final Advice: The 80/20 Rule

**If you only have limited time, focus on:**

1. **BM25** (50% of your time):
   - Understand it deeply
   - Derive from scratch
   - Implement perfectly
   - This is THE most important model in modern IR

2. **Dense Retrieval** (20%):
   - Sentence transformers
   - FAISS basics
   - Hybrid with BM25

3. **Evaluation** (15%):
   - MAP, NDCG, MRR
   - Create good test set
   - Statistical testing

4. **RAG** (15%):
   - Basic pipeline
   - For your project deliverable

This 80/20 approach will get you 90% of the value while being realistic about time constraints.

---

## Questions to Ask Yourself Weekly

**End of each week:**
1. Can I explain this concept to someone without looking at notes?
2. Can I derive the key formulas from first principles?
3. Can I implement the core algorithm from scratch?
4. Do I understand when/why this would fail?
5. Have I applied this to my project data?

If any answer is "no", spend more time on that topic.

---

**Remember:** The goal isn't to memorize everything, but to **deeply understand** the core concepts so you can:
- Make informed design decisions
- Debug when things don't work
- Innovate and adapt methods to your specific use case
- Critically evaluate new papers and techniques

Good luck! This is an exciting journey into one of the most practical and impactful areas of AI/ML.
