# Integrated Learning & Project Plan

## Overview
This document maps **what to learn** (modules) with **what to build** (project phases). Each project phase requires specific theoretical knowledge first.

---

## 🎯 The Big Picture

```
LEARNING (theory + toy examples) → PROJECT (real implementation)

Module 1: Indexing & TF-IDF    →  Phase 1-2: Data + Preprocessing + Basic Index
Module 2: Classical IR         →  Phase 3-5: Queries + BM25 + Evaluation
Module 3: Evaluation           →  Phase 6: Full Evaluation Framework
Module 4: Dense (optional)     →  Phase 7-8: Help Marek + Final integration
```

---

# 📋 Phase-by-Phase Breakdown

## **PHASE 0: Setup & Foundation** (Week 1, Days 1-2)

### **What to Do:**
✅ Environment setup (already done!)
✅ Download data (already done!)
✅ Understand project structure
✅ Review assignment requirements

### **Learning Required:**
📚 **Skim reading**:
- Manning Ch 1 (pages 1-10) - What is IR?
- Review `documents/learning_roadmap.md`
- Review `documents/implementation_phases.md`

### **Deliverables:**
- ✅ Virtual environment ready
- ✅ Data downloaded
- ✅ Git structure clear
- ✅ Team roles understood

### **Time:** 0.5 days (already done!)

---

## **PHASE 1: Data Exploration** (Week 1, Days 3-5)

### **What to Do:**
1. Run `notebooks/01_data_exploration.ipynb`
2. Understand OpenSanctions dataset structure
3. Analyze field distributions
4. Identify data quality issues
5. Document findings in notebook

### **Learning Required:**
📚 **Module 1 - Part 1: Basic Concepts**
- Read: `learning_modules/01_indexing_tfidf/theory.md` (first half)
  - What is a document?
  - What is a term?
  - What is a collection?
- Manning Ch 1 (pages 1-20) - Boolean retrieval basics

### **No toy example needed yet** - just understand concepts

### **Project Work:**
- 📓 `notebooks/01_data_exploration.ipynb`
  - Load targets.simple.csv
  - Analyze entity types (People, Companies, Vessels, etc.)
  - Check field coverage (name, alias, description, etc.)
  - Visualize distributions
  - Document: What fields will we use for indexing?

### **Deliverables:**
- Completed data exploration notebook with findings
- List of fields to use for text blobs
- Understanding of data structure

### **Time:** 2-3 days

### **Success Criteria:**
- ✅ You know what fields exist
- ✅ You know which fields to index
- ✅ You understand data complexity (nested structures, aliases, etc.)

---

## **PHASE 2: Text Preprocessing & Indexing** (Week 1-2, Days 6-12)

### **What to Do:**
1. Build streaming JSON parser
2. Implement text preprocessing pipeline
3. Flatten nested documents
4. Build basic inverted index
5. Test on 100K subset

### **Learning Required:** 🎓 **COMPLETE MODULE 1**

#### **Step 1: Learn Theory (2 days)**
📚 `learning_modules/01_indexing_tfidf/theory.md`
- **Manning Ch 1 (full)**: Boolean Retrieval
  - Inverted index structure
  - Postings lists
  - Boolean query processing
- **Manning Ch 2 (full)**: Term Vocabulary & Postings
  - Tokenization
  - Normalization (case folding)
  - Stemming vs Lemmatization
  - Porter Stemmer algorithm
  - Zipf's Law
  - Heap's Law
- **Manning Ch 6 (sections 6.1-6.3)**: TF-IDF
  - Term frequency (TF)
  - Document frequency (DF)
  - Inverse document frequency (IDF)
  - TF-IDF weighting scheme

**Math to Master:**
```
TF(t,d) = count of term t in document d
DF(t) = number of documents containing t
IDF(t) = log(N / DF(t))
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

#### **Step 2: Toy Example (1 day)**
🔬 `learning_modules/01_indexing_tfidf/toy_example.py`
- Build inverted index from scratch on 5 documents
- Implement tokenizer, lowercasing, stopword removal
- Calculate TF-IDF by hand and verify with code
- Query the index

**Run and understand every line!**

#### **Step 3: Review Lab 1 (1 day)**
🧪 `class_materials/week_2/Lab_1_Python_Indexing.ipynb`
- Understand the lab implementation
- Compare with your toy example
- What's different? What's similar?

#### **Step 4: Project Implementation (3-4 days)**
🛠️ **Build the preprocessing pipeline:**

**4a. Create `src/preprocessing/parser.py`** (1 day)
```python
def stream_json_documents(file_path):
    """
    Stream large JSON file line by line
    Yield one entity at a time
    Handle 3.68GB without memory overflow
    """
```

**4b. Create `src/preprocessing/text_processing.py`** (1 day)
```python
def preprocess_text(text):
    """
    1. Lowercase
    2. Remove stopwords (nltk)
    3. Lemmatize (spaCy)
    Return: list of processed tokens
    """
```

**4c. Create `src/preprocessing/indexer.py`** (1-2 days)
```python
def flatten_entity(entity_json):
    """
    Input: nested JSON entity
    Output:
    {
        'id': entity_id,
        'text_blob': concatenated text (name + alias + description + ...),
        'identifiers': {imoNumber, mmsi, ...},
        'metadata': {schema, country, programId, ...}
    }
    """

def build_inverted_index(entities):
    """
    Build inverted index from flattened entities
    Structure: {term: [(doc_id, tf), (doc_id, tf), ...]}
    """
```

**4d. Test on 100K subset** (0.5 day)
- Extract first 100K entities from targets.nested.json
- Run full preprocessing pipeline
- Save processed data to `data/processed/subset_100k/`
- Verify: Can you query the index?

### **Deliverables:**
- ✅ `src/preprocessing/parser.py` - working streaming parser
- ✅ `src/preprocessing/text_processing.py` - tokenization, lemmatization
- ✅ `src/preprocessing/indexer.py` - document flattening + basic index
- ✅ `data/processed/subset_100k/` - preprocessed data
- ✅ Basic inverted index that works

### **Time:** 6-7 days total
- 2 days theory
- 1 day toy example
- 1 day lab review
- 3-4 days implementation

### **Success Criteria:**
- ✅ You understand inverted index deeply
- ✅ You can explain TF-IDF formula
- ✅ Preprocessing pipeline works on 100K subset
- ✅ You have searchable index (even if basic)

---

## **PHASE 3: Query Set Development** (Week 2, Days 13-15)

### **What to Do:**
1. Brainstorm 20+ queries (all 7 types)
2. Create Excel file for expert review
3. Get expert feedback
4. Convert to JSON format

### **Learning Required:**
📚 **No new module** - use domain knowledge

But **skim** Manning Ch 9 (Query Operations) to understand:
- Query types
- Query refinement
- Relevance feedback

### **Project Work:**
🛠️ Create `notebooks/query_generation.ipynb`

**For each query type, create 2-3 queries:**

1. **Type 1 - Exact Identifier**: IMO numbers, MMSI
   - Example: `IM09296822`

2. **Type 2 - Name/Alias**: Entity names with variations
   - Example: `Angelica Schulte`, `Vladimir Putin`

3. **Type 3 - Semantic**: Natural language, no exact match
   - Example: `Russian crude oil tanker evading sanctions Baltic Sea`

4. **Type 4 - Relational**: Links between entities
   - Example: `What vessels is Ketee Co Ltd linked to?`

5. **Type 5 - Cross-Dataset**: Same entity, multiple sources
   - Example: `SAGITTA sanctions all sources`

6. **Type 6 - Jurisdiction**: Filtered by authority
   - Example: `OFAC sanctioned vessels Russian oil 2025`

7. **Type 7 - RAG**: Summarization (skip for now, Marek's task)

**Export to Excel:**
- `data/evaluation/queries_draft.xlsx`
- Columns: query_id, query_type, query_text, expected_difficulty, notes

**Get expert review** (or self-review with domain knowledge)

**Convert to JSON:**
- Create `scripts/excel_to_json.py`
- Output: `data/evaluation/queries.json`

### **Deliverables:**
- ✅ 20 queries in JSON format
- ✅ Queries validated by expert (or team)

### **Time:** 2-3 days

---

## **PHASE 4: Ground Truth Construction** (Week 2, Days 16-17)

### **What to Do:**
1. Automatic ground truth for Types 1, 2, 5, 6
2. Prepare pooling for Types 3, 4

### **Learning Required:**
📚 **Module 3 - Part 1: Relevance & Ground Truth**
- Read: `learning_modules/03_evaluation/theory.md` (first section)
- Manning Ch 8 (sections 8.1-8.3)
  - What is relevance?
  - Ground truth construction
  - Pooling methodology

### **Project Work:**
🛠️ Create `scripts/generate_ground_truth.py`

**Automatic extraction:**
```python
def extract_ground_truth_type1(query, dataset):
    """Type 1: Exact identifier match"""
    # Extract entities where imoNumber == query

def extract_ground_truth_type2(query, dataset):
    """Type 2: Name/alias match"""
    # Fuzzy match on name, alias, previousName

def extract_ground_truth_type6(query, dataset):
    """Type 6: Jurisdiction filter"""
    # Filter by programId, schema
```

**Output:** `data/evaluation/qrels.json` (TREC format)
```json
{
  "Q001": {
    "entity_id_123": 1,
    "entity_id_456": 1
  }
}
```

**For Types 3, 4:** Note that pooling will happen after retrieval systems are built

### **Deliverables:**
- ✅ Ground truth for Types 1, 2, 5, 6
- ✅ `data/evaluation/qrels.json`

### **Time:** 1-2 days

---

## **PHASE 5: Classical IR Implementation** ⭐ **YOUR MAIN TASK** (Week 2-3, Days 18-28)

### **What to Do:**
1. Implement BM25 (primary)
2. Implement TF-IDF + cosine (baseline)
3. Implement fuzzy matching (name variants)
4. Test on 100K subset
5. Tune hyperparameters

### **Learning Required:** 🎓 **COMPLETE MODULE 2** (MOST IMPORTANT!)

#### **Step 1: Learn Theory (3-4 days)**
📚 `learning_modules/02_classical_ir/theory.md`

**Part A: Boolean Model**
- Manning Ch 1 (review)
- Boolean queries: AND, OR, NOT
- Postings list intersection

**Part B: Vector Space Model (VSM)**
- Manning Ch 6 (full chapter)
- Documents as vectors
- Queries as vectors
- **Cosine Similarity** (derive formula!)
  ```
  cos(q, d) = (q · d) / (|q| × |d|)
  ```
- Length normalization
- TF-IDF weighting

**Part C: Binary Independence Model (BIM)**
- Manning Ch 11 (sections 11.1-11.2)
- Probability Ranking Principle
- Relevance probability
- RSJ formula

**Part D: BM25** (DEEP DIVE - MOST IMPORTANT!)
- Manning Ch 11 (section 11.4)
- **BM25 Formula** (derive and understand every term!):
  ```
  score(D,Q) = Σ IDF(qi) × [f(qi,D) × (k1 + 1)] /
               [f(qi,D) + k1 × (1 - b + b × |D|/avgdl)]
  ```
- **Parameters:**
  - k1: controls TF saturation (1.2 - 2.0)
  - b: controls length normalization (0.75)
- **Why saturation?** Draw TF curve with/without saturation
- **Why length norm?** Long docs vs short docs
- Compare BM25 vs TF-IDF

**Part E: BM25F (Multi-field)**
- Manning Ch 11 (section 11.4.4)
- Different weights for different fields
- Relevant for OpenSanctions (name > description)

#### **Step 2: Toy Example (2 days)**
🔬 `learning_modules/02_classical_ir/toy_example.py`

Implement from scratch on 5-10 toy documents:
1. **Boolean retrieval**
2. **VSM with TF-IDF + cosine**
3. **BM25**

Compare results:
- Same query on all 3 models
- Why does BM25 rank differently?
- Calculate by hand, verify with code

#### **Step 3: Review Lab 2 (1 day)**
🧪 `class_materials/week_3/Lab 2 Classifical IR Models.ipynb`
- How do they implement VSM?
- How do they implement BM25?
- Compare with your toy example

#### **Step 4: Project Implementation (4-5 days)** ⭐
🛠️ **Create `src/retrieval/classical_ir.py`**

```python
class BM25Retriever:
    """
    Okapi BM25 retriever
    """
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def build_index(self, documents):
        """Build BM25 index"""

    def search(self, query, k=10):
        """Return top-k documents"""

class TFIDFRetriever:
    """TF-IDF + cosine similarity (baseline)"""

class FuzzyMatcher:
    """Handle name variations using rapidfuzz"""
```

**Notebook: `notebooks/02_classical_ir_experiments.ipynb`**

**Experiments to run:**
1. **BM25 on 100K subset**
   - Index all entities
   - Run all queries
   - Generate ranked lists

2. **TF-IDF baseline**
   - Same queries
   - Compare with BM25

3. **Fuzzy matching**
   - Test on Type 2 queries (name variants)
   - Levenshtein distance threshold

4. **Hyperparameter tuning**
   - Grid search: k1 = [1.2, 1.5, 2.0], b = [0.5, 0.75, 1.0]
   - Which works best for your queries?

5. **BM25F (multi-field)**
   - Weight: name=3.0, alias=2.0, description=1.0
   - Does it improve results?

### **Deliverables:**
- ✅ `src/retrieval/classical_ir.py` - Clean, working code
- ✅ `notebooks/02_classical_ir_experiments.ipynb` - All experiments
- ✅ Ranked lists for all queries
- ✅ Best hyperparameters documented

### **Time:** 10-11 days total
- 3-4 days theory (DEEP study)
- 2 days toy example
- 1 day lab review
- 4-5 days implementation & experiments

### **Success Criteria:**
- ✅ You can derive BM25 formula from scratch
- ✅ You understand why k1 and b matter
- ✅ BM25 works on OpenSanctions data
- ✅ You have ranked lists for evaluation
- ✅ You know which parameters work best

---

## **PHASE 6: Evaluation Framework** (Week 3-4, Days 29-33)

### **What to Do:**
1. Implement evaluation metrics
2. Run evaluation on your retrieval systems
3. Pooling for Types 3, 4
4. Analyze results

### **Learning Required:** 🎓 **COMPLETE MODULE 3**

#### **Step 1: Learn Theory (2 days)**
📚 `learning_modules/03_evaluation/theory.md`

**Manning Ch 8 (full chapter):**
- **Precision & Recall** (derive formulas)
  ```
  Precision = TP / (TP + FP)
  Recall = TP / (TP + FN)
  ```
- **Precision@K**: Precision at rank K
- **Recall@K**: Recall at rank K
- **F-measure**: Harmonic mean
  ```
  F1 = 2 × (P × R) / (P + R)
  ```
- **Mean Average Precision (MAP)** (understand deeply!)
  ```
  MAP = (1/|Q|) × Σ AP(q)
  AP(q) = (1/|Rel|) × Σ [P@k × rel(k)]
  ```
- **nDCG** (Normalized Discounted Cumulative Gain)
  ```
  DCG@k = Σ [2^rel(i) - 1] / log₂(i + 1)
  nDCG@k = DCG@k / IDCG@k
  ```
- **Mean Reciprocal Rank (MRR)**
  ```
  MRR = (1/|Q|) × Σ (1/rank_i)
  ```

**Why each metric?**
- Precision@1 for Type 1 queries (must get #1 right!)
- Recall@10 for regulatory compliance (can't miss entities!)
- MAP for overall quality
- nDCG for graded relevance

#### **Step 2: Toy Example (1 day)**
🔬 `learning_modules/03_evaluation/toy_example.py`

Calculate metrics **by hand** on toy results:
- 5 queries
- 10 retrieved docs per query
- Known ground truth
- Calculate: Precision@5, Recall@10, MAP, nDCG@10
- Verify with code

#### **Step 3: Review Lab 3 (0.5 day)**
🧪 `class_materials/week_4/Lab3_Evaluation and Interface.ipynb`
- How do they calculate metrics?
- Compare with your toy example

#### **Step 4: Project Implementation (1-2 days)**
🛠️ **Create `src/evaluation/metrics.py`**

```python
def precision_at_k(retrieved, relevant, k):
    """Precision@K"""

def recall_at_k(retrieved, relevant, k):
    """Recall@K"""

def average_precision(retrieved, relevant):
    """AP for single query"""

def mean_average_precision(results, qrels):
    """MAP across all queries"""

def ndcg_at_k(retrieved, relevance_scores, k):
    """nDCG@K with graded relevance"""
```

**Or use library:**
```python
from ranx import Qrels, Run, evaluate
```

**Notebook: `notebooks/04_evaluation_analysis.ipynb`**

**Evaluation experiments:**
1. **Evaluate BM25**
   - Precision@1 (for Type 1 queries)
   - Recall@10, Recall@20 (primary metric)
   - MAP (overall quality)

2. **Evaluate TF-IDF**
   - Same metrics
   - Compare with BM25

3. **Per-query-type analysis**
   - Which query types work well?
   - Which fail?

4. **Pooling for Types 3, 4**
   - Merge top-20 from BM25, TF-IDF
   - Manual relevance judgement (0, 1, 2)
   - Update qrels.json

5. **Error analysis**
   - Which queries fail? Why?
   - False positives? False negatives?

### **Deliverables:**
- ✅ `src/evaluation/metrics.py` - All metrics implemented
- ✅ `notebooks/04_evaluation_analysis.ipynb` - Complete evaluation
- ✅ Results: `results/metrics/baseline_100k.json`
- ✅ Plots: `results/plots/` (Precision-Recall curves, etc.)
- ✅ Updated qrels with pooling judgements

### **Time:** 4-5 days total
- 2 days theory
- 1 day toy example + lab
- 1-2 days implementation

### **Success Criteria:**
- ✅ You can calculate MAP by hand
- ✅ You understand when to use which metric
- ✅ Your system is properly evaluated
- ✅ You know your system's strengths/weaknesses

---

## **PHASE 7: Dense Retrieval (Optional/Assist Marek)** (Week 4, Days 34-36)

### **What to Do:**
1. Understand dense retrieval concepts
2. Help Marek with integration
3. Learn how to combine sparse + dense

### **Learning Required:** 🎓 **MODULE 4 (Light)**

#### **Step 1: Learn Theory (1-2 days)**
📚 `learning_modules/04_dense_retrieval/theory.md`

**Concepts (not in Manning - modern IR):**
- **Sentence embeddings**
  - BERT, Sentence-BERT
  - all-MiniLM-L6-v2 (384-dim vectors)
- **Semantic similarity**
  - Cosine distance in embedding space
  - Why better than keyword matching?
- **Vector databases**
  - ChromaDB, FAISS
  - Approximate Nearest Neighbor (ANN)
- **Hybrid retrieval**
  - Sparse (BM25) + Dense (embeddings)
  - Reciprocal Rank Fusion (RRF)
    ```
    RRF(d) = Σ 1/(k + rank_i(d))
    ```

#### **Step 2: Toy Example (1 day)**
🔬 `learning_modules/04_dense_retrieval/toy_example.py`

Simple semantic search:
- Encode 5 documents with sentence-transformers
- Encode query
- Find most similar via cosine distance
- Compare with BM25 results

#### **Step 3: Understand Marek's Code**
- Review `src/retrieval/dense_retrieval.py`
- Understand ChromaDB usage
- How does he encode documents?

#### **Step 4: Help with Integration (Optional)**
- Test dense retrieval on your queries
- Compare: BM25 vs Dense vs Hybrid
- Provide feedback to Marek

### **Deliverables:**
- ✅ Understanding of dense retrieval
- ✅ Can explain to others
- ✅ Optional: experiments comparing sparse vs dense

### **Time:** 2-3 days (optional)

---

## **PHASE 8: Scale to Full Dataset** (Week 4-5, Days 37-40)

### **What to Do:**
1. Scale preprocessing to 1.3M entities
2. Build full BM25 index
3. Re-run evaluation
4. Final hyperparameter tuning

### **Learning Required:**
📚 **No new module** - apply everything learned

But review:
- Manning Ch 4 (Index Construction) - scaling techniques
- Manning Ch 5 (Index Compression) - optimization

### **Project Work:**
🛠️ **Scale everything up:**

1. **Preprocess full dataset** (Kieren's task mainly)
   - Run parser on full targets.nested.json
   - Save to `data/processed/full/`

2. **Build full BM25 index**
   - Index all 1.3M entities
   - Measure: indexing time, memory usage
   - Save index to `models/bm25_index/`

3. **Re-run all queries**
   - Do results change with more data?
   - Better or worse?

4. **Final tuning**
   - Tune k1, b on full dataset
   - BM25F field weights

5. **Final evaluation**
   - All metrics on full dataset
   - Generate final results
   - Create visualizations

### **Deliverables:**
- ✅ Full dataset indexed
- ✅ Final evaluation results
- ✅ Performance report (speed, accuracy)

### **Time:** 3-4 days

---

## **PHASE 9: Report Writing** (Week 5, Days 41-45)

### **What to Do:**
Write Assignment 2 report

### **Report Structure:**
1. **Introduction**
   - Problem statement
   - Approach overview

2. **System Architecture**
   - Expand Assignment 1 design
   - What changed during implementation?

3. **Implementation Details**
   - Preprocessing pipeline
   - Classical IR (BM25, TF-IDF)
   - Dense retrieval (Marek's part)
   - Fusion (Marek's part)
   - RAG (Marek's part)

4. **Evaluation Setup**
   - Query set (20 queries, 7 types)
   - Ground truth construction
   - Metrics (Precision@1, Recall@10, MAP, nDCG)
   - Pooling methodology

5. **Results & Analysis**
   - YOUR MAIN CONTRIBUTION
   - BM25 performance by query type
   - Comparison: BM25 vs TF-IDF vs Dense vs Hybrid
   - Hyperparameter analysis
   - Error analysis

6. **Discussion**
   - What worked well? (likely: BM25 for exact matches)
   - What didn't? (likely: name transliterations, relational queries)
   - Limitations
   - Future improvements

7. **Conclusion**

### **Your Contribution Focus:**
- Classical IR implementation (BM25, TF-IDF)
- Hyperparameter tuning
- Evaluation results analysis
- Error analysis

### **Time:** 4-5 days

---

# 📊 Summary Timeline

| Phase | What | Learning | Project | Time |
|-------|------|----------|---------|------|
| **0** | Setup | Course overview | Environment setup | 0.5d |
| **1** | Data exploration | Basic concepts | Explore data | 2-3d |
| **2** | Preprocessing | **Module 1** | Build pipeline | 6-7d |
| **3** | Queries | Skim Ch 9 | Create query set | 2-3d |
| **4** | Ground truth | Module 3 (part) | Extract qrels | 1-2d |
| **5** | Classical IR | **Module 2** ⭐ | BM25 + experiments | 10-11d |
| **6** | Evaluation | **Module 3** | Metrics + analysis | 4-5d |
| **7** | Dense (opt) | Module 4 | Assist Marek | 2-3d |
| **8** | Scale | Review Ch 4-5 | Full dataset | 3-4d |
| **9** | Report | - | Write report | 4-5d |

**Total:** ~35-45 days (5-6 weeks)

---

# 🎯 Your Focus Areas

## **HIGH PRIORITY (Your Main Tasks):**
1. ⭐ **Module 2 + Phase 5**: Classical IR (BM25) - MOST IMPORTANT
2. ⭐ **Module 3 + Phase 6**: Evaluation & Analysis
3. ⭐ **Module 1 + Phase 2**: Preprocessing (collaborate with Kieren)

## **MEDIUM PRIORITY:**
4. **Phase 3**: Query set development (collaborate with Marek)
5. **Phase 4**: Ground truth construction
6. **Phase 8**: Scaling to full dataset

## **LOW PRIORITY (Optional):**
7. **Module 4 + Phase 7**: Dense retrieval (Marek's main task, you assist)

---

# 🚀 Getting Started

## **Week 1 (Now!):**

### **Days 1-2: Foundation**
- ✅ Setup complete
- Read this document fully
- Understand the big picture

### **Days 3-5: Data Exploration (Phase 1)**
- **Learn**: Skim Manning Ch 1 (pages 1-20)
- **Do**: Run `notebooks/01_data_exploration.ipynb`
- **Output**: Document findings

### **Days 6-7: Start Module 1 Theory**
- **Learn**: Start `learning_modules/01_indexing_tfidf/theory.md`
- **Read**: Manning Ch 1 (full)
- **Do**: Take notes, derive formulas

## **Week 2: Module 1 + Phase 2 (Preprocessing)**
- **Days 8-9**: Finish Module 1 theory + Manning Ch 2, 6
- **Days 10-11**: Toy example + Lab 1 review
- **Days 12-14**: Build preprocessing pipeline

## **Week 2-3: Module 2 + Phase 5 (Classical IR)** ⭐
- **Days 15-18**: Module 2 theory (deep dive into BM25)
- **Days 19-20**: Toy example (implement BM25 from scratch)
- **Day 21**: Lab 2 review
- **Days 22-25**: Project implementation (BM25 on OpenSanctions)
- **Days 26-28**: Experiments & tuning

## **Week 3-4: Module 3 + Phase 6 (Evaluation)**
- **Days 29-30**: Module 3 theory
- **Day 31**: Toy example + Lab 3
- **Days 32-33**: Implement evaluation, analyze results

## **Week 4-5: Scaling + Report**
- **Days 34-36**: Optional dense retrieval understanding
- **Days 37-40**: Scale to full dataset
- **Days 41-45**: Write report

---

# 📝 Daily Study Routine

## **Theory Days:**
1. **Morning (3h)**: Read Manning chapters
   - Take detailed notes
   - Derive formulas by hand
   - Work through examples

2. **Afternoon (2h)**: Review slides
   - Connect to book concepts
   - Prepare questions

3. **Evening (1h)**: Consolidate
   - Summarize key points
   - Identify gaps

## **Implementation Days:**
1. **Morning (3h)**: Code
   - Write from scratch first
   - Test on toy data
   - Debug

2. **Afternoon (3h)**: Project work
   - Apply to OpenSanctions
   - Experiment
   - Document

3. **Evening (1h)**: Review
   - Did it work? Why/why not?
   - What did you learn?

---

# ✅ Success Checkpoints

## **After Phase 2 (Preprocessing):**
- [ ] I understand inverted index structure
- [ ] I can explain TF-IDF formula
- [ ] I have working preprocessing pipeline
- [ ] I have indexed 100K entities

## **After Phase 5 (Classical IR):** ⭐
- [ ] I can derive BM25 from scratch
- [ ] I understand k1 and b parameters
- [ ] I have working BM25 implementation
- [ ] I have ranked lists for all queries
- [ ] I know which hyperparameters work best

## **After Phase 6 (Evaluation):**
- [ ] I can calculate MAP by hand
- [ ] I understand trade-offs between metrics
- [ ] I have evaluated my system properly
- [ ] I know strengths and weaknesses

## **After Phase 8 (Scaling):**
- [ ] System works on full 1.3M dataset
- [ ] Final evaluation complete
- [ ] Ready to write report

---

# 🔥 Pro Tips

1. **Don't skip theory**: You need to understand WHY before HOW
2. **Derive by hand**: Don't just read formulas, derive them
3. **Toy examples are crucial**: Build intuition on simple data first
4. **Test incrementally**: Don't build everything then test
5. **Document as you go**: Write findings immediately
6. **Ask questions**: Use lecture time, office hours
7. **Collaborate**: Share insights with team
8. **Review labs**: They're valuable learning resources

---

# 🎓 Learning Objectives Checklist

By end of project, you should be able to:

## **Theoretical Understanding:**
- [ ] Explain inverted index structure
- [ ] Derive TF-IDF formula
- [ ] Derive BM25 formula and explain every component
- [ ] Explain difference between BM25 and TF-IDF
- [ ] Calculate Precision, Recall, MAP, nDCG by hand
- [ ] Explain when to use which evaluation metric
- [ ] Understand dense vs sparse retrieval

## **Practical Skills:**
- [ ] Build inverted index from scratch
- [ ] Implement BM25 from scratch
- [ ] Preprocess large-scale datasets (1.3M docs)
- [ ] Tune hyperparameters systematically
- [ ] Evaluate IR systems properly
- [ ] Analyze errors and improve system
- [ ] Write clear technical reports

## **Project Deliverables:**
- [ ] Working IR system on OpenSanctions
- [ ] 20 queries with ground truth
- [ ] Complete evaluation results
- [ ] Technical report
- [ ] Clean, documented code

---

# 🚀 Ready to Start?

**Next immediate action:**
1. Review this document fully
2. Set up your learning journal
3. Start Phase 1: Data Exploration
4. Then move to Module 1 theory

**Let's build something amazing! 🎯**
