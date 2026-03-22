# Complete IR Learning & Project Roadmap

## Overview
This roadmap integrates **course lectures**, **Manning's IR book**, **lab exercises**, and **your project** into a cohesive learning plan. Focus on understanding the **math and logic** behind each concept, then apply it practically.

---

## 📚 Course Structure (5 Weeks)

| Week | Topic | Labs | Book Chapters |
|------|-------|------|---------------|
| **Week 1** | Introduction & Indexing | - | Ch 1-2 |
| **Week 2** | Retrieval Models I (Boolean, VSM, BIRM) | Lab 1: Indexing | Ch 3, 6 |
| **Week 3** | Retrieval Models II (BM25, LM, Evaluation) | Lab 2: Classical IR | Ch 11-12 |
| **Week 4** | Retrieval Models III (DfR, PageRank) | Lab 3: Evaluation | Ch 21 |
| **Week 5** | Applications (ElasticSearch) | Lab 4: ElasticSearch | - |

---

## 🎯 Integrated Learning Plan

### **WEEK 1: Foundations - Introduction & Indexing**

#### **Theory (Manning's Book)**
**📖 Chapter 1: Boolean Retrieval**
- **Math to Learn**:
  - Term-document incidence matrix
  - Boolean queries (AND, OR, NOT operations)
  - Inverted index structure
  - Set operations on postings lists

**📖 Chapter 2: The Term Vocabulary & Postings Lists**
- **Math to Learn**:
  - Tokenization algorithms
  - Normalization (case folding, stemming, lemmatization)
  - Porter stemmer algorithm
  - Zipf's Law: frequency ∝ 1/rank
  - Heap's Law: V = K × n^β (vocabulary growth)

**Key Concepts**:
- **Inverted Index**: term → list of documents
- **Postings List**: sorted list of document IDs containing term
- **Dictionary**: all unique terms with pointers to postings

#### **Course Material**
- 📊 **Slides**: `week_1/slide.pdf`
  - Course overview
  - Data vs Information vs Knowledge
  - Indexing fundamentals
  - TF-IDF introduction

#### **Lab Practice**
- ✅ **No Lab Week 1** (starts Week 2)

#### **Project Work**
- ✅ Download data (targets.nested.json, targets.simple.csv)
- ✅ Run data exploration notebook
- 📝 Document data characteristics
- 🔧 Set up preprocessing pipeline structure

#### **Study Tasks**
1. **Read**: Manning Ch 1-2 (focus on math derivations)
2. **Understand**: Why inverted index? Time complexity analysis
3. **Derive**: How many disk seeks for Boolean query?
4. **Code**: Implement simple inverted index from scratch (practice)

---

### **WEEK 2: Classical IR - Boolean, VSM, BIRM**

#### **Theory (Manning's Book)**
**📖 Chapter 6: Scoring, Term Weighting & the Vector Space Model**
- **Math to Master**:
  - **TF (Term Frequency)**: tf(t,d) = count of t in d
  - **IDF (Inverse Document Frequency)**: idf(t) = log(N / df(t))
    - N = total documents
    - df(t) = document frequency of term t
  - **TF-IDF Weight**: w(t,d) = tf(t,d) × log(N / df(t))
  - **Vector Space Model**:
    - Documents as vectors in |V|-dimensional space
    - Query as vector
  - **Cosine Similarity**:
    ```
    cos(θ) = (q · d) / (|q| × |d|)
           = Σ(q_i × d_i) / sqrt(Σq_i²) × sqrt(Σd_i²)
    ```
  - **Length Normalization**: Divide by Euclidean length

**📖 Chapter 11: Probabilistic Information Retrieval**
- **Math to Master**:
  - **Binary Independence Model (BIM)**
  - **Probability Ranking Principle (PRP)**
  - **Relevance Score**:
    ```
    score(d,q) = Σ log[P(t|R) × P(~t|NR)] / [P(~t|R) × P(t|NR)]
    ```
  - **RSJ (Robertson-Sparck Jones) formula**

#### **Course Material**
- 📊 **Slides**: `week_2/slide.pdf`
  - Retrieval models definition
  - Relevance: complex, idiosyncratic, variable
  - Boolean Model
  - Vector Space Model (VSM)
  - Binary Independence Retrieval Model (BIRM)

#### **Lab Practice**
- 🧪 **Lab 1**: `week_2/Lab_1_Python_Indexing.ipynb`
  - Build inverted index in Python
  - Parse documents
  - Create dictionary and postings lists
  - Query processing
- 📝 **Lab 1 ANSWERS**: Review `Lab_1_Python_Indexing_ANSWERS.ipynb`

#### **Project Work**
- 🔧 **Phase 2, Step 4**: Build streaming JSON parser
  - Handle 3.68GB file without memory overflow
  - Parse line by line
  - Create `src/preprocessing/parser.py`

- 🔧 **Phase 2, Step 5**: Text preprocessing
  - Lowercase normalization
  - Stopword removal (nltk)
  - Lemmatization (spaCy)
  - Create `src/preprocessing/text_processing.py`

#### **Study Tasks**
1. **Derive**: Why IDF is log(N/df)? What happens without log?
2. **Prove**: Cosine similarity range is [-1, 1]
3. **Implement**: TF-IDF from scratch (no libraries)
4. **Understand**: Why normalize by document length?
5. **Compare**: Boolean vs VSM vs BIM - when to use each?

#### **Deep Dive Questions**
- Why does TF-IDF work? Mathematical intuition?
- What are limitations of cosine similarity?
- How does BIM model relevance probabilistically?

---

### **WEEK 3: Advanced Ranking - BM25, Language Models, Evaluation**

#### **Theory (Manning's Book)**
**📖 Chapter 11 (continued): BM25**
- **Math to Master - BM25 Formula**:
  ```
  score(D,Q) = Σ IDF(qi) × [f(qi,D) × (k1 + 1)] /
               [f(qi,D) + k1 × (1 - b + b × |D|/avgdl)]
  ```
  Where:
  - f(qi,D) = term frequency of qi in document D
  - |D| = length of document D in words
  - avgdl = average document length in the collection
  - k1 = controls term frequency saturation (typically 1.2-2.0)
  - b = controls length normalization (typically 0.75)
  - IDF(qi) = log[(N - n(qi) + 0.5) / (n(qi) + 0.5)]
  - N = total number of documents
  - n(qi) = number of documents containing qi

- **Key Insights**:
  - **Saturation**: TF component saturates (diminishing returns)
  - **Length normalization**: Controlled by b parameter
  - **IDF component**: Similar to classic IDF but adjusted

**📖 Chapter 12: Language Models for IR**
- **Math to Master**:
  - **Query Likelihood Model**:
    ```
    P(Q|D) = Π P(t|D)^count(t,Q)
    ```
  - **Smoothing** (avoid zero probabilities):
    - **Jelinek-Mercer**: P(t|D) = λ × P_ml(t|D) + (1-λ) × P(t|C)
    - **Dirichlet**: P(t|D) = [count(t,D) + μ × P(t|C)] / [|D| + μ]
  - **Maximum Likelihood Estimate**:
    ```
    P_ml(t|D) = count(t,D) / |D|
    ```
  - **Collection Model**: P(t|C) = Σcount(t,D) / Σ|D|

**📖 Chapter 8: Evaluation in IR**
- **Math to Master**:
  - **Precision**: P = TP / (TP + FP) = relevant retrieved / total retrieved
  - **Recall**: R = TP / (TP + FN) = relevant retrieved / total relevant
  - **F-measure**: F1 = 2PR / (P + R)
  - **Precision@K**: Precision at rank K
  - **Recall@K**: Recall at rank K
  - **Mean Average Precision (MAP)**:
    ```
    MAP = (1/|Q|) × Σ AP(q)
    AP(q) = (1/|Rel|) × Σ [P@k × rel(k)]
    ```
  - **Normalized Discounted Cumulative Gain (nDCG)**:
    ```
    DCG@k = Σ [2^rel(i) - 1] / log₂(i + 1)
    nDCG@k = DCG@k / IDCG@k
    ```
  - **Mean Reciprocal Rank (MRR)**:
    ```
    MRR = (1/|Q|) × Σ (1 / rank_i)
    ```

#### **Course Material**
- 📊 **Slides**: `week_3/slide.pdf`
  - BM25 (Okapi BM25)
  - Language Models (LM)
  - IR System Evaluation
  - Precision, Recall, F-measure
  - MAP, nDCG

#### **Lab Practice**
- 🧪 **Lab 2**: `week_3/Lab 2 Classifical IR Models.ipynb`
  - Implement Boolean retrieval
  - Implement Vector Space Model
  - Calculate TF-IDF
  - Compute cosine similarity
  - Compare retrieval models
- 📝 **Lab 2 ANSWERS**: Review `Lab_2_Boolean_Vector_Space_ANSWERS.ipynb`

#### **Project Work**
- 🔧 **Phase 2, Step 6**: Document flattening
  - Flatten nested JSON into searchable text blobs
  - Concatenate: name, alias, previousName, description, summary
  - Preserve identifiers separately
  - Create `src/preprocessing/indexer.py`

- 🔧 **Phase 2, Step 7**: Test on 100K subset
  - Extract first 100K entities
  - Validate pipeline
  - Store in `data/processed/subset_100k/`

- 🔧 **Phase 5, Step 14**: Classical IR indexing (YOUR MAIN TASK!)
  - **BM25 implementation**: Use `rank-bm25` library
  - **TF-IDF baseline**: Use scikit-learn
  - **Fuzzy matching**: Use rapidfuzz
  - Test on 100K subset
  - Tune k1 and b parameters
  - Create `src/retrieval/classical_ir.py`

#### **Study Tasks**
1. **Derive**: Why does BM25 saturate? Graph TF component
2. **Prove**: Language model ranking = query likelihood
3. **Implement**: BM25 from scratch (understand every term)
4. **Calculate**: MAP by hand for sample query results
5. **Understand**: Why nDCG uses log discount?

#### **Deep Dive Questions**
- What's the intuition behind k1 and b in BM25?
- Why does LM need smoothing? What happens without it?
- When is Recall more important than Precision?
- Why is MAP sensitive to rank position?

---

### **WEEK 4: Advanced Topics - DfR, PageRank, Evaluation**

#### **Theory (Manning's Book)**
**📖 Chapter 21: Link Analysis (PageRank)**
- **Math to Master - PageRank**:
  ```
  PR(A) = (1-d)/N + d × Σ[PR(Ti) / C(Ti)]
  ```
  Where:
  - d = damping factor (typically 0.85)
  - N = total number of pages
  - Ti = pages that link to page A
  - C(Ti) = number of outbound links from page Ti

- **Matrix Form**:
  ```
  PR = (1-d)/N × e + d × M × PR
  ```
  - M = adjacency matrix (transition probabilities)
  - Solved iteratively (power iteration method)

- **Random Surfer Model**: probability of random walk

**Divergence from Randomness (DfR)**
- **Key Idea**: Informative terms deviate from random distribution
- **Formula**:
  ```
  score(t,d) = -log₂[P(tf | T)] × (1 - P(t | d))
  ```
- **Components**:
  - **Randomness model**: Expected term frequency under randomness
  - **Information content**: How much does observed deviate?
  - **Normalization**: Account for document length

#### **Course Material**
- 📊 **Slides**: `week_5/slide.pdf` (Week 4 lecture)
  - Divergence from Randomness (DfR)
  - PageRank algorithm
  - Link analysis
  - Evaluation continued

- 📊 **Slides**: `week_4/Lecture 4 Evaluation Applications and Trends in LLM-driven IR.pptx`
  - Modern evaluation techniques
  - LLM-driven IR trends
  - Applications

#### **Lab Practice**
- 🧪 **Lab 3**: `week_4/Lab3_Evaluation and Interface.ipynb`
  - Implement evaluation metrics
  - Calculate Precision, Recall, F1
  - Compute MAP, nDCG
  - Build simple search interface
  - Compare retrieval systems

#### **Project Work**
- 🔧 **Phase 3**: Query set development
  - Create query brainstorming notebook
  - Generate 20+ queries (7 types)
  - Export to Excel for expert review
  - Convert reviewed Excel → JSON

- 🔧 **Phase 6, Step 17**: Implement metrics
  - Use `ranx` library
  - Precision@1, Recall@10, Recall@20
  - MAP, nDCG@10
  - Create `src/evaluation/metrics.py`

- 🔧 **Phase 6, Step 19**: Initial evaluation
  - Run on 100K subset
  - Measure baseline performance
  - Identify weaknesses
  - Create `notebooks/02_classical_ir_experiments.ipynb`

#### **Study Tasks**
1. **Derive**: PageRank convergence conditions
2. **Prove**: PageRank sums to 1 across all pages
3. **Simulate**: Run PageRank by hand on small graph
4. **Understand**: Why damping factor? What if d=1?
5. **Calculate**: nDCG for sample ranking with graded relevance

#### **Deep Dive Questions**
- How does PageRank handle dangling nodes?
- What is the relationship between PageRank and random walks?
- Why is DfR called "divergence from randomness"?
- How to choose between different evaluation metrics?

---

### **WEEK 5: Modern Applications - ElasticSearch & Dense Retrieval**

#### **Theory (Beyond Course - Modern IR)**
**📖 Dense Retrieval (Not in Manning)**
- **Sentence Embeddings**:
  - Transformer-based models (BERT, Sentence-BERT)
  - all-MiniLM-L6-v2 (384-dimensional vectors)
  - Semantic similarity via cosine distance

- **Vector Databases**:
  - ChromaDB, FAISS, Pinecone
  - Approximate Nearest Neighbor (ANN) search
  - Trade-off: speed vs accuracy

- **Hybrid Retrieval**:
  - Combine sparse (BM25) + dense (embeddings)
  - Reciprocal Rank Fusion (RRF):
    ```
    RRF(d) = Σ 1 / (k + rank_i(d))
    ```
    - k = constant (typically 60)
    - rank_i(d) = rank of document d in system i

**📖 RAG (Retrieval-Augmented Generation)**
- **Pipeline**:
  1. Query → Retrieve top-k documents
  2. Documents → Context for LLM
  3. LLM generates answer grounded in context

- **Evaluation** (RAGAS):
  - **Faithfulness**: Is output grounded in retrieved docs?
  - **Answer Relevance**: Does it address the query?

#### **Course Material**
- 📊 **Slides**: `week_5/slide.pdf`
  - ElasticSearch overview
  - Practical IR systems
  - Industry applications

#### **Lab Practice**
- 🧪 **Lab 4**: `week_5/Lab_4_ElasticSearch.ipynb`
  - Set up ElasticSearch
  - Index documents
  - Query API
  - Analyze results
  - Production-ready IR

#### **Project Work**
- 🔧 **Phase 5, Step 15**: Dense retrieval (Marek's task, but learn it!)
  - Sentence embeddings (all-MiniLM-L6-v2)
  - ChromaDB vector database
  - Semantic search
  - Create `src/retrieval/dense_retrieval.py`

- 🔧 **Phase 7**: Fusion & RAG
  - **Step 20**: RRF fusion → `src/retrieval/fusion.py`
  - **Step 21**: RAG layer (flan-t5-base) → `src/rag/generator.py`
  - **Step 22**: RAGAS evaluation → `src/evaluation/ragas_eval.py`

- 🔧 **Phase 8**: Scale & optimize
  - Scale to 1.3M dataset
  - Hyperparameter tuning
  - Final evaluation

#### **Study Tasks**
1. **Understand**: How ElasticSearch uses BM25 internally
2. **Learn**: Transformer architecture basics (for embeddings)
3. **Experiment**: Compare sparse vs dense retrieval
4. **Implement**: RRF fusion from scratch
5. **Explore**: When does dense retrieval outperform BM25?

---

## 📖 Manning Book Reading Plan

### **Essential Chapters (Core IR)**
- ✅ **Ch 1**: Boolean Retrieval (Week 1)
- ✅ **Ch 2**: Term Vocabulary & Postings (Week 1)
- ✅ **Ch 6**: Vector Space Model (Week 2)
- ✅ **Ch 11**: Probabilistic IR & BM25 (Week 3)
- ✅ **Ch 12**: Language Models (Week 3)
- ✅ **Ch 8**: Evaluation (Week 3-4)
- ✅ **Ch 21**: Link Analysis & PageRank (Week 4)

### **Supplementary Chapters (Go Deeper)**
- 📚 **Ch 3**: Dictionaries and Tolerant Retrieval
- 📚 **Ch 4**: Index Construction
- 📚 **Ch 5**: Index Compression
- 📚 **Ch 7**: Computing Scores in a Complete System
- 📚 **Ch 9**: Relevance Feedback & Query Expansion
- 📚 **Ch 13**: Text Classification
- 📚 **Ch 19**: Web Search Basics

### **Reading Strategy**
1. **First pass**: Read for concepts (skip proofs)
2. **Second pass**: Work through math derivations
3. **Third pass**: Implement algorithms from scratch
4. **Fourth pass**: Relate to your project implementation

---

## 🛠️ Project Milestones (Integrated with Learning)

### **Week 1: Foundation**
- ✅ Data exploration
- ✅ Understand data structure
- 📖 Read Manning Ch 1-2
- 🔧 Set up preprocessing pipeline

### **Week 2: Preprocessing & Classical IR**
- 🔧 Build streaming parser
- 🔧 Text preprocessing
- 🔧 Document flattening
- 🧪 Complete Lab 1 (Indexing)
- 📖 Read Manning Ch 6, 11
- 🔧 Start BM25 implementation

### **Week 3: Retrieval & Evaluation**
- 🔧 Complete Classical IR (BM25, TF-IDF, fuzzy)
- 🔧 Query set development
- 🔧 Ground truth construction
- 🧪 Complete Lab 2 (Classical IR)
- 📖 Read Manning Ch 8, 12
- 🔧 Implement evaluation metrics

### **Week 4: Advanced & Evaluation**
- 🔧 Initial evaluation on 100K subset
- 🔧 Dense retrieval (assist Marek)
- 🧪 Complete Lab 3 (Evaluation)
- 📖 Read Manning Ch 21
- 🔧 Hyperparameter tuning

### **Week 5: Integration & Scaling**
- 🔧 RRF fusion
- 🔧 RAG layer
- 🔧 Scale to 1.3M dataset
- 🧪 Complete Lab 4 (ElasticSearch)
- 🔧 Final evaluation

---

## 🎯 Study Goals by Topic

### **For Each Topic, Master:**

#### **1. Mathematical Foundation**
- Derive formulas from first principles
- Understand why, not just what
- Work through examples by hand
- Prove key theorems

#### **2. Computational Implementation**
- Code from scratch (no libraries first)
- Then use libraries efficiently
- Understand time/space complexity
- Optimize bottlenecks

#### **3. Practical Application**
- When to use which algorithm?
- What are failure modes?
- How to tune hyperparameters?
- How to evaluate properly?

---

## 📝 Weekly Study Checklist

### **Before Each Week**
- [ ] Review lecture slides
- [ ] Read assigned Manning chapters
- [ ] Derive key formulas on paper
- [ ] Prepare questions for lecture

### **During Week**
- [ ] Attend lecture (Thursday 10-12)
- [ ] Complete lab exercises (Monday 2-4pm)
- [ ] Implement algorithms from scratch
- [ ] Work on project tasks

### **After Week**
- [ ] Review lab solutions
- [ ] Re-derive complex formulas
- [ ] Connect theory to project
- [ ] Document learnings

---

## 🔥 Deep Learning Tips

### **Math Mastery**
1. **Don't memorize**: Understand derivations
2. **Work examples**: Calculate by hand first
3. **Visualize**: Draw graphs, diagrams
4. **Teach others**: Explain to teammates

### **Code Mastery**
1. **Type don't copy-paste**: Build muscle memory
2. **Debug intentionally**: Break things to learn
3. **Refactor**: Improve incrementally
4. **Test**: Write unit tests for learning

### **Project Integration**
1. **Theory → Practice**: Implement what you learn
2. **Practice → Theory**: Understand why code works
3. **Compare**: Your implementation vs library
4. **Optimize**: Make it fast and scalable

---

## 📊 Progress Tracking

Create a study journal:
```
## Date: YYYY-MM-DD
**Topic**: BM25
**Lecture**: Week 3
**Book**: Ch 11

### What I Learned:
- BM25 formula components
- Why saturation matters
- How b controls length norm

### Math I Derived:
- [ ] BM25 formula from scratch
- [ ] IDF component
- [ ] Saturation curve

### Code I Wrote:
- [ ] BM25 from scratch
- [ ] Tested on toy dataset
- [ ] Compared with rank-bm25 library

### Questions:
- Why k1=1.2 typical?
- How to tune b parameter?

### Project Application:
- Implemented in src/retrieval/classical_ir.py
- Tested on OpenSanctions 100K subset
- Results: P@10=0.85, R@10=0.62
```

---

## 🎓 Final Exam Prep (Beyond Project)

### **Key Topics to Master**
1. **Indexing**: Inverted index, postings lists
2. **Ranking**: TF-IDF, BM25, Language Models
3. **Evaluation**: Precision, Recall, MAP, nDCG
4. **Link Analysis**: PageRank
5. **Modern**: Dense retrieval, RAG

### **Types of Questions**
- **Derivations**: Prove BM25 formula
- **Calculations**: Compute MAP by hand
- **Comparisons**: When to use BM25 vs LM?
- **Applications**: Design IR system for X

---

## 🚀 Beyond the Course

### **Additional Topics to Explore**
1. **Neural IR**: BERT, ColBERT, DPR
2. **Query Understanding**: Intent classification, entity recognition
3. **Cross-Lingual IR**: Multilingual retrieval
4. **Personalization**: User-specific ranking
5. **Learning to Rank**: RankNet, LambdaMART

### **Papers to Read**
1. "Attention Is All You Need" (Transformers)
2. "BERT: Pre-training of Deep Bidirectional Transformers"
3. "Dense Passage Retrieval for Open-Domain QA"
4. "ColBERT: Efficient and Effective Passage Search"

---

## 🎯 Success Metrics

### **You've Mastered IR When You Can:**
- [ ] Derive BM25 formula from first principles
- [ ] Implement any IR algorithm from scratch
- [ ] Explain trade-offs between different models
- [ ] Design evaluation strategy for new problem
- [ ] Build production-ready IR system
- [ ] Debug why retrieval fails on specific query
- [ ] Tune hyperparameters intelligently
- [ ] Understand modern neural IR approaches

---

## 📞 Resources & Support

### **Office Hours**
- Ask professor about math derivations
- Discuss project implementation
- Clarify lecture concepts

### **Study Group**
- Form group with Kieren & Marek
- Each teach one concept per week
- Code review each other's work
- Share insights from book

### **Online Resources**
- Manning book online edition
- Stanford CS276 (IR course materials)
- Papers With Code (IR section)
- ElasticSearch documentation

---

## ✅ Final Checklist

### **By End of Course**
- [ ] Completed all 5 labs
- [ ] Read Manning Ch 1-2, 6, 8, 11-12, 21
- [ ] Derived all key formulas
- [ ] Implemented classical IR system
- [ ] Built complete project pipeline
- [ ] Evaluated system properly
- [ ] Written comprehensive report
- [ ] Ready for exam

**Remember**: This is not just about passing the course. It's about truly **understanding Information Retrieval** - the math, the logic, the practice. Take your time, go deep, and enjoy the journey! 🚀
