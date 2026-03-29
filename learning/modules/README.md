# Learning Modules Roadmap - Pure Theory & Practice

## Overview
This document outlines **what to learn** in each module - theory, math, concepts, and toy implementations. **No project work** - just pure learning to build deep understanding of Information Retrieval.

---

# 📚 Module Structure

Each module follows the same pattern:

```
Module X/
├── theory.md           # Math, derivations, concepts, logic
├── toy_example.py      # Implementation on 5-10 simple documents
└── exercises.ipynb     # Practice problems to test understanding
```

**Learning flow for each module:**
1. Read `theory.md` - understand concepts deeply
2. Derive all formulas by hand on paper
3. Run `toy_example.py` - see it work on simple data
4. Complete `exercises.ipynb` - test your understanding
5. Review class materials (slides + labs) - connect to course
6. **Only then** apply to project

---

# 📖 Module 1: Indexing & TF-IDF

## **Goal:**
Understand how to build an inverted index and calculate TF-IDF weights.

## **Duration:** 4-5 days of focused study

---

## **Part 1: Boolean Retrieval & Inverted Index** (Day 1)

### **Concepts to Learn:**
1. **What is Information Retrieval?**
   - Difference: Data → Information → Knowledge
   - IR vs Database systems
   - Unstructured vs structured data

2. **The Boolean Model**
   - Boolean queries: AND, OR, NOT
   - How to process: "Brutus AND Caesar"
   - Query evaluation using postings lists

3. **Inverted Index Structure**
   - **Dictionary**: All unique terms with metadata
   - **Postings list**: Document IDs containing each term
   - Why inverted? (vs forward index)

### **Math to Master:**
```
Term-Document Incidence Matrix:
  - Rows = terms
  - Columns = documents
  - Cell = 1 if term in document, 0 otherwise

Postings List:
  term → [doc1, doc3, doc7, ...]  (sorted by doc_id)

Boolean AND operation:
  Intersection of two postings lists
  Time complexity: O(x + y) where x, y are list lengths
```

### **Manning Book:**
- **Chapter 1** (pages 1-20):
  - Section 1.1: Boolean retrieval
  - Section 1.2: Inverted index
  - Section 1.3: Processing Boolean queries
  - Section 1.4: Extended Boolean model

### **Derive by Hand:**
1. Given 5 documents, build term-document matrix
2. Convert matrix to inverted index
3. Process query "dog AND cat" step by step
4. Calculate time complexity

### **Questions to Answer:**
- Why is inverted index better than scanning all documents?
- What's the space complexity of inverted index?
- How to handle phrase queries like "to be or not to be"?

---

## **Part 2: Term Vocabulary & Text Processing** (Day 2)

### **Concepts to Learn:**
1. **Tokenization**
   - What is a token?
   - Language-specific rules
   - Handling punctuation, numbers, dates

2. **Normalization**
   - Case folding (lowercase)
   - Accents and diacritics
   - Equivalence classes

3. **Stemming vs Lemmatization**
   - **Stemming**: Crude chopping (Porter Stemmer)
   - **Lemmatization**: Linguistic analysis (spaCy)
   - Trade-offs: speed vs accuracy

4. **Stopwords**
   - What are stopwords? (the, a, is, ...)
   - Why remove them?
   - When NOT to remove?

5. **Statistical Laws**
   - **Zipf's Law**: frequency ∝ 1/rank
   - **Heap's Law**: vocabulary growth

### **Math to Master:**
```
Zipf's Law:
  cf_i = k / i
  where cf_i = collection frequency of i-th most common term
  (Power law distribution)

Heap's Law:
  V = K × n^β
  where:
    V = vocabulary size
    n = number of tokens
    K, β = constants (typically β = 0.4-0.6)
```

### **Porter Stemmer Algorithm:**
```
Step 1: Remove plurals
  "caresses" → "caress"
  "ponies" → "poni"

Step 2: Remove -ed, -ing
  "agreed" → "agree"
  "running" → "run"

... (5 steps total)
```

### **Manning Book:**
- **Chapter 2** (full chapter):
  - Section 2.1: Document delineation & tokenization
  - Section 2.2: Normalization, stemming, lemmatization
  - Section 2.3: Stopwords
  - Section 2.4: Statistical properties (Zipf, Heap)

### **Derive by Hand:**
1. Apply Porter Stemmer to 10 words manually
2. Plot Zipf's Law on sample text
3. Calculate vocabulary growth using Heap's Law
4. Compare stemming vs lemmatization on 10 sentences

### **Questions to Answer:**
- When should you NOT remove stopwords?
- Why does Zipf's Law hold for natural language?
- What's the trade-off between stemming and lemmatization?

---

## **Part 3: TF-IDF Weighting** (Day 3)

### **Concepts to Learn:**
1. **Term Frequency (TF)**
   - Raw count
   - Log-scaled TF
   - Normalized TF
   - Why do we need multiple variants?

2. **Document Frequency (DF)**
   - How many documents contain term?
   - Common words have high DF
   - Rare words have low DF

3. **Inverse Document Frequency (IDF)**
   - Penalize common terms
   - Reward rare terms
   - Why logarithm?

4. **TF-IDF Weight**
   - Combining TF and IDF
   - Different weighting schemes
   - When to use which?

### **Math to Master:**
```
Raw Term Frequency:
  tf(t,d) = count of term t in document d

Log-scaled TF:
  tf_log(t,d) = 1 + log(tf(t,d))  if tf(t,d) > 0
                0                  otherwise

Normalized TF:
  tf_norm(t,d) = tf(t,d) / max{tf(t',d) : t' ∈ d}

Document Frequency:
  df(t) = |{d ∈ D : t ∈ d}|
  (number of documents containing term t)

Inverse Document Frequency:
  idf(t) = log(N / df(t))
  where N = total number of documents

TF-IDF Weight:
  w(t,d) = tf(t,d) × idf(t)

Common variants:
  1. w = (1 + log(tf)) × log(N/df)
  2. w = tf × log(N/df)
  3. w = tf_norm × log(N/df)
```

### **Why IDF Works:**
```
Term    DF    IDF      Intuition
"the"   1000  log(10000/1000) = 1    Very common → low weight
"obama" 10    log(10000/10) = 3      Rare → high weight
```

### **Manning Book:**
- **Chapter 6** (sections 6.1-6.3):
  - Section 6.1: Parametric and zone indexes
  - Section 6.2: Term frequency and weighting
  - Section 6.3: TF-IDF weighting
  - Section 6.4: Vector space model (next part)

### **Derive by Hand:**
1. Calculate TF for all terms in 3 documents
2. Calculate IDF for all terms across corpus
3. Calculate TF-IDF matrix (terms × documents)
4. Compare raw TF vs log-scaled TF vs TF-IDF

### **Questions to Answer:**
- Why log in IDF? What happens without it?
- Why multiply TF × IDF? Why not add?
- What happens if a term appears in all documents?
- What happens if a term appears only once in corpus?

---

## **Part 4: Vector Space Model** (Day 4)

### **Concepts to Learn:**
1. **Documents as Vectors**
   - Each document is a vector in |V|-dimensional space
   - |V| = vocabulary size
   - Coordinates = TF-IDF weights

2. **Queries as Vectors**
   - Query is also a vector in same space
   - Short vector (few terms)

3. **Similarity Measurement**
   - How to compare two vectors?
   - Why cosine similarity?

4. **Cosine Similarity**
   - Angle between vectors
   - Range: [-1, 1] (or [0, 1] for TF-IDF)
   - Length normalization

### **Math to Master:**
```
Document Vector:
  d = (w₁, w₂, ..., w|V|)
  where w_i = TF-IDF weight of term i in document

Query Vector:
  q = (q₁, q₂, ..., q|V|)
  where q_i = TF-IDF weight of term i in query

Dot Product:
  q · d = Σ(q_i × d_i)

Euclidean Length:
  |q| = sqrt(Σ q_i²)
  |d| = sqrt(Σ d_i²)

Cosine Similarity:
  cos(q, d) = (q · d) / (|q| × |d|)
            = Σ(q_i × d_i) / (sqrt(Σ q_i²) × sqrt(Σ d_i²))

Normalized vectors (unit length):
  cos(q, d) = q · d  (when |q| = |d| = 1)
```

### **Why Cosine Similarity?**
```
Example:
  Document A: "dog dog dog cat"
  Document B: "dog cat"

Both about dogs and cats, but A is longer.

Using dot product: A scores higher (just because longer)
Using cosine: Both score similarly (normalized by length)
```

### **Manning Book:**
- **Chapter 6** (sections 6.3-6.4):
  - Section 6.3: TF-IDF weighting
  - Section 6.4: Vector space model

### **Derive by Hand:**
1. Convert 3 documents to TF-IDF vectors
2. Convert query to vector
3. Calculate cosine similarity for each document
4. Rank documents by similarity
5. Prove: cos(q,d) ∈ [-1, 1]

### **Questions to Answer:**
- Why cosine and not Euclidean distance?
- What does cos(q,d) = 1 mean? cos(q,d) = 0?
- How does document length affect similarity?
- Why normalize vectors to unit length?

---

## **Part 5: Toy Example Implementation** (Day 5)

### **What to Build:**
A complete working implementation from scratch:

```python
# toy_example.py

# Sample Documents (5-10 simple documents)
documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are enemies",
    "The cat and the dog played together",
    "A dog chased a cat"
]

# Step 1: Tokenization
def tokenize(text):
    """Convert text to lowercase tokens"""

# Step 2: Build Inverted Index
def build_inverted_index(documents):
    """
    Returns:
      index = {
        'cat': [0, 2, 3, 4],
        'dog': [1, 2, 3, 4],
        ...
      }
    """

# Step 3: Calculate TF
def calculate_tf(term, document):
    """Raw term frequency"""

# Step 4: Calculate IDF
def calculate_idf(term, documents):
    """Inverse document frequency"""

# Step 5: Calculate TF-IDF
def calculate_tfidf(term, document, documents):
    """TF-IDF weight"""

# Step 6: Build Document Vectors
def build_document_vectors(documents):
    """Convert all documents to TF-IDF vectors"""

# Step 7: Query Processing
def search(query, documents):
    """
    1. Convert query to TF-IDF vector
    2. Calculate cosine similarity with all docs
    3. Return ranked list
    """

# Step 8: Test
query = "cat and dog"
results = search(query, documents)
print(results)
```

### **Expected Output:**
```
Query: "cat and dog"

Rankings:
1. Doc 2: "Cats and dogs are enemies" (score: 0.85)
2. Doc 3: "The cat and the dog played together" (score: 0.78)
3. Doc 4: "A dog chased a cat" (score: 0.72)
4. Doc 0: "The cat sat on the mat" (score: 0.45)
5. Doc 1: "The dog sat on the log" (score: 0.43)
```

### **Implementation Tasks:**
1. **No libraries first** - implement everything from scratch
2. Calculate TF-IDF matrix manually
3. Implement cosine similarity
4. Test on queries
5. **Then** compare with scikit-learn's TfidfVectorizer

### **Things to Observe:**
- How does stopword removal affect results?
- How does stemming affect results?
- What happens with very common terms?
- What happens with query terms not in any document?

---

## **Part 6: Exercises & Practice** (Ongoing)

### **Exercises in `exercises.ipynb`:**

**Exercise 1: Build Inverted Index**
- Given 10 documents, build inverted index by hand
- Verify with code

**Exercise 2: Boolean Queries**
- Process "cat AND dog"
- Process "cat OR dog"
- Process "cat AND NOT dog"

**Exercise 3: TF-IDF Calculation**
- Calculate TF-IDF for specific term in specific document
- Calculate for all terms in all documents

**Exercise 4: Cosine Similarity**
- Given two document vectors, calculate by hand
- Verify with code

**Exercise 5: Ranking**
- Given query and documents, rank by TF-IDF + cosine
- Compare different TF weighting schemes

**Exercise 6: Stemming**
- Apply Porter Stemmer manually
- Compare before/after stemming retrieval results

**Exercise 7: Stopwords**
- Remove stopwords and reindex
- How do results change?

**Exercise 8: Zipf's Law**
- Plot term frequency distribution
- Verify it follows power law

---

## **Class Materials Integration:**

### **Week 1 Slides:**
- Review `class_materials/week_1/slide.pdf`
- Topics: Introduction to IR, Indexing, TF-IDF
- Connect to Manning Ch 1-2, 6

### **Week 2 Lab:**
- Study `class_materials/week_2/Lab_1_Python_Indexing.ipynb`
- Understand their implementation
- Compare with your toy example
- What did they do differently?

---

## **Module 1 Success Criteria:**

By the end of Module 1, you should be able to:

### **Theoretical Understanding:**
- [ ] Explain what an inverted index is and why it's efficient
- [ ] Describe tokenization, stemming, lemmatization
- [ ] State Zipf's Law and Heap's Law
- [ ] Derive TF-IDF formula from first principles
- [ ] Explain why IDF uses logarithm
- [ ] Derive cosine similarity formula
- [ ] Explain why cosine is better than Euclidean distance

### **Mathematical Skills:**
- [ ] Build inverted index by hand for 5 documents
- [ ] Calculate TF-IDF weights by hand
- [ ] Calculate cosine similarity by hand
- [ ] Rank documents by similarity (manual calculation)

### **Implementation Skills:**
- [ ] Implement inverted index from scratch
- [ ] Implement TF-IDF from scratch
- [ ] Implement cosine similarity from scratch
- [ ] Build working search engine on toy data

### **Conceptual Understanding:**
- [ ] When to use stemming vs lemmatization?
- [ ] When to remove stopwords? When not to?
- [ ] Why does TF-IDF work?
- [ ] What are limitations of TF-IDF?

---

# 📖 Module 2: Classical IR Models (Boolean, VSM, BM25)

## **Goal:**
Master classical retrieval models, especially BM25 (your project focus).

## **Duration:** 7-8 days of focused study

---

## **Part 1: Review & Boolean Model** (Day 1)

### **Concepts to Learn:**
1. **What is a Retrieval Model?**
   - Mathematical framework for ranking
   - Assumptions about relevance
   - Basis for algorithms

2. **Boolean Model (Deep Dive)**
   - Exact match semantics
   - Set-based retrieval
   - Boolean algebra operations
   - Extended Boolean models

3. **Limitations of Boolean**
   - No ranking (all results equal)
   - No partial matches
   - Hard to express information needs

### **Math to Master:**
```
Boolean AND:
  q = t₁ AND t₂
  Result = {d : t₁ ∈ d AND t₂ ∈ d}

Boolean OR:
  q = t₁ OR t₂
  Result = {d : t₁ ∈ d OR t₂ ∈ d}

Boolean NOT:
  q = t₁ AND NOT t₂
  Result = {d : t₁ ∈ d AND t₂ ∉ d}

Extended Boolean (fuzzy):
  Allow partial matches with weights
```

### **Manning Book:**
- **Chapter 1** (review, focus on limitations)

### **Questions to Answer:**
- Why can't Boolean model rank documents?
- How to extend Boolean to allow ranking?

---

## **Part 2: Vector Space Model Deep Dive** (Day 2)

### **Concepts to Learn:**
1. **VSM Assumptions**
   - Bag of words (order doesn't matter)
   - Independence assumption
   - Similarity = closeness in vector space

2. **Weighting Schemes**
   - Many variants of TF × IDF
   - Which to choose?

3. **Query Operations**
   - Query expansion
   - Relevance feedback

4. **Limitations of VSM**
   - Vocabulary mismatch
   - Synonymy (different words, same meaning)
   - Polysemy (same word, different meanings)

### **Math to Master:**
```
Different TF Variants:
1. Raw: tf(t,d)
2. Binary: 1 if t ∈ d, else 0
3. Log: 1 + log(tf(t,d))
4. Normalized: tf(t,d) / max_tf(d)

Different IDF Variants:
1. Standard: log(N / df(t))
2. Smoothed: log(N / (1 + df(t)))
3. Probabilistic: log((N - df(t)) / df(t))

Weighting Scheme Notation: xxx.yyy
  xxx = term frequency component
  yyy = document frequency component

Examples:
  - lnc.ltc (SMART notation)
  - tf-idf (common)
```

### **Manning Book:**
- **Chapter 6** (full chapter, detailed study)

### **Derive by Hand:**
1. Compare different TF weighting schemes on same data
2. Compare different IDF weighting schemes
3. Why does log TF work better than raw TF?

---

## **Part 3: Probabilistic IR & Binary Independence Model** (Day 3)

### **Concepts to Learn:**
1. **Probability Ranking Principle (PRP)**
   - Rank by probability of relevance
   - Theoretical foundation

2. **Binary Independence Model (BIM)**
   - Documents as binary vectors
   - Independence assumption
   - Relevance probability

3. **Robertson-Sparck Jones (RSJ) Formula**
   - How to estimate probabilities
   - Without relevance information
   - With relevance feedback

### **Math to Master:**
```
Probability Ranking Principle:
  Rank documents by P(R|d,q)
  where R = relevance

Bayes' Theorem:
  P(R|d,q) = P(d,q|R) × P(R) / P(d,q)

Binary Independence Model:
  - Document d = (x₁, x₂, ..., x_M) where x_i ∈ {0, 1}
  - x_i = 1 if term i in document, 0 otherwise

Relevance Score:
  score(d,q) = Σ log[P(x_i=1|R) × P(x_i=0|NR)] /
                    [P(x_i=0|R) × P(x_i=1|NR)]

  where:
    P(x_i|R) = prob term i appears given relevance
    P(x_i|NR) = prob term i appears given non-relevance

RSJ Formula (simplified):
  score(d,q) = Σ log[(p_i / (1-p_i)) × ((1-q_i) / q_i)]

  where:
    p_i = P(term i | R)
    q_i = P(term i | NR)
```

### **Manning Book:**
- **Chapter 11** (sections 11.1-11.3):
  - Section 11.1: Review of probability
  - Section 11.2: Probability Ranking Principle
  - Section 11.3: Binary Independence Model

### **Derive by Hand:**
1. Derive ranking formula from Bayes' theorem
2. Calculate relevance score for sample documents
3. Compare with TF-IDF ranking

### **Questions to Answer:**
- What does "independence" mean in BIM?
- Why is this assumption often wrong?
- How to estimate probabilities without relevance data?

---

## **Part 4: BM25 - The Most Important Part!** (Days 4-5)

### **Concepts to Learn:**
1. **Evolution: From TF-IDF to BM25**
   - Why TF-IDF isn't perfect
   - What BM25 fixes

2. **BM25 = Best Match 25**
   - "25" because it's the 25th variant in series
   - Most successful classical IR model
   - Foundation for many modern systems

3. **Key Innovations**
   - **Term Frequency Saturation**: Diminishing returns
   - **Document Length Normalization**: Fair comparison
   - **Tunable Parameters**: k1 and b

### **Math to Master - BM25 Formula:**
```
BM25 Score:
  score(D,Q) = Σ IDF(q_i) × [f(q_i,D) × (k₁ + 1)] /
                            [f(q_i,D) + k₁ × (1 - b + b × |D|/avgdl)]

Components:

1. IDF(q_i):
   IDF(q_i) = log[(N - n(q_i) + 0.5) / (n(q_i) + 0.5)]

   where:
     N = total documents
     n(q_i) = documents containing q_i

2. TF Component (the magic part):
   TF_saturated = [f(q_i,D) × (k₁ + 1)] / [f(q_i,D) + k₁]

   This saturates as f → ∞

3. Length Normalization:
   norm = 1 - b + b × |D|/avgdl

   where:
     |D| = length of document D
     avgdl = average document length
     b = controls how much length matters (0 to 1)

Parameters:
  - k₁: Controls term frequency saturation (typical: 1.2 to 2.0)
  - b: Controls length normalization (typical: 0.75)
```

### **Understanding Each Component:**

#### **IDF Component:**
```
Standard TF-IDF: log(N / df)
BM25 IDF: log[(N - df + 0.5) / (df + 0.5)]

Why +0.5? Smoothing to avoid extremes
```

#### **TF Saturation:**
```
TF in Document:  1    2    3    5    10   20   100
Raw TF:          1    2    3    5    10   20   100    (linear)
BM25 (k₁=1.5):  0.6  1.0  1.2  1.5  1.8  1.9  2.0    (saturates!)

Key insight: 100 occurrences is NOT 100× better than 1 occurrence!
```

**Graph to Draw:**
```
  Score
    ^
2.0 |               _______________  (BM25 with k₁=1.5)
    |           ___/
1.5 |       __/
    |    __/
1.0 | __/
    |/
0   +------------------------> Term Frequency
    0    5    10   15   20

versus

  Score
    ^
    |                        /
20  |                      /   (Raw TF - linear)
    |                    /
15  |                  /
    |                /
10  |              /
    |            /
5   |          /
    |        /
    |      /
0   +----/-------------------> Term Frequency
```

#### **Length Normalization:**
```
b = 0:   No length normalization (treat all docs equally)
b = 1:   Full length normalization (penalize long docs heavily)
b = 0.75: Balanced (typical default)

Example:
  Document A: 100 words, contains "Obama" 5 times
  Document B: 1000 words, contains "Obama" 5 times

  Without normalization: Both score equally
  With b=0.75: B is penalized (Obama is proportionally rarer in B)
```

### **Manning Book:**
- **Chapter 11** (section 11.4):
  - Section 11.4.1: Okapi BM25
  - Section 11.4.2: BM25F (multi-field)
  - Section 11.4.3: Page rank combination

### **Derive by Hand (CRITICAL!):**

**Exercise 1: Understand Saturation**
```
Given: k₁ = 1.5
Calculate BM25 TF component for f = 1, 2, 5, 10, 20, 50, 100
Plot the curve
```

**Exercise 2: Length Normalization**
```
Given: avgdl = 500, b = 0.75

Document A: |D| = 250 (short)
Document B: |D| = 500 (average)
Document C: |D| = 1000 (long)

Calculate normalization factor for each
How does it affect final score?
```

**Exercise 3: Full BM25 Calculation**
```
Collection: 5 documents
Query: "information retrieval"
Document D₃: "information retrieval is about finding information"

Step 1: Calculate IDF for "information" and "retrieval"
Step 2: Calculate TF component for each term
Step 3: Apply length normalization
Step 4: Sum up final score
```

### **Compare BM25 vs TF-IDF:**
```
Document: "dog dog dog cat"
Query: "dog"

TF-IDF:
  TF(dog) = 3
  score ∝ 3 × IDF(dog)

BM25 (k₁=1.5):
  TF_component = (3 × 2.5) / (3 + 1.5) = 7.5/4.5 = 1.67
  score ∝ 1.67 × IDF(dog)

Key: BM25 scores LOWER because of saturation
```

### **Questions to Answer (IMPORTANT!):**
1. Why does BM25 saturate TF? What real-world phenomenon does this model?
2. What happens when k₁ → 0? When k₁ → ∞?
3. What happens when b = 0? When b = 1?
4. Why is BM25 better than TF-IDF empirically?
5. When might TF-IDF be better than BM25?

---

## **Part 5: BM25F - Multi-field Extension** (Day 6)

### **Concepts to Learn:**
1. **Structured Documents**
   - Documents have fields (title, body, metadata)
   - Not all fields equally important
   - Title match > body match

2. **BM25F Formula**
   - Extends BM25 to multiple fields
   - Field-specific weights
   - Field-specific length normalization

### **Math to Master:**
```
BM25F Score:
  score(D,Q) = Σ IDF(q_i) × [f̃(q_i,D) × (k₁ + 1)] /
                            [f̃(q_i,D) + k₁]

where f̃(q_i,D) is normalized frequency across fields:

  f̃(q_i,D) = Σ [w_f × f(q_i,D_f)] / [1 - b_f + b_f × |D_f|/avgdl_f]

where:
  f = field (e.g., title, body, anchor)
  w_f = weight of field f
  b_f = length normalization for field f
  |D_f| = length of field f in document D
  avgdl_f = average length of field f

Example Field Weights:
  w_title = 3.0      (title is 3× more important)
  w_body = 1.0       (body is baseline)
  w_metadata = 0.5   (metadata is half as important)
```

### **Manning Book:**
- **Chapter 11** (section 11.4.4): BM25F

### **Derive by Hand:**
```
Document:
  Title: "Information Retrieval"
  Body: "Information retrieval is the process of finding information"

Query: "information"

Calculate BM25F score with:
  w_title = 2.0
  w_body = 1.0
  b_title = 0.5
  b_body = 0.75
```

### **Relevance to Your Project:**
- OpenSanctions has fields: name, alias, description
- Name match >> description match
- BM25F can handle this!

---

## **Part 6: Toy Example Implementation** (Days 7-8)

### **What to Build:**

```python
# toy_example.py

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are enemies",
    "The cat and the dog played together",
    "A dog chased a cat"
]

# Part A: Boolean Retrieval
def boolean_and(term1, term2, index):
    """Intersect two postings lists"""

def boolean_or(term1, term2, index):
    """Union of two postings lists"""

# Part B: VSM with TF-IDF
class TFIDFRetriever:
    def __init__(self):
        self.documents = []
        self.vocabulary = set()
        self.idf = {}

    def build_index(self, docs):
        """Build TF-IDF index"""

    def search(self, query, k=5):
        """Return top-k by cosine similarity"""

# Part C: BM25
class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.avgdl = 0
        self.idf = {}

    def build_index(self, docs):
        """Build BM25 index"""

    def search(self, query, k=5):
        """Return top-k by BM25 score"""

# Part D: Compare All Three
query = "cat and dog"

print("Boolean AND results:")
print(boolean_and("cat", "dog", index))

print("\nTF-IDF VSM results:")
tfidf = TFIDFRetriever()
print(tfidf.search(query))

print("\nBM25 results:")
bm25 = BM25Retriever()
print(bm25.search(query))

# Part E: Parameter Sensitivity
print("\nBM25 with different k1:")
for k1 in [0.5, 1.0, 1.5, 2.0, 3.0]:
    bm25 = BM25Retriever(k1=k1, b=0.75)
    print(f"k1={k1}: {bm25.search(query)}")

print("\nBM25 with different b:")
for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
    bm25 = BM25Retriever(k1=1.5, b=b)
    print(f"b={b}: {bm25.search(query)}")
```

### **Observations to Make:**
1. How do rankings differ between Boolean, TF-IDF, BM25?
2. How does k₁ affect rankings?
3. How does b affect rankings?
4. On which documents does BM25 differ most from TF-IDF?

### **Then Compare with Library:**
```python
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# Compare your implementation with libraries
# Are results identical? If not, why?
```

---

## **Part 7: Class Materials Integration**

### **Week 2 Slides:**
- Review `class_materials/week_2/slide.pdf`
- Boolean, VSM, BIRM concepts
- Relevance definition

### **Week 3 Slides:**
- Review `class_materials/week_3/slide.pdf`
- BM25 deep dive
- Language models (next module)

### **Week 3 Lab:**
- Study `class_materials/week_3/Lab 2 Classifical IR Models.ipynb`
- Understand their implementations
- Compare with your toy examples
- What's similar? What's different?

---

## **Module 2 Success Criteria:**

By the end of Module 2, you should be able to:

### **Theoretical Understanding:**
- [ ] Explain limitations of Boolean model
- [ ] Describe how VSM ranks documents
- [ ] State Probability Ranking Principle
- [ ] **Derive BM25 formula from first principles** ⭐
- [ ] Explain what each component of BM25 does
- [ ] Describe how k₁ and b affect ranking
- [ ] Explain BM25F multi-field extension

### **Mathematical Skills:**
- [ ] Calculate Boolean query results by hand
- [ ] Calculate TF-IDF + cosine by hand
- [ ] **Calculate BM25 score by hand for any document** ⭐
- [ ] Plot TF saturation curve
- [ ] Analyze effect of k₁ and b parameters

### **Implementation Skills:**
- [ ] Implement Boolean retrieval from scratch
- [ ] Implement TF-IDF + VSM from scratch
- [ ] **Implement BM25 from scratch** ⭐
- [ ] Test with different parameters
- [ ] Compare all three models

### **Conceptual Understanding:**
- [ ] When to use Boolean vs ranked retrieval?
- [ ] Why does BM25 outperform TF-IDF?
- [ ] How to tune k₁ and b for specific collection?
- [ ] When might simpler models (TF-IDF) be sufficient?

### **MOST IMPORTANT:**
- [ ] **You can explain BM25 to someone else clearly**
- [ ] **You understand every term in the formula**
- [ ] **You know why BM25 is the standard baseline**

---

# 📖 Module 3: Evaluation Metrics

## **Goal:**
Master IR evaluation - how to measure if your system is good.

## **Duration:** 4-5 days of focused study

**Structured theory write-up (this repo):** [03_evaluation/theory.md](03_evaluation/theory.md) — IIR Chapter 8 outline, pooling, ranx / project qrels, and **recommended metrics by OpenSanctions query type** (aligned with [documents/assignment_1.pdf](../../documents/assignment_1.pdf) §3.4).

---

## **Part 1: Relevance & Test Collections** (Day 1)

### **Concepts to Learn:**
1. **What is Relevance?**
   - Complex, subjective concept
   - Topical relevance vs user relevance
   - Binary vs graded relevance

2. **Test Collections**
   - Corpus of documents
   - Set of queries
   - Relevance judgements (qrels)

3. **Ground Truth Construction**
   - Exhaustive judgement (impossible for large collections)
   - Pooling methodology
   - Inter-annotator agreement

4. **Cranfield Paradigm**
   - Standard evaluation methodology
   - Offline evaluation
   - Assumptions and limitations

### **Math to Master:**
```
Test Collection Components:
1. Document collection D = {d₁, d₂, ..., d_N}
2. Query set Q = {q₁, q₂, ..., q_M}
3. Relevance judgements:
   qrels(q_i, d_j) ∈ {0, 1}  (binary)
   or
   qrels(q_i, d_j) ∈ {0, 1, 2, 3}  (graded)

Pooling Method:
1. Run K different systems on query q
2. Take top-n results from each system
3. Pool = Union of all top-n results (max K×n documents)
4. Judge only documents in pool
5. Documents outside pool assumed non-relevant
```

### **Manning Book:**
- **Chapter 8** (sections 8.1-8.3):
  - Section 8.1: Information retrieval evaluation
  - Section 8.2: Standard test collections
  - Section 8.3: Evaluation of unranked retrieval

### **Questions to Answer:**
- Why is exhaustive judgement infeasible?
- What are limitations of pooling?
- How does subjectivity affect evaluation?

---

## **Part 2: Precision & Recall** (Day 2)

### **Concepts to Learn:**
1. **Confusion Matrix**
   - True Positives (TP)
   - False Positives (FP)
   - False Negatives (FN)
   - True Negatives (TN)

2. **Precision**
   - Fraction of retrieved docs that are relevant
   - Quality measure

3. **Recall**
   - Fraction of relevant docs that are retrieved
   - Completeness measure

4. **Trade-off**
   - High precision vs high recall
   - Can't maximize both simultaneously

### **Math to Master:**
```
Given a query q:
  R = set of relevant documents (from ground truth)
  A = set of retrieved documents (from system)

Confusion Matrix:
  TP = |R ∩ A|  (relevant AND retrieved)
  FP = |A - R|  (retrieved but NOT relevant)
  FN = |R - A|  (relevant but NOT retrieved)
  TN = |D - (R ∪ A)|  (neither relevant nor retrieved)

Precision:
  P = TP / (TP + FP)
    = |R ∩ A| / |A|
    = (relevant retrieved) / (total retrieved)

Recall:
  R = TP / (TP + FN)
    = |R ∩ A| / |R|
    = (relevant retrieved) / (total relevant)

F-measure (harmonic mean):
  F₁ = 2PR / (P + R)
     = 2TP / (2TP + FP + FN)

F_β measure (weighted):
  F_β = (1 + β²) × PR / (β²P + R)

  β < 1: Emphasize precision
  β > 1: Emphasize recall
```

### **Manning Book:**
- **Chapter 8** (section 8.3):
  - Precision and recall definitions
  - F-measure
  - Trade-offs

### **Derive by Hand:**
```
Example:
  10 relevant documents in collection
  System returns 15 documents
  8 of the 15 are relevant

  TP = 8
  FP = 15 - 8 = 7
  FN = 10 - 8 = 2

  Precision = 8/15 = 0.533
  Recall = 8/10 = 0.80
  F₁ = 2 × (0.533 × 0.80) / (0.533 + 0.80) = 0.640
```

### **Questions to Answer:**
- Can precision be 1.0 with low recall? Example?
- Can recall be 1.0 with low precision? Example?
- Why use harmonic mean instead of arithmetic mean?

---

## **Part 3: Ranked Retrieval Evaluation** (Day 3)

### **Concepts to Learn:**
1. **Precision@K**
   - Precision at rank K
   - What fraction of top-K are relevant?

2. **Recall@K**
   - Recall at rank K
   - What fraction of relevant docs in top-K?

3. **Precision-Recall Curve**
   - Trade-off visualization
   - 11-point interpolated precision

4. **Average Precision (AP)**
   - Single-number summary
   - Emphasis on ranking quality

5. **Mean Average Precision (MAP)**
   - Average AP across queries
   - Most common IR metric

### **Math to Master:**
```
Precision at rank K:
  P@K = (# relevant in top K) / K

Recall at rank K:
  R@K = (# relevant in top K) / (total relevant)

Average Precision:
  AP = (1/|R|) × Σ [P(k) × rel(k)]

  where:
    |R| = total relevant documents
    P(k) = precision at rank k
    rel(k) = 1 if document at rank k is relevant, 0 otherwise

  Intuition: Average of precisions at ranks where relevant docs appear

Example:
  Relevant docs: {d₃, d₅, d₉, d₂₅, d₃₉, d₄₄, d₅₆, d₇₁, d₈₉, d₉₂}
  System returns ranked list: [d₁, d₃, d₄, d₅, d₈, d₉, d₁₀, ...]

  Relevant at ranks: 2, 4, 6, ...

  P@2 = 1/2 = 0.50  (d₃ is relevant)
  P@4 = 2/4 = 0.50  (d₃, d₅ are relevant)
  P@6 = 3/6 = 0.50  (d₃, d₅, d₉ are relevant)
  ...

  AP = (1/10) × (0.50 + 0.50 + 0.50 + ...) = X

Mean Average Precision:
  MAP = (1/|Q|) × Σ AP(q_i)

  Average AP across all queries
```

### **Manning Book:**
- **Chapter 8** (sections 8.4-8.5):
  - Section 8.4: Evaluation of ranked retrieval
  - Section 8.5: Assessing relevance

### **Derive by Hand:**
```
Query q₁:
  10 relevant documents
  System ranking: [d₁*, d₂, d₃*, d₄, d₅*, d₆, d₇*, d₈, d₉, d₁₀*]
  (* = relevant)

  Relevant at ranks: 1, 3, 5, 7, 10

  P@1 = 1/1 = 1.00
  P@3 = 2/3 = 0.67
  P@5 = 3/5 = 0.60
  P@7 = 4/7 = 0.57
  P@10 = 5/10 = 0.50

  AP = (1/10) × (1.00 + 0.67 + 0.60 + 0.57 + 0.50) = 0.334

Query q₂:
  Similar calculation...
  AP = 0.512

MAP = (0.334 + 0.512) / 2 = 0.423
```

### **Questions to Answer:**
- Why does AP emphasize early ranks?
- What's a "good" MAP score?
- When is MAP not appropriate?

---

## **Part 4: nDCG (Normalized Discounted Cumulative Gain)** (Day 4)

### **Concepts to Learn:**
1. **Graded Relevance**
   - Not binary (0 or 1)
   - Multiple levels: 0, 1, 2, 3
   - More realistic

2. **Cumulative Gain (CG)**
   - Sum of relevance scores

3. **Discounted Cumulative Gain (DCG)**
   - Logarithmic discount by rank
   - Earlier ranks matter more

4. **Ideal DCG (IDCG)**
   - Best possible ranking

5. **Normalized DCG (nDCG)**
   - Ratio: DCG / IDCG
   - Range: [0, 1]

### **Math to Master:**
```
Graded Relevance:
  rel(d) ∈ {0, 1, 2, 3}
  0 = not relevant
  1 = marginally relevant
  2 = relevant
  3 = highly relevant

Cumulative Gain at position K:
  CG@K = Σ rel(i)  for i=1 to K

  Simply sum relevance scores

Discounted Cumulative Gain at position K:
  DCG@K = Σ [rel(i) / log₂(i + 1)]  for i=1 to K

  or alternative formulation:

  DCG@K = Σ [(2^rel(i) - 1) / log₂(i + 1)]  for i=1 to K

  (Second formula emphasizes highly relevant docs more)

Ideal DCG:
  IDCG@K = DCG@K for perfect ranking
  (Sort documents by relevance, then calculate DCG)

Normalized DCG:
  nDCG@K = DCG@K / IDCG@K

  Range: [0, 1]
  1.0 = perfect ranking
  0.0 = worst possible ranking
```

### **Why Logarithmic Discount?**
```
Position:    1     2     3     4     5     10
Discount:  1.0  0.63  0.50  0.43  0.39  0.30

Users are less likely to look at later results
Logarithmic discount reflects user behavior
```

### **Manning Book:**
- **Chapter 8** (section 8.4.3): nDCG

### **Derive by Hand:**
```
System ranking:     [d₁, d₂, d₃, d₄, d₅]
Relevance scores:   [ 3,  2,  0,  1,  2]  (graded)

DCG@5 (formula 1):
  = 3/log₂(2) + 2/log₂(3) + 0/log₂(4) + 1/log₂(5) + 2/log₂(6)
  = 3/1 + 2/1.58 + 0/2 + 1/2.32 + 2/2.58
  = 3.00 + 1.27 + 0 + 0.43 + 0.78
  = 5.48

Ideal ranking:      [d₁, d₂, d₅, d₄, d₃]
Relevance scores:   [ 3,  2,  2,  1,  0]  (sorted)

IDCG@5:
  = 3/1 + 2/1.58 + 2/2 + 1/2.32 + 0/2.58
  = 3.00 + 1.27 + 1.00 + 0.43 + 0
  = 5.70

nDCG@5 = 5.48 / 5.70 = 0.961
```

### **Questions to Answer:**
- Why use log₂ and not log₁₀?
- What if all documents have same relevance?
- When to use nDCG vs MAP?

---

## **Part 5: Other Metrics** (Day 5)

### **Concepts to Learn:**
1. **Mean Reciprocal Rank (MRR)**
   - For tasks where only one relevant doc
   - Example: Navigational queries

2. **Success@K**
   - Binary: Did we find any relevant doc in top-K?

3. **Time-Based Metrics**
   - How long to find first relevant doc?

4. **User-Centric Metrics**
   - Click-through rate
   - Dwell time
   - User satisfaction

### **Math to Master:**
```
Mean Reciprocal Rank:
  RR(q) = 1 / rank_first_relevant

  MRR = (1/|Q|) × Σ RR(q_i)

Example:
  Query 1: First relevant at rank 2 → RR = 1/2 = 0.50
  Query 2: First relevant at rank 1 → RR = 1/1 = 1.00
  Query 3: First relevant at rank 5 → RR = 1/5 = 0.20

  MRR = (0.50 + 1.00 + 0.20) / 3 = 0.567

Success@K:
  S@K(q) = 1 if any relevant doc in top-K
           0 otherwise

  Success@K = (1/|Q|) × Σ S@K(q_i)
```

### **Manning Book:**
- **Chapter 8** (sections 8.6-8.7):
  - Section 8.6: Results snippets
  - Section 8.7: Alternative evaluation measures

### **Questions to Answer:**
- When is MRR appropriate? When not?
- How do offline metrics relate to user satisfaction?

---

## **Part 6: Toy Example Implementation** (Day 6)

### **What to Build:**

```python
# toy_example.py

# Sample query results
query_results = {
    'q1': ['d1', 'd3', 'd2', 'd5', 'd4', 'd7', 'd6', 'd10', 'd8', 'd9'],
    'q2': ['d2', 'd1', 'd4', 'd3', 'd6', 'd5', 'd8', 'd7', 'd9', 'd10'],
}

# Ground truth (relevance judgements)
qrels = {
    'q1': {'d1': 1, 'd3': 1, 'd5': 1, 'd7': 1, 'd10': 1},  # 5 relevant
    'q2': {'d1': 1, 'd2': 1, 'd4': 1},  # 3 relevant
}

# Graded relevance for nDCG
qrels_graded = {
    'q1': {'d1': 3, 'd3': 2, 'd5': 2, 'd7': 1, 'd10': 1},
    'q2': {'d1': 3, 'd2': 2, 'd4': 1},
}

# Implement all metrics

def precision_at_k(retrieved, relevant, k):
    """Calculate Precision@K"""
    retrieved_k = retrieved[:k]
    relevant_retrieved = set(retrieved_k) & set(relevant.keys())
    return len(relevant_retrieved) / k

def recall_at_k(retrieved, relevant, k):
    """Calculate Recall@K"""
    retrieved_k = retrieved[:k]
    relevant_retrieved = set(retrieved_k) & set(relevant.keys())
    return len(relevant_retrieved) / len(relevant)

def average_precision(retrieved, relevant):
    """Calculate Average Precision"""
    precisions = []
    relevant_count = 0

    for i, doc in enumerate(retrieved):
        if doc in relevant:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))

    if not precisions:
        return 0.0

    return sum(precisions) / len(relevant)

def mean_average_precision(results, qrels):
    """Calculate MAP across all queries"""
    aps = []
    for query_id in results:
        ap = average_precision(results[query_id], qrels[query_id])
        aps.append(ap)
    return sum(aps) / len(aps)

def dcg_at_k(retrieved, relevance, k):
    """Calculate DCG@K"""
    dcg = 0.0
    for i in range(min(k, len(retrieved))):
        doc = retrieved[i]
        rel = relevance.get(doc, 0)
        dcg += rel / math.log2(i + 2)  # i+2 because i starts at 0
    return dcg

def ndcg_at_k(retrieved, relevance, k):
    """Calculate nDCG@K"""
    dcg = dcg_at_k(retrieved, relevance, k)

    # Ideal ranking (sort by relevance)
    ideal = sorted(relevance.keys(), key=lambda x: relevance[x], reverse=True)
    idcg = dcg_at_k(ideal, relevance, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg

def reciprocal_rank(retrieved, relevant):
    """Calculate Reciprocal Rank"""
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0

def mean_reciprocal_rank(results, qrels):
    """Calculate MRR"""
    rrs = []
    for query_id in results:
        rr = reciprocal_rank(results[query_id], qrels[query_id])
        rrs.append(rr)
    return sum(rrs) / len(rrs)

# Calculate all metrics
print("Query 1 metrics:")
print(f"P@5: {precision_at_k(query_results['q1'], qrels['q1'], 5)}")
print(f"R@5: {recall_at_k(query_results['q1'], qrels['q1'], 5)}")
print(f"AP: {average_precision(query_results['q1'], qrels['q1'])}")
print(f"nDCG@5: {ndcg_at_k(query_results['q1'], qrels_graded['q1'], 5)}")

print("\nOverall metrics:")
print(f"MAP: {mean_average_precision(query_results, qrels)}")
print(f"MRR: {mean_reciprocal_rank(query_results, qrels)}")

# Compare with library
from ranx import Qrels, Run, evaluate

# Convert to ranx format
qrels_ranx = Qrels({q: qrels[q] for q in qrels})
run_ranx = Run({q: {doc: 1.0/(i+1) for i, doc in enumerate(query_results[q])}
                for q in query_results})

print("\nUsing ranx library:")
print(evaluate(qrels_ranx, run_ranx, ["map", "mrr", "ndcg@5", "precision@5", "recall@5"]))
```

### **Observations:**
1. Calculate everything by hand first
2. Verify with your code
3. Compare with library (ranx)
4. Understand why metrics differ

---

## **Part 7: Class Materials Integration**

### **Week 3 Slides:**
- Review evaluation sections
- Connect to Manning Ch 8

### **Week 4 Slides:**
- Review `class_materials/week_4/Lecture 4 Evaluation...pptx`
- Modern evaluation techniques

### **Week 4 Lab:**
- Study `class_materials/week_4/Lab3_Evaluation and Interface.ipynb`
- Understand their metric implementations
- Compare with your toy example

---

## **Module 3 Success Criteria:**

By the end of Module 3, you should be able to:

### **Theoretical Understanding:**
- [ ] Explain what relevance means
- [ ] Describe pooling methodology
- [ ] State Precision and Recall definitions
- [ ] Explain Precision-Recall trade-off
- [ ] **Derive MAP formula from first principles** ⭐
- [ ] **Derive nDCG formula from first principles** ⭐
- [ ] Explain when to use which metric

### **Mathematical Skills:**
- [ ] Calculate Precision & Recall by hand
- [ ] Calculate F-measure by hand
- [ ] **Calculate MAP by hand** ⭐
- [ ] **Calculate nDCG by hand** ⭐
- [ ] Calculate MRR by hand
- [ ] Plot Precision-Recall curves

### **Implementation Skills:**
- [ ] Implement all metrics from scratch
- [ ] Use evaluation libraries (ranx)
- [ ] Generate evaluation reports
- [ ] Visualize results

### **Conceptual Understanding:**
- [ ] Why does MAP emphasize ranking?
- [ ] When is binary relevance sufficient?
- [ ] When do you need graded relevance?
- [ ] How to interpret metric values?
- [ ] What makes a "good" IR system?

### **Critical Thinking:**
- [ ] Which metric for navigational queries?
- [ ] Which metric for recall-oriented tasks?
- [ ] How do metrics relate to user satisfaction?
- [ ] What are limitations of offline evaluation?

---

# 📖 Module 4: Dense Retrieval (Optional/Lightweight)

## **Goal:**
Understand modern dense retrieval approaches (to help Marek).

## **Duration:** 2-3 days (optional, lightweight)

---

## **Part 1: From Sparse to Dense** (Day 1)

### **Concepts to Learn:**
1. **Sparse Representations (Review)**
   - BM25, TF-IDF = sparse vectors
   - Most dimensions are zero
   - Exact keyword matching

2. **Dense Representations**
   - Neural embeddings
   - Every dimension non-zero
   - Semantic matching

3. **Why Dense?**
   - Vocabulary mismatch problem
   - Synonymy: "car" vs "automobile"
   - Paraphrasing
   - Conceptual similarity

### **Math to Master:**
```
Sparse Vector (BM25):
  d = [0, 0, 3.2, 0, 0, 1.5, 0, 0, 0, ...]  # |V| dimensions, mostly zeros

Dense Vector (Embedding):
  d = [0.12, -0.45, 0.33, 0.78, -0.22, ...]  # Fixed dimensions (e.g., 384)

Comparison:
  Sparse: |V| dimensions (10,000 - 1,000,000)
  Dense: Fixed dimensions (128 - 1024)

  Sparse: Exact keyword overlap
  Dense: Semantic similarity
```

### **No specific Manning chapter** (too modern)
- Search online resources: "dense passage retrieval"
- Papers: "Dense Passage Retrieval for Open-Domain QA" (Karpukhin et al.)

---

## **Part 2: Sentence Embeddings** (Day 2)

### **Concepts to Learn:**
1. **Word Embeddings**
   - Word2Vec, GloVe
   - Similar words → similar vectors

2. **Sentence Embeddings**
   - Sentence-BERT (SBERT)
   - all-MiniLM-L6-v2 (your project uses this)
   - 384-dimensional vectors

3. **Cosine Similarity (Review)**
   - Same as before, but in embedding space
   - Range: [-1, 1]

### **Math to Master:**
```
Sentence Embedding:
  encode("The cat sat on the mat") → [0.12, -0.45, ..., 0.33]  # 384 dims

Semantic Similarity:
  sim("cat", "feline") > sim("cat", "computer")

  In embedding space:
    cos(embed("cat"), embed("feline")) ≈ 0.85
    cos(embed("cat"), embed("computer")) ≈ 0.12
```

---

## **Part 3: Hybrid Retrieval & RRF** (Day 3)

### **Concepts to Learn:**
1. **Combining Sparse + Dense**
   - BM25 for exact matches
   - Embeddings for semantic matches
   - Best of both worlds

2. **Reciprocal Rank Fusion (RRF)**
   - Combine ranked lists
   - No score normalization needed
   - Simple and effective

### **Math to Master:**
```
Reciprocal Rank Fusion:
  RRF(d) = Σ [1 / (k + rank_i(d))]

  where:
    rank_i(d) = rank of document d in system i
    k = constant (typically 60)

Example:
  BM25 ranking:  [d₁, d₂, d₃, d₄, d₅]
  Dense ranking: [d₃, d₁, d₅, d₆, d₂]

  For d₁:
    RRF(d₁) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0327

  For d₃:
    RRF(d₃) = 1/(60+3) + 1/(60+1) = 1/63 + 1/61 = 0.0323

  Final ranking based on RRF scores
```

---

## **Part 4: Lightweight Toy Example**

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode documents
documents = [
    "The cat sat on the mat",
    "The dog played in the park",
    "A feline rested on the carpet"  # Synonym of first doc!
]

doc_embeddings = model.encode(documents)

# Encode query
query = "A cat on a rug"
query_embedding = model.encode(query)

# Calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# Compare with BM25
print("Dense retrieval:")
for i, sim in enumerate(similarities):
    print(f"Doc {i}: {sim:.3f}")

# Observe: Doc 0 and Doc 2 both score high (semantic similarity!)
# BM25 would miss Doc 2 (no keyword overlap)
```

---

## **Module 4 Success Criteria:**

**Lightweight goals** (this is Marek's main task):

- [ ] Understand difference between sparse and dense
- [ ] Know what sentence embeddings are
- [ ] Understand RRF formula
- [ ] Can explain when dense helps
- [ ] Can assist Marek with integration

**NOT required:**
- Deep understanding of transformers
- Training embeddings
- Advanced neural IR

---

# ✅ Overall Success: Modules Complete

After completing all modules, you should:

1. **Deeply understand** classical IR (TF-IDF, BM25)
2. **Be able to derive** all formulas from scratch
3. **Implement** retrieval systems from first principles
4. **Evaluate** systems properly with multiple metrics
5. **Understand** modern approaches (dense retrieval)

**Then you're ready to build your project! 🚀**

---

# 📝 Study Journal Template

For each day of learning, keep a journal:

```markdown
## Date: YYYY-MM-DD
**Module**: Module 2, Part 4 (BM25)
**Time spent**: 4 hours

### What I studied:
- BM25 formula derivation
- Saturation curve analysis
- Parameter sensitivity

### Math I derived:
- ✅ BM25 formula from scratch
- ✅ TF saturation component
- ✅ Length normalization component

### Code I wrote:
- ✅ BM25 from scratch (toy_example.py)
- ✅ Tested with k1 = [1.0, 1.5, 2.0]
- ✅ Tested with b = [0.5, 0.75, 1.0]

### Key insights:
- k1 controls how quickly TF saturates
- b=0.75 is a good default for most collections
- BM25 handles document length better than TF-IDF

### Questions remaining:
- How to tune k1 for specific collection?
- When would k1=2.0 be better than k1=1.5?

### Tomorrow:
- Study BM25F (multi-field)
- Start project implementation
```

---

**This is your pure learning roadmap. No project mixing. Deep understanding first, then apply!** 🎓
