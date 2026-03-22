# Module 1: Indexing & TF-IDF - Theory

## Overview
**Goal**: Understand how to build an inverted index and calculate TF-IDF weights.

**Duration**: 4-5 days of focused study

**Manning Book**: Chapters 1, 2, 6

**Key Concepts**:
- Inverted index structure
- Tokenization, normalization, stemming, lemmatization
- TF-IDF weighting scheme
- Vector Space Model
- Cosine similarity

---

# Part 1: Boolean Retrieval & Inverted Index

## What is Information Retrieval?

### The Information Hierarchy
```
Data → Information → Knowledge

Data: Raw, unstructured facts
  Example: "cat", "dog", "mat", "sat"

Information: Organized, contextualized data
  Example: "The cat sat on the mat"

Knowledge: Processed information that enables action
  Example: Understanding that cats typically sit on mats
```

### IR vs Database Systems

| Aspect | Database Systems | Information Retrieval |
|--------|------------------|----------------------|
| **Data** | Structured (tables, schemas) | Unstructured (text, documents) |
| **Queries** | SQL (exact match) | Natural language (partial match) |
| **Results** | Exact matches | Ranked by relevance |
| **Example** | `SELECT * WHERE id=5` | "Find documents about cats" |

---

## The Boolean Model

### Definition
The **Boolean Model** is the simplest retrieval model based on set theory and Boolean algebra.

- Documents represented as sets of terms
- Queries expressed as Boolean expressions
- Results: exact matches (no ranking)

### Boolean Operations

#### AND Operation
```
Query: "Brutus AND Caesar"
Result: Documents containing BOTH "Brutus" AND "Caesar"

Set notation: D_result = D_Brutus ∩ D_Caesar
```

#### OR Operation
```
Query: "Brutus OR Caesar"
Result: Documents containing "Brutus" OR "Caesar" OR both

Set notation: D_result = D_Brutus ∪ D_Caesar
```

#### NOT Operation
```
Query: "Brutus AND NOT Caesar"
Result: Documents containing "Brutus" but NOT "Caesar"

Set notation: D_result = D_Brutus - D_Caesar
```

### Example

**Documents:**
```
Doc 1: "Brutus killed Caesar"
Doc 2: "Caesar was a Roman emperor"
Doc 3: "Brutus was Caesar's friend"
Doc 4: "The Roman empire fell"
```

**Queries:**
- `Brutus AND Caesar` → {Doc 1, Doc 3}
- `Brutus OR Caesar` → {Doc 1, Doc 2, Doc 3}
- `Caesar AND NOT Brutus` → {Doc 2}
- `Roman AND (Brutus OR Caesar)` → {Doc 2, Doc 3}

---

## Inverted Index Structure

### The Problem
**Naive approach**: For each query, scan all documents
- Time complexity: O(N×M) where N=docs, M=avg doc length
- Infeasible for large collections (millions of documents)

**Solution**: Pre-build an index!

### Term-Document Incidence Matrix

|        | Doc1 | Doc2 | Doc3 | Doc4 |
|--------|------|------|------|------|
| Brutus |  1   |  0   |  1   |  0   |
| Caesar |  1   |  1   |  1   |  0   |
| killed |  1   |  0   |  0   |  0   |
| Roman  |  0   |  1   |  0   |  1   |
| empire |  0   |  1   |  0   |  1   |

- **Rows** = terms
- **Columns** = documents
- **Cell** = 1 if term in document, 0 otherwise

**Problem with matrix**: Too sparse! (Mostly zeros)

### Inverted Index (Better Solution)

Instead of matrix, use:

```
Dictionary (terms) → Postings Lists (documents)

Brutus  → [1, 3]
Caesar  → [1, 2, 3]
empire  → [2, 4]
killed  → [1]
Roman   → [2, 4]
```

**Components:**

1. **Dictionary**:
   - All unique terms
   - Sorted alphabetically
   - Points to postings list

2. **Postings List**:
   - Document IDs containing the term
   - Sorted by document ID
   - Enables efficient intersection/union

### Why "Inverted"?

**Forward Index**: Document → Terms
```
Doc 1 → [Brutus, killed, Caesar]
Doc 2 → [Caesar, Roman, emperor]
```

**Inverted Index**: Term → Documents
```
Brutus → [Doc 1, Doc 3]
Caesar → [Doc 1, Doc 2, Doc 3]
```

"Inverted" because we flip the mapping!

---

## Boolean Query Processing

### Algorithm: AND Query

```python
def intersect(p1, p2):
    """
    Intersect two postings lists
    Both lists must be sorted by document ID
    """
    result = []
    i, j = 0, 0

    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            result.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1

    return result
```

**Time Complexity**: O(x + y) where x, y are lengths of postings lists

**Much better than O(N×M) naive scan!**

### Example: Process "Brutus AND Caesar"

```
Step 1: Look up "Brutus" in dictionary → postings [1, 3]
Step 2: Look up "Caesar" in dictionary → postings [1, 2, 3]
Step 3: Intersect [1, 3] and [1, 2, 3]

Intersection process:
  Compare 1 and 1 → match! Add 1 to result
  Compare 3 and 2 → 2 < 3, advance pointer in Caesar list
  Compare 3 and 3 → match! Add 3 to result

Result: [1, 3]
```

---

## Limitations of Boolean Model

1. **No Ranking**: All results treated equally
   - "Brutus" appears 100 times → same as appears once

2. **All or Nothing**: Either matches or doesn't
   - Query "Brutus AND Caesar AND empire" might return nothing
   - But "Brutus AND Caesar" without "empire" could be useful

3. **Hard to Express Information Needs**
   - How to say "documents mostly about Brutus"?
   - How to express "prefer Caesar but not required"?

4. **No Notion of Relevance**
   - Which result is better?
   - How to order results?

**Solution**: We need ranked retrieval! (Coming in TF-IDF and BM25)

---

# Part 2: Term Vocabulary & Text Processing

## Tokenization

### What is a Token?

**Token**: An instance of a sequence of characters that are grouped together as a useful semantic unit for processing.

**Type**: The class of all tokens containing the same character sequence.

```
Text: "The cat sat on the mat. The cat likes the mat."

Tokens: ["The", "cat", "sat", "on", "the", "mat", "The", "cat", "likes", "the", "mat"]
         (11 tokens)

Types: {"The", "cat", "sat", "on", "the", "mat", "likes"}
       (7 types - unique tokens)
```

### Tokenization Rules

**English**: Relatively easy
```
Input: "San Francisco is a city."
Tokens: ["San", "Francisco", "is", "a", "city"]

But what about:
  "Ph.D." → ["Ph.D."] or ["Ph", "D"]?
  "don't" → ["don't"] or ["do", "n't"] or ["don", "t"]?
  "New York" → ["New", "York"] or ["New_York"]?
```

**Other Languages**: Much harder!
```
Chinese: "我爱北京天安门" (no spaces!)
German: "Lebensversicherungsgesellschaftsangestellter" (compound words!)
```

### Common Tokenization Issues

1. **Hyphens**: "state-of-the-art", "co-education"
2. **Apostrophes**: "it's", "don't", "O'Neill"
3. **Periods**: "Mr.", "Ph.D.", "U.S.A."
4. **Numbers**: "555-1234", "$100.50", "3.14159"
5. **Email/URLs**: "user@example.com", "http://example.com"

**Rule of thumb**: Keep it simple, consistent, and language-aware

---

## Normalization

### Case Folding

**Goal**: Reduce all letters to lowercase (usually)

```
"Apple" → "apple"
"APPLE" → "apple"
"aPpLe" → "apple"
```

**Benefits**:
- "Apple" and "apple" treated as same term
- Reduces vocabulary size

**Considerations**:
- Proper nouns: "Bush" (president) vs "bush" (plant)
- Acronyms: "US" vs "us"
- Sometimes case matters!

### Accents and Diacritics

```
"café" → "cafe"?
"naïve" → "naive"?
"Zürich" → "Zurich"?
```

**Trade-off**:
- Remove → More matches (recall), but less precision
- Keep → More precise, but might miss variants

**Solution**: Depends on language and user base!

---

## Stemming

### What is Stemming?

**Stemming**: Crude heuristic process that chops off word endings to get root form

**Goal**: Reduce inflectional/derivational variants to base form

```
"running" → "run"
"runs" → "run"
"runner" → "run"

"automate" → "autom"
"automatic" → "autom"
"automation" → "autom"
```

### Porter Stemmer Algorithm

**Most famous stemmer for English** (1980, still widely used!)

**5-step algorithm** with rewrite rules:

#### Step 1: Plurals and -ED/-ING

```
Rules:
  SSES → SS      : "caresses" → "caress"
  IES  → I       : "ponies" → "poni"
  SS   → SS      : "caress" → "caress"
  S    → ∅       : "cats" → "cat"

  (m > 0) EED → EE  : "agreed" → "agree"
  (*v*) ED → ∅      : "played" → "play"
  (*v*) ING → ∅     : "playing" → "play"

where:
  m = measure (number of VC sequences)
  *v* = contains vowel
```

#### Example Walkthrough

```
Word: "computational"

Step 1: No change (not plural, not -ED/-ING)
Step 2: ATIONAL → ATE : "computational" → "computate"
Step 3: No applicable rule
Step 4: No applicable rule
Step 5: Remove final E : "computate" → "computat"

Result: "computational" → "computat"
```

### Porter Stemmer Errors

**Over-stemming** (false positives):
```
"universe" → "univers"
"university" → "univers"
  → These are different concepts!

"policy" → "polici"
"police" → "polic"
  → Unrelated words grouped together
```

**Under-stemming** (false negatives):
```
"alumnus" → "alumnu"
"alumni" → "alumni"
  → Should be same, but different stems
```

**Key Insight**: Stemming is heuristic and imperfect!

---

## Lemmatization

### What is Lemmatization?

**Lemmatization**: Proper linguistic analysis to find the lemma (dictionary form)

**Lemma**: Canonical/dictionary form of a word

```
"am", "are", "is" → "be"
"better" → "good"
"running" → "run"
"mice" → "mouse"
```

### Stemming vs Lemmatization

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| **Method** | Crude chopping | Linguistic analysis |
| **Output** | May not be real word | Always real word |
| **Example** | "running" → "run" | "running" → "run" |
| **Example** | "better" → "better" | "better" → "good" |
| **Example** | "saw" → "saw" | "saw" → "see" (verb) or "saw" (noun) |
| **Speed** | Fast | Slower (needs POS tagging) |
| **Accuracy** | Lower | Higher |
| **Tools** | Porter, Snowball | spaCy, NLTK WordNet |

### When to Use Which?

**Use Stemming when**:
- Speed matters
- English language
- Simple retrieval
- Don't need linguistic accuracy

**Use Lemmatization when**:
- Accuracy matters
- Multiple languages
- NLP tasks beyond IR
- Can afford computational cost

---

## Stopwords

### What are Stopwords?

**Stopwords**: Very common words with little semantic content

**Examples**: the, a, an, is, are, was, were, in, on, at, to, ...

### Why Remove Stopwords?

1. **Reduce Index Size**:
   ```
   "the" appears in 90% of documents
   → Huge postings list
   → Takes memory and disk space
   ```

2. **Improve Efficiency**:
   ```
   Query: "the cat and the dog"
   Without stopword removal: Look up 5 terms
   With stopword removal: Look up 2 terms ("cat", "dog")
   ```

3. **Reduce Noise**:
   - Stopwords don't help distinguish documents
   - "the" in all documents → zero discriminative power

### Stopword List

**Common English stopwords** (top 25):
```
a, an, and, are, as, at, be, by, for, from,
has, he, in, is, it, its, of, on, that, the,
to, was, will, with
```

**Full lists**: 100-500 words

### When NOT to Remove Stopwords?

1. **Phrase Queries**:
   ```
   "to be or not to be"
   "king of the hill"
   "flights to London"
   ```
   Removing stopwords breaks meaning!

2. **Proper Nouns**:
   ```
   "The Who" (band name)
   "Take That" (band name)
   ```

3. **Modern Trend**: Keep stopwords!
   - Disk space is cheap
   - Removal can hurt precision
   - Better to weight them low (coming in TF-IDF)

---

## Statistical Properties of Text

### Zipf's Law

**Observation**: Term frequency is inversely proportional to rank

**Formula**:
```
cf_i ∝ 1/i

where:
  cf_i = collection frequency of i-th most common term
  i = rank (1, 2, 3, ...)
```

**More precisely**:
```
cf_i = c / i^α

where:
  c = constant
  α ≈ 1 (exponent, typically close to 1)
```

### Zipf's Law: Example

| Rank | Term | Frequency | 1/Rank |
|------|------|-----------|--------|
| 1 | the | 50,000 | 1.00 |
| 2 | of | 25,000 | 0.50 |
| 3 | and | 16,667 | 0.33 |
| 10 | in | 5,000 | 0.10 |
| 100 | about | 500 | 0.01 |

**Plot**: log(cf) vs log(rank) gives straight line with slope ≈ -1

**Implications**:
- Few very common terms (the, of, and, ...)
- Many rare terms (long tail)
- Power law distribution

---

### Heap's Law

**Observation**: Vocabulary grows sublinearly with collection size

**Formula**:
```
V = K × n^β

where:
  V = vocabulary size (number of unique terms)
  n = collection size (number of tokens)
  K, β = constants
  β typically between 0.4 and 0.6 (< 1)
```

**Example**:
```
If β = 0.5 and K = 10:
  n = 100 tokens → V = 10 × 100^0.5 = 100 types
  n = 10,000 tokens → V = 10 × 10,000^0.5 = 1,000 types
  n = 1,000,000 tokens → V = 10 × 1,000,000^0.5 = 10,000 types
```

**Implication**: Vocabulary keeps growing (never saturates)
- Always see new words
- Index size prediction
- Important for system design

---

# Part 3: TF-IDF Weighting

## Beyond Boolean: Need for Ranking

**Problem with Boolean Model**:
```
Query: "cat"
Boolean result: {Doc1, Doc5, Doc15, Doc23, ...}

But which document is BEST?
- Doc1 mentions "cat" once
- Doc5 mentions "cat" 50 times
  → Should Doc5 rank higher?
```

**Solution**: Assign weights to terms!

---

## Term Frequency (TF)

### Raw Term Frequency

**Definition**: Count of term in document

```
tf(t, d) = number of times term t appears in document d
```

**Example**:
```
Doc: "The cat sat on the mat. The cat likes the mat."

tf("cat", doc) = 2
tf("the", doc) = 3
tf("mat", doc) = 2
tf("likes", doc) = 1
```

### Problem with Raw TF

**Issue**: Linear growth doesn't match relevance

```
Document A: "cat" appears 1 time
Document B: "cat" appears 100 times

Is B really 100× more relevant?
Probably not! (Diminishing returns)
```

### Log-scaled TF

**Better approach**: Use logarithm

```
tf_log(t, d) = {
  1 + log(tf(t,d))   if tf(t,d) > 0
  0                   if tf(t,d) = 0
}
```

**Why logarithm?**
- Sub-linear growth
- 1 occurrence → score 1.0
- 10 occurrences → score 2.0
- 100 occurrences → score 3.0

**Example**:
```
tf     tf_log
1   →  1.0
2   →  1.3
5   →  1.7
10  →  2.0
100 →  3.0
```

### Normalized TF

**Alternative**: Normalize by document length

```
tf_norm(t, d) = tf(t, d) / max{tf(t', d) : t' ∈ d}
```

**Range**: [0, 1]

**Example**:
```
Doc: "cat cat cat dog dog bird"

Raw frequencies:
  cat: 3
  dog: 2
  bird: 1

Max frequency = 3

Normalized:
  tf_norm(cat) = 3/3 = 1.0
  tf_norm(dog) = 2/3 = 0.67
  tf_norm(bird) = 1/3 = 0.33
```

---

## Document Frequency (DF) & Inverse Document Frequency (IDF)

### The Problem: All Terms Not Equal

```
Query: "the cat"

Problem:
  "the" appears in 90% of documents → not discriminative
  "cat" appears in 5% of documents → discriminative!

We should weight "cat" higher than "the"
```

### Document Frequency (DF)

**Definition**: Number of documents containing term

```
df(t) = |{d ∈ D : t ∈ d}|
```

**Example**:
```
Collection: 10,000 documents

df("the") = 9,000 (very common)
df("cat") = 500 (less common)
df("pneumonoultramicroscopicsilicovolcanoconiosis") = 1 (very rare)
```

### Inverse Document Frequency (IDF)

**Intuition**: Rare terms are more informative!

**Formula**:
```
idf(t) = log(N / df(t))

where:
  N = total number of documents
  df(t) = document frequency of term t
```

**Why logarithm?**: Dampen the effect (common pattern in IR)

**Example** (N = 10,000):
```
Term     df      N/df    idf = log(N/df)
"the"    9,000   1.11    0.05  (low weight - common term)
"cat"    500     20      1.30  (medium weight)
"obama"  10      1,000   3.00  (high weight - rare term)
```

### IDF Insights

**Limits**:
```
df = N (term in all docs)    → idf = log(1) = 0   (no discriminative power)
df = 1 (term in one doc)     → idf = log(N) ≈ 10 (maximum discriminative power)
df = N/2 (term in half docs) → idf = log(2) ≈ 0.3 (some discriminative power)
```

**Key Property**: IDF is global (same for all documents in collection)

---

## TF-IDF Weight

### Combining TF and IDF

**Idea**: Weight = how important term is to document × how rare term is in collection

**Formula**:
```
w(t, d) = tf(t, d) × idf(t)
```

**With log-scaling**:
```
w(t, d) = (1 + log(tf(t,d))) × log(N / df(t))   if tf(t,d) > 0
        = 0                                        if tf(t,d) = 0
```

### TF-IDF Example

**Collection**: 10,000 documents

**Document**: "The cat sat on the mat. The cat likes the mat."

**Term frequencies**:
- "cat": 2
- "the": 3
- "mat": 2
- "sat": 1
- "likes": 1

**Document frequencies** (assume):
- "cat": df = 500
- "the": df = 9,000
- "mat": df = 1,000
- "sat": df = 2,000
- "likes": df = 1,500

**Calculate TF-IDF**:
```
Term   | tf  | log(1+tf) | df    | idf=log(N/df) | TF-IDF
-------|-----|-----------|-------|---------------|--------
cat    | 2   | 1.30      | 500   | 1.30          | 1.69
the    | 3   | 1.48      | 9,000 | 0.05          | 0.07
mat    | 2   | 1.30      | 1,000 | 1.00          | 1.30
sat    | 1   | 1.00      | 2,000 | 0.70          | 0.70
likes  | 1   | 1.00      | 1,500 | 0.82          | 0.82
```

**Observations**:
- "cat" has highest weight (frequent in doc, rare in collection)
- "the" has lowest weight (frequent in doc, but very common in collection)
- TF-IDF balances local (TF) and global (IDF) importance

---

## SMART Notation

### Weighting Scheme Notation

**Format**: `xxx.yyy`
- `xxx` = term frequency component (document)
- `yyy` = document frequency component (collection)

### Common Schemes

**Term Frequency (xxx)**:
- `n` (natural): tf
- `l` (log): 1 + log(tf)
- `a` (augmented): 0.5 + 0.5 × tf/max(tf)
- `b` (binary): 1 if tf > 0, else 0

**Document Frequency (yyy)**:
- `n` (no): 1 (no IDF)
- `t` (IDF): log(N/df)
- `p` (prob IDF): max(0, log((N-df)/df))

**Examples**:
- `ltc`: log TF × IDF × cosine normalization
- `lnc`: log TF × no IDF × cosine normalization
- `atn`: augmented TF × IDF × no normalization

**Most common**: `ltc` for documents, `lnc` for queries

---

# Part 4: Vector Space Model

## Documents as Vectors

### The Vector Space

**Idea**: Represent documents and queries as vectors in high-dimensional space

**Dimensions**: |V| (vocabulary size)

**Example**:
```
Vocabulary: {cat, dog, mat, sat, on, the}
|V| = 6 dimensions

Document: "The cat sat on the mat"

Vector representation:
  cat: 1
  dog: 0
  mat: 1
  sat: 1
  on: 1
  the: 2

d = [1, 0, 1, 1, 1, 2]
```

**With TF-IDF weights**:
```
d = [1.69, 0, 1.30, 0.70, 0.50, 0.07]
    (TF-IDF weights for each term)
```

### Vector Space Visualization

**2D example** (only 2 terms):
```
       ^
    2  |      d2
       |     /
TF-IDF |    /
"cat"  |   /
    1  |  / d1
       | /
       +----------->
       0  1  2
       TF-IDF "dog"
```

**Reality**: thousands or millions of dimensions!

---

## Queries as Vectors

**Same representation as documents**:
```
Query: "cat and dog"

After stopword removal: "cat dog"

Query vector:
q = [tf-idf(cat,q), tf-idf(dog,q), 0, 0, ..., 0]
```

**Key insight**: Query and documents live in same vector space!

---

## Measuring Similarity

### The Problem

**How to compare query and document vectors?**

**Option 1: Euclidean Distance**
```
dist(q, d) = √(Σ(q_i - d_i)²)
```

**Problem**: Length-dependent!
```
Document A: "cat dog" (short)
Document B: "cat dog cat dog cat dog" (long, same topic)

B has larger magnitude → farther from query!
But B is about same topic as A
```

### Cosine Similarity (The Solution!)

**Idea**: Measure angle between vectors (not distance)

**Formula**:
```
cos(θ) = cos(q, d) = (q · d) / (|q| × |d|)
```

**Components**:

1. **Dot Product**:
   ```
   q · d = Σ(q_i × d_i)
         = q_1×d_1 + q_2×d_2 + ... + q_n×d_n
   ```

2. **Euclidean Length (L2 norm)**:
   ```
   |q| = √(Σ q_i²) = √(q_1² + q_2² + ... + q_n²)
   |d| = √(Σ d_i²) = √(d_1² + d_2² + ... + d_n²)
   ```

3. **Cosine Similarity**:
   ```
   cos(q, d) = (q · d) / (|q| × |d|)
   ```

### Cosine Similarity: Geometric Intuition

```
cos(θ) = 1   → θ = 0°    → Same direction (identical)
cos(θ) = 0   → θ = 90°   → Orthogonal (no overlap)
cos(θ) = -1  → θ = 180°  → Opposite direction
```

**For TF-IDF vectors** (all positive weights):
- Range: [0, 1]
- 1 = perfectly similar
- 0 = no terms in common

---

## Cosine Similarity: Complete Example

### Setup

**Documents**:
```
d1: "The cat sat on the mat"
d2: "The dog sat on the log"
d3: "Cats and dogs"
```

**Query**:
```
q: "cat dog"
```

**Vocabulary** (after stopword removal):
```
V = {cat, cats, dog, dogs, log, mat, sat}
```

### Step 1: Calculate TF-IDF Vectors

**Assume** (simplified):
```
Term | TF(d1) | TF(d2) | TF(d3) | IDF  | TF-IDF(d1) | TF-IDF(d2) | TF-IDF(d3)
-----|--------|--------|--------|------|------------|------------|------------
cat  | 1      | 0      | 0      | 1.0  | 1.0        | 0.0        | 0.0
cats | 0      | 0      | 1      | 1.5  | 0.0        | 0.0        | 1.5
dog  | 0      | 1      | 0      | 1.2  | 0.0        | 1.2        | 0.0
dogs | 0      | 0      | 1      | 1.5  | 0.0        | 0.0        | 1.5
log  | 0      | 1      | 0      | 2.0  | 0.0        | 2.0        | 0.0
mat  | 1      | 0      | 0      | 2.0  | 2.0        | 0.0        | 0.0
sat  | 1      | 1      | 0      | 1.5  | 1.5        | 1.5        | 0.0
```

**Vectors**:
```
d1 = [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5]
d2 = [0.0, 0.0, 1.2, 0.0, 2.0, 0.0, 1.5]
d3 = [0.0, 1.5, 0.0, 1.5, 0.0, 0.0, 0.0]

q  = [1.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0]  (just cat and dog)
```

### Step 2: Calculate Dot Products

```
q · d1 = 1.0×1.0 + 0×0 + 0×0 + 0×0 + 0×0 + 0×2.0 + 0×1.5
       = 1.0

q · d2 = 1.0×0 + 0×0 + 1.2×1.2 + 0×0 + 0×2.0 + 0×0 + 0×1.5
       = 1.44

q · d3 = 1.0×0 + 0×1.5 + 1.2×0 + 0×1.5 + 0×0 + 0×0 + 0×0
       = 0
```

### Step 3: Calculate Lengths

```
|q| = √(1.0² + 0² + 1.2² + 0² + 0² + 0² + 0²)
    = √(1.0 + 1.44)
    = √2.44
    = 1.56

|d1| = √(1.0² + 0² + 0² + 0² + 0² + 2.0² + 1.5²)
     = √(1.0 + 4.0 + 2.25)
     = √7.25
     = 2.69

|d2| = √(0² + 0² + 1.2² + 0² + 2.0² + 0² + 1.5²)
     = √(1.44 + 4.0 + 2.25)
     = √7.69
     = 2.77

|d3| = √(0² + 1.5² + 0² + 1.5² + 0² + 0² + 0²)
     = √(2.25 + 2.25)
     = √4.5
     = 2.12
```

### Step 4: Calculate Cosine Similarities

```
cos(q, d1) = 1.0 / (1.56 × 2.69) = 1.0 / 4.20 = 0.238

cos(q, d2) = 1.44 / (1.56 × 2.77) = 1.44 / 4.32 = 0.333

cos(q, d3) = 0 / (1.56 × 2.12) = 0 / 3.31 = 0
```

### Step 5: Rank Documents

```
Ranking:
1. d2 (score: 0.333) - "The dog sat on the log"
2. d1 (score: 0.238) - "The cat sat on the mat"
3. d3 (score: 0) - "Cats and dogs"
```

**Why this ranking?**
- d2 has "dog" which query also has
- d1 has "cat" which query also has
- d3 has neither "cat" nor "dog" (different word forms: "cats", "dogs")

---

## Length Normalization

### Why Normalize?

**Problem**: Longer documents have larger magnitude

```
Document A: "cat" (short)
  |A| = small

Document B: "cat cat cat cat cat" (long, repetitive)
  |B| = large

Without normalization: B scores higher just because it's longer!
```

**Solution**: Normalize vectors to unit length

### Unit Vectors

**Definition**: Vector with length 1

**Formula**:
```
d_normalized = d / |d|
```

**After normalization**:
```
cos(q, d) = q_norm · d_norm
```

**All documents on unit sphere** → fair comparison!

---

## VSM: Putting It All Together

### Complete VSM Retrieval Algorithm

```python
def vsm_search(query, documents):
    """
    Vector Space Model search with TF-IDF and cosine similarity
    """
    # 1. Preprocess query and documents
    query_terms = preprocess(query)

    # 2. Calculate TF-IDF for query
    query_vector = calculate_tfidf_vector(query_terms)

    # 3. Normalize query vector
    query_norm = normalize(query_vector)

    # 4. For each document:
    scores = []
    for doc in documents:
        # Calculate TF-IDF vector
        doc_vector = calculate_tfidf_vector(doc)

        # Normalize
        doc_norm = normalize(doc_vector)

        # Cosine similarity = dot product of normalized vectors
        score = dot_product(query_norm, doc_norm)

        scores.append((doc, score))

    # 5. Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    # 6. Return ranked list
    return scores
```

---

## VSM Advantages and Limitations

### Advantages

1. **Ranked Retrieval**: Documents ordered by relevance
2. **Partial Matching**: Documents can match some query terms
3. **Term Weighting**: Important terms weighted higher
4. **Intuitive**: Vector space is easy to visualize (in 2D/3D)
5. **Effective**: Works well in practice

### Limitations

1. **Vocabulary Mismatch**:
   ```
   Query: "car"
   Document: "automobile"
   → Zero similarity! (different words, same meaning)
   ```

2. **Synonymy**: Same meaning, different words
   ```
   "laptop" vs "notebook"
   "physician" vs "doctor"
   ```

3. **Polysemy**: Same word, different meanings
   ```
   "bank" → financial institution or river bank?
   "apple" → fruit or computer company?
   ```

4. **Bag of Words**: Ignores word order
   ```
   "dog bites man" vs "man bites dog"
   → Same vector representation!
   ```

5. **Independence Assumption**: Terms treated as independent
   ```
   P("New York") ≠ P("New") × P("York")
   ```

**Solutions**:
- Synonymy → Dense retrieval (embeddings) - Module 4
- Better ranking → BM25, language models - Module 2
- Order → Phrase indexing, positional indexes

---

# Summary & Key Takeaways

## Module 1 Core Concepts

### Inverted Index
- **Structure**: Term → Postings list (document IDs)
- **Complexity**: O(x + y) for Boolean AND
- **Foundation**: All search engines use this!

### TF-IDF Formula
```
w(t,d) = (1 + log(tf(t,d))) × log(N/df(t))
```

- **TF**: Local importance (in document)
- **IDF**: Global importance (across collection)
- **Together**: Balanced term weighting

### Cosine Similarity
```
cos(q, d) = (q · d) / (|q| × |d|)
```

- **Measures**: Angle between vectors
- **Range**: [0, 1] for TF-IDF
- **Property**: Length-normalized

### Vector Space Model
- Documents and queries as vectors
- Similarity = cosine of angle
- Ranked retrieval

---

## What You Should Master

### Theory
- [ ] Explain inverted index structure and complexity
- [ ] Derive TF-IDF formula and explain each component
- [ ] Prove cosine similarity range is [0, 1]
- [ ] Explain why log-scaling for TF and IDF
- [ ] Describe Zipf's Law and Heap's Law

### Math
- [ ] Build inverted index by hand (5 documents)
- [ ] Calculate TF-IDF weights by hand
- [ ] Calculate cosine similarity by hand
- [ ] Rank documents manually

### Implementation
- [ ] Code inverted index from scratch
- [ ] Code TF-IDF calculation
- [ ] Code cosine similarity
- [ ] Build complete VSM search engine

### Conceptual
- [ ] When to use stemming vs lemmatization?
- [ ] When to remove stopwords?
- [ ] Why cosine better than Euclidean distance?
- [ ] Limitations of TF-IDF and VSM?

---

## Next Steps

1. **Work through toy_example.py**
   - Implement everything from scratch
   - Test on 5-10 simple documents
   - Verify calculations by hand

2. **Complete exercises.ipynb**
   - Practice problems
   - Build intuition
   - Test understanding

3. **Review class materials**
   - Week 1 & 2 slides
   - Lab 1 (Indexing)
   - Compare implementations

4. **Then move to Module 2**
   - Classical IR models
   - BM25 (most important!)
   - Advanced ranking

---

## References

### Manning Book
- **Chapter 1**: Boolean Retrieval (pages 1-20)
- **Chapter 2**: Term Vocabulary & Postings (pages 21-46)
- **Chapter 6**: Scoring, Term Weighting & VSM (pages 109-133)

### Additional Resources
- Porter Stemmer: https://tartarus.org/martin/PorterStemmer/
- NLTK Documentation: https://www.nltk.org/
- spaCy Documentation: https://spacy.io/

---

**You're now ready for the toy example and exercises!** 🎯

**Next**: `toy_example.py` - Implement everything we learned!
