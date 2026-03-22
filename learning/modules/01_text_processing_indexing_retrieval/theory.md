# Module 1: Text Processing, Indexing & Boolean Retrieval

**Duration:** 2-3 weeks
**Manning Chapters:** 1, 2, 4, 5 (partial)
**Slides:** Week 1, Week 2

---

## 📚 Table of Contents

### Part 1: Introduction to Information Retrieval

1.1 What is Information Retrieval?
1.2 The IR Problem
1.3 Precision and Recall

### Part 2: Text Processing & Tokenization

2.1 Document Delineation
2.2 Character Encoding (Unicode, UTF-8)
2.3 Tokenization Rules
2.4 Normalization
2.5 Stop Words
2.6 Stemming (Porter Algorithm)
2.7 Lemmatization
2.8 Statistical Laws (Zipf's Law, Heap's Law)

### Part 3: Inverted Index Construction

3.1 Term-Document Incidence Matrix
3.2 Inverted Index Structure
3.3 Sort-Based Indexing
3.4 Blocked Sort-Based Indexing (BSBI)
3.5 Single-Pass In-Memory Indexing (SPIMI)
3.6 Distributed Indexing (MapReduce)

### Part 4: Boolean Retrieval

4.1 Boolean Queries (AND, OR, NOT)
4.2 Postings List Intersection
4.3 Query Optimization
4.4 Skip Pointers

### Part 5: Phrase Queries & Positional Indexes

5.1 Biword Indexes
5.2 Positional Indexes
5.3 Positional Intersection Algorithm

### Part 6: Index Compression

6.1 Why Compression?
6.2 Dictionary Compression
6.3 Postings Compression (Gap Encoding)
6.4 Variable Byte Encoding
6.5 Gamma Codes
6.6 Delta Codes

---

## Part 1: Introduction to Information Retrieval

### 1.1 What is Information Retrieval?

**Definition:**

> Information Retrieval (IR) is finding material (usually documents) of an unstructured nature (usually text) that satisfies an information need from within large collections (usually stored on computers).

**Key Characteristics:**

- **Unstructured data**: No fixed schema like databases
- **Large scale**: Millions to billions of documents
- **Information need**: User's underlying question or goal
- **Relevance**: Not exact matching, but "good enough"

**Examples:**

- Web search engines (Google, Bing)
- Email search (Gmail, Outlook)
- Enterprise search (SharePoint, Elasticsearch)
- Digital libraries
- E-commerce product search

### 1.2 The IR Problem

**The Challenge:**
Given a collection of N documents and a user query, return the **most relevant** documents, ranked by relevance.

**Why is this hard?**

1. **Vocabulary mismatch**: User query words ≠ document words (synonyms, paraphrasing)
2. **Ambiguity**: Same word, different meanings (bank = financial institution or river bank)
3. **Scale**: Billions of documents, sub-second response time required
4. **Relevance is subjective**: What's relevant to one user may not be to another

**Simple Baseline: Grep**

- Linear scan through all documents
- Match query terms exactly
- Works for small collections (Shakespeare's works)
- **Fails** for web-scale (billions of documents, hours to search)

**Solution: Build an Index**

- Preprocess documents offline
- Create data structures for fast lookup
- Query time: O(query terms) not O(documents)

### 1.3 Precision and Recall

To evaluate IR systems, we need metrics:

**Precision (P):**

```
P = (# relevant documents retrieved) / (# total documents retrieved)
```

- "Of what I retrieved, how much was relevant?"
- High precision = few false positives

**Recall (R):**

```
R = (# relevant documents retrieved) / (# total relevant documents in collection)
```

- "Of all relevant documents, how many did I retrieve?"
- High recall = few false negatives

**The Precision-Recall Tradeoff:**

- Retrieve MORE → Recall ↑, Precision ↓
- Retrieve LESS → Precision ↑, Recall ↓
- Goal: Balance both (F1 score, MAP, NDCG - covered in Module 4)

---

## Part 2: Text Processing & Tokenization

### 2.1 Document Delineation

**Question:** What is a "document"?

**Answer:** Depends on the application!

- **Web search**: A single web page
- **Email**: One email message (with or without attachments?)
- **Books**: 
  - Entire book? 
  - Chapter? 
  - Paragraph?
- **Code**: Repository? File? Function?

**Index Granularity:**

- Too large (entire book) → Imprecise, users must scan long texts
- Too small (sentences) → Miss context, many false positives
- **Sweet spot**: Depends on user information needs

**For OpenSanctions:** Each entity (person, organization, vessel) is one document.

### 2.2 Character Encoding

**ASCII (7-bit):** Only English, 128 characters
**Extended ASCII (8-bit):** 256 characters, language-specific
**Unicode:** Universal character set, 143,000+ characters

**UTF-8 Encoding:**

- Variable-length: 1-4 bytes per character
- ASCII-compatible (first 128 characters)
- Dominant on the web (~98% of web pages)

**Example:**

- `A` (U+0041): `0x41` (1 byte)
- `é` (U+00E9): `0xC3 0xA9` (2 bytes)
- `中` (U+4E2D): `0xE4 0xB8 0xAD` (3 bytes)

**Why it matters:**

- Must decode correctly before tokenization
- Different encodings → different byte sequences for same character
- **Normalization**: Convert to standard form (usually UTF-8)

### 2.3 Tokenization Rules

**Definition:**
Tokenization = chopping character sequence into **tokens** (words)

**Simple rule:** Split on whitespace and punctuation

```python
text = "Friends, Romans, Countrymen, lend me your ears;"
tokens = text.split()
# Result: ['Friends,', 'Romans,', 'Countrymen,', 'lend', 'me', 'your', 'ears;']
```

**Problems:**

1. Punctuation attached to words
2. Possessives: `O'Neill` → `O` + `Neill` or `O'Neill`?
3. Hyphens: `state-of-the-art` → one token or many?
4. Numbers: `1,000,000` vs `1000000`
5. Dates: `01/15/2024` → tokenize how?
6. URLs: `http://www.example.com/path?query=value`
7. Emails: `user@domain.com`

**Language-Specific Challenges:**

**German:**

- Compound words: `Lebensversicherungsgesellschaftsangestellter` (life insurance company employee)
- Split or not? Affects retrieval!

**Chinese/Japanese:**

- No spaces between words!
- Need word segmentation algorithms
- Example: `信息检索` → `信息` (information) + `检索` (retrieval)

**Arabic/Hebrew:**

- Right-to-left scripts
- Vowel marks (diacritics) optional

**Best Practice:**

- Use language-specific tokenizers (NLTK, spaCy)
- Keep punctuation context when needed
- Normalize special characters

### 2.4 Normalization

**Goal:** Map tokens to same term even if written differently

**Case Folding:**

- Convert all to lowercase
- `Apple` = `apple` = `APPLE`

**Exception:** Sometimes case matters!

- `US` (United States) ≠ `us` (pronoun)
- `Apple` (company) ≠ `apple` (fruit)
- `Bush` (president) ≠ `bush` (plant)

**Solution:** Use capitalization as a feature, but usually fold for general search

**Accents and Diacritics:**

- `café` = `cafe`?
- `naïve` = `naive`?
- **Unicode normalization forms:**
  - NFC: Composed (é as single character)
  - NFD: Decomposed (e + combining accent)
  - NFKC/NFKD: Compatibility forms

**Example:**

```python
import unicodedata

text1 = "café"  # é as single character (U+00E9)
text2 = "café"  # e (U+0065) + combining accent (U+0301)

unicodedata.normalize('NFC', text1) == unicodedata.normalize('NFC', text2)  # True
```

**Other Normalizations:**

- Remove periods: `U.S.A.` → `USA`
- Expand contractions: `don't` → `do not`
- Handle currency: `$100` → `100 dollars`
- Date formats: `01/15/2024` → `2024-01-15`

### 2.5 Stop Words

**Definition:** Extremely common words with little semantic value

**Classic Stop Words:**

```
a, an, the, is, are, was, were, be, been, being,
have, has, had, do, does, did, will, would, could, should,
of, at, by, for, with, about, against, between, into,
through, during, before, after, above, below, to, from,
up, down, in, out, on, off, over, under
```

**Why remove them?**

1. **Frequency**: Account for 20-30% of tokens
2. **Storage**: Huge postings lists (the, of, to)
3. **Query time**: Expensive to process
4. **Relevance**: Don't discriminate between documents

**Why keep them?**

1. **Phrase queries**: "to be or not to be"
2. **Modern systems**: Fast enough, storage cheap
3. **Some are important**: "The Who" (band), "Take That" (band)

**Modern Approach:**

- Don't remove stop words during indexing
- Handle them specially during ranking (downweight)
- Use positional indexes for phrases

### 2.6 Stemming (Porter Algorithm)

**Goal:** Reduce words to their root form

**Examples:**

- `running` → `run`
- `ran` → `run`
- `runs` → `run`

**Why?** Improves recall (match related forms)

**Porter Stemmer (1980):**
Most widely used, rule-based algorithm with 5 steps

**Step 1:** Remove plurals and -ed/-ing

Rules (apply longest suffix):

```
SSES → SS        (caresses → caress)
IES  → I         (ponies → poni)
SS   → SS        (caress → caress)
S    → ε         (cats → cat)
```

```
(m>0) EED → EE   (agreed → agree, feed → feed)
(*v*) ED  → ε    (matted → mat, bled → bled)
(*v*) ING → ε    (motoring → motor, sing → sing)
```

**Measure m:**

- Sequence of: [C](VC){m}[V]
- C = consonant, V = vowel
- m = number of VC sequences

**Examples:**

- `TR` → m=0
- `TREE` → m=0
- `TREES` → m=1 (VC)
- `TROUBLE` → m=1
- `TROUBLES` → m=2 (VC-VC)

**Conditions:**

- `(m>0)`: Measure > 0
- `(*v*)`: Contains vowel
- `(*d)`: Ends with double consonant
- `(*o)`: Ends with CVC where last C is not w, x, or y

**Step 2:** More suffix removal

```
(m>0) ATIONAL → ATE      (relational → relate)
(m>0) TIONAL → TION      (conditional → condition)
(m>0) ENCI → ENCE        (valenci → valence)
(m>0) ANCI → ANCE        (hesitanci → hesitance)
(m>0) IZER → IZE         (digitizer → digitize)
```

**Step 3:** -IC-, -FULL, -NESS suffixes

```
(m>0) ICATE → IC         (triplicate → triplic)
(m>0) ATIVE → ε          (formative → form)
(m>0) ALIZE → AL         (formalize → formal)
(m>0) ICITI → IC         (electricity → electric)
(m>0) ICAL → IC          (electrical → electric)
(m>0) FUL → ε            (hopeful → hope)
(m>0) NESS → ε           (goodness → good)
```

**Step 4:** Remove -ANT, -ENCE, etc.

```
(m>1) AL → ε             (revival → reviv)
(m>1) ANCE → ε           (allowance → allow)
(m>1) ENCE → ε           (inference → infer)
(m>1) ER → ε             (airliner → airlin)
(m>1) IC → ε             (gyroscopic → gyroscop)
(m>1) ABLE → ε           (adjustable → adjust)
(m>1) IBLE → ε           (defensible → defens)
(m>1) ANT → ε            (irritant → irrit)
(m>1) EMENT → ε          (replacement → replac)
(m>1) MENT → ε           (adjustment → adjust)
(m>1) ENT → ε            (dependent → depend)
```

**Step 5:** Remove final -E

```
(m>1) E → ε              (probate → probat, rate → rate)
(m=1 and not *o) E → ε   (cease → ceas, horse → hors)
```

And remove double consonants in -LL:

```
(m>1 and *d and *L) → single letter
                         (controll → control)
```

**Complete Example:**

Let's stem `"computational"`:

1. **Initial:** `computational`
2. **Step 1:** No change (no -ED/-ING)
3. **Step 2:** `(m>0) ATIONAL → ATE`
  - `computate` (wait, check measure)
  - Measure of `comput` = 2 (VC-VC)
  - m > 0 ✓
  - Result: `computate` (wrong! should apply more)
4. Actually in Porter: `computational` → `comput` (through multiple steps)

**Correct sequence:**

- `computational` → `computation` (step 2: -AL)
- `computation` → `compute` (step 4: -ION if m>1)
- Actually: Porter gives `comput` (removes -E in step 5)

**Pros:**

- Fast, simple
- Language-independent rules (but designed for English)
- Widely used, many implementations

**Cons:**

- Over-stemming: `universal` → `univers`, `university` → `univers` (same!)
- Under-stemming: `alumnus` vs `alumni`, `create` vs `creation`
- Not linguistically motivated
- Errors on irregular forms

### 2.7 Lemmatization

**Definition:** Reduce words to dictionary form (lemma) using vocabulary and morphological analysis

**Examples:**

- `am, are, is` → `be`
- `was, were` → `be`
- `better` → `good`
- `ran` → `run`

**Difference from Stemming:**


| Stemming                 | Lemmatization                        |
| ------------------------ | ------------------------------------ |
| Crude chopping           | Vocabulary lookup                    |
| `saw` → `s`              | `saw` → `see` (verb) or `saw` (noun) |
| `university` → `univers` | `university` → `university`          |
| Fast                     | Slower (needs POS tagging)           |
| No meaning               | Meaningful words                     |


**Example with WordNet:**

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("running", pos='v'))  # run
print(lemmatizer.lemmatize("better", pos='a'))   # good
print(lemmatizer.lemmatize("geese", pos='n'))    # goose
```

**Part-of-Speech (POS) Matters:**

- `meeting` as noun → `meeting`
- `meeting` as verb → `meet`

**Tools:**

- **WordNet**: English lexical database
- **spaCy**: Modern, fast, neural lemmatizer
- **TreeTagger**: Multi-language

**When to Use:**

- **Stemming**: Speed matters, approximate matching OK
- **Lemmatization**: Accuracy matters, have POS information

**For OpenSanctions:**

- Entity names: Usually no stemming/lemmatization
- Descriptions: Lemmatization for better matching
- Addresses: Normalize but don't stem

### 2.8 Statistical Properties of Text

Understanding the statistical properties of text is crucial for designing efficient IR systems. Two fundamental laws describe how vocabulary grows and how term frequencies are distributed.

#### Zipf's Law: The Power Law of Word Frequencies

**Zipf's Law** is one of the most robust empirical laws in linguistics, stating that the frequency of a term is inversely proportional to its rank in the frequency table.

##### Mathematical Formulation

**Basic form:**
$$cf_i \propto \frac{1}{i}$$

**Precise formulation:**
$$cf_i = \frac{c}{i^\alpha}$$

Where:
- $cf_i$ = collection frequency of the $i$-th most common term
- $i$ = rank (1 = most frequent term, 2 = second most frequent, etc.)
- $c$ = normalization constant (depends on collection size)
- $\alpha$ = exponent (typically $\alpha \approx 1$ for natural language text)

**Log-log form:**
Taking logarithms of both sides:
$$\log(cf_i) = \log(c) - \alpha \cdot \log(i)$$

This linear relationship in log-log space means that if we plot $\log(cf_i)$ against $\log(i)$, we get a straight line with:
- **Slope**: $-\alpha$ (typically around -1)
- **Intercept**: $\log(c)$

##### Why α ≈ 1 is Important

When $\alpha = 1$, we have the classic Zipf distribution. This means:
- The most frequent word occurs approximately twice as often as the second most frequent
- The second most frequent occurs twice as often as the fourth most frequent
- The frequency decreases in a hyperbolic fashion

##### Empirical Evidence

**Example 1: Reuters-RCV1 corpus (806,791 documents, 100M+ tokens)**

| Rank $i$ | Term | Collection Frequency $cf_i$ | Predicted $(c/i)$ | Ratio |
|----------|------|--------------------------|-------------------|-------|
| 1 | the | 7,253,381 | 7,253,381 | 1.00 |
| 10 | said | 496,969 | 725,338 | 0.68 |
| 100 | like | 60,289 | 72,534 | 0.83 |
| 1,000 | earned | 7,326 | 7,253 | 1.01 |
| 10,000 | carefully | 721 | 725 | 0.99 |

Notice how the law holds better for mid-frequency and rare terms than for very frequent terms.

**Example 2: Web corpus (1 trillion tokens)**
Even at web scale, Zipf's Law continues to hold, though with slight variations in $\alpha$ depending on the domain (news: $\alpha \approx 1.0$, scientific text: $\alpha \approx 0.9$, social media: $\alpha \approx 1.1$).

##### Theoretical Interpretation

**Why does Zipf's Law exist?** Several theories:

1. **Least effort principle**: Balance between speaker effort (few distinct words) and listener comprehension (many distinct words)
2. **Generative process**: Random texts generated by preferential attachment exhibit Zipfian distribution
3. **Information theory**: Optimal coding of messages in a communication channel

**Mathematical insight**: If we have $N$ total tokens and rank $i$ has frequency $cf_i$, then:
$$\sum_{i=1}^{M} cf_i = N$$

For Zipf distribution with $\alpha = 1$:
$$\sum_{i=1}^{M} \frac{c}{i} = N \implies c \sum_{i=1}^{M} \frac{1}{i} = N$$

The sum $\sum_{i=1}^{M} \frac{1}{i}$ is the harmonic series, which for large $M$ approximates $\ln(M)$. Thus:
$$c \approx \frac{N}{\ln(M)}$$

This means the constant $c$ is proportional to collection size $N$ and inversely proportional to the log of vocabulary size.

##### Deviations from Zipf's Law

Real text exhibits deviations:

1. **Head of distribution** (very frequent terms): Function words like "the", "of", "and" occur MORE frequently than predicted
2. **Tail of distribution** (rare terms): Hapax legomena (terms appearing once) are LESS frequent than predicted
3. **Middle of distribution** (content words): Best fit to the power law

**Modified Zipf-Mandelbrot Law** accounts for these deviations:
$$cf_i = \frac{c}{(i + b)^\alpha}$$

where $b$ is a correction factor (typically $b \approx 2.7$ for English).

##### Practical Implications for IR

**1. Compression Strategy**
- Assign variable-length codes to terms based on frequency
- Most frequent 100 terms account for ~50% of all tokens
- Can achieve high compression ratios by optimizing codes for these terms

**2. Query Processing Optimization**
```
If query = "the history of computational linguistics":
- "the" appears in 50% of documents → not discriminative
- "computational" appears in 0.1% of documents → highly discriminative
- Process rare terms first to minimize postings list intersections
```

**3. Caching Strategy**
- Top 1% of terms by frequency account for 40-50% of query traffic
- Cache their postings lists in memory
- Memory usage: If vocabulary has $M = 10^6$ terms, cache top $10^4$ terms
- Hit rate: 40-50% of postings lookups served from cache

**4. Stop Word Selection**
- Very high-frequency terms (top 0.1% by rank) are often stop words
- For Reuters corpus: top 100 terms are mostly function words
- These contribute little to document relevance

**5. Vocabulary Estimation**
Given Zipf's Law, if we sample $n$ tokens and observe $m$ distinct terms, we can estimate total vocabulary $M$ for full collection of $N$ tokens using statistical extrapolation techniques.

##### Visualizing Zipf's Law

**Log-log plot** (typical for English text):
```
log(cf)
   ^
   |     •  (rank 1: "the")
   |      \
   |       •  (rank 10: "said")
   |        \
   |         •  (rank 100: "like")
   |          \___
   |              •  (rank 1000: "earned")
   |               \___
   |                   •• (rank 10000+: rare terms)
   +---------------------------------> log(rank)
```

The slope of this line is approximately -1, confirming $\alpha \approx 1$.

---

#### Heap's Law: Vocabulary Growth

**Heap's Law** (also called **Herdan's Law**) describes how vocabulary size grows sublinearly with collection size. This is one of the most important laws for predicting index size in IR systems.

##### Mathematical Formulation

**Formula:**
$$M = kT^b$$

Where:
- $M$ = vocabulary size (number of distinct terms)
- $T$ = total number of tokens in the collection
- $k$ = characteristic constant ($30 \le k \le 100$ for English)
- $b$ = growth exponent ($0.4 \le b \le 0.6$, typically $b \approx 0.5$)

**Taking logarithms:**
$$\log(M) = \log(k) + b \cdot \log(T)$$

This linear relationship in log-log space means vocabulary growth is a **power law**.

##### Why b < 1 is Critical

The key insight: **$b < 1$ means sublinear growth**.

- If $b = 1$: vocabulary would grow linearly with collection size (every new token is a new term) → unrealistic
- If $b = 0$: vocabulary would be constant (fixed vocabulary) → too restrictive
- $b \approx 0.5$: vocabulary grows as the square root of collection size → realistic for natural language

**Example:**
- Collection with $T = 1{,}000{,}000$ tokens has $M \approx 44 \times (10^6)^{0.5} = 44{,}000$ distinct terms
- Doubling to $T = 2{,}000{,}000$ tokens: $M \approx 44 \times (2 \times 10^6)^{0.5} = 62{,}225$ terms
- Vocabulary increased by only 41% while collection size doubled

##### Empirical Evidence

**Example 1: Reuters-RCV1 corpus**

| Tokens $T$ | Distinct Terms $M$ | Predicted ($44T^{0.49}$) | Error |
|------------|-------------------|------------------------|-------|
| 1,000,020 | 38,323 | 38,280 | 0.1% |
| 10,000,000 | 95,734 | 97,165 | 1.5% |
| 100,000,000 | 391,523 | 387,896 | 0.9% |

Parameters: $k = 44$, $b = 0.49$

**Example 2: Different text genres**

| Corpus Type | $k$ | $b$ | Notes |
|-------------|-----|-----|-------|
| News (Reuters) | 44 | 0.49 | Controlled vocabulary |
| Web pages | 65 | 0.53 | More variation (typos, names) |
| Scientific papers | 35 | 0.45 | Technical terminology, less variation |
| Social media | 75 | 0.55 | Highest variation (slang, typos, neologisms) |

**Key observation:**
- Higher $k$ → larger initial vocabulary
- Higher $b$ → faster vocabulary growth
- Web and social media have both higher $k$ and higher $b$ due to informal language, typos, and creative word usage

##### Theoretical Interpretation

**Why does vocabulary grow sublinearly?**

1. **Finite vocabulary of a language**: Most languages have a core vocabulary of 10,000-100,000 words
2. **Zipfian distribution**: Common words are reused frequently, rare words appear occasionally
3. **Sampling process**: As collection grows, probability of encountering new terms decreases

**Stochastic model:**
- Probability of a new term appearing in position $t$ decreases as $t$ increases
- If $P(\text{new term at position } t) \propto t^{-(1-b)}$, then vocabulary grows as $M \propto T^b$

**Connection to Zipf's Law:**
There's a mathematical relationship between Zipf and Heap's laws. If term frequencies follow Zipf's Law with exponent $\alpha$, then vocabulary growth follows Heap's Law with:
$$b = \frac{1}{\alpha + 1}$$

For $\alpha = 1$ (classical Zipf), we get $b = 0.5$ (classical Heap).

##### Does Vocabulary Ever Stop Growing?

**Answer: No!** Even after billions of tokens, new terms continue to appear.

**Sources of new terms:**
1. **Neologisms**: "selfie", "blockchain", "COVID-19"
2. **Proper nouns**: New people, places, organizations
3. **Typos and misspellings**: Essentially infinite in web text
4. **Compound words**: "machine-learning-based-approach"
5. **Numbers and codes**: "AB123XYZ", "192.168.1.1"
6. **Foreign words**: Borrowed terms from other languages

**Empirical evidence from web corpora:**
- Google's trillion-word corpus: vocabulary still growing after $10^{12}$ tokens
- Growth rate slows but never reaches zero
- Hapax legomena (terms appearing once) constitute 40-60% of vocabulary

##### Practical Implications for IR

**1. Index Size Estimation**

Given a collection of $T$ tokens, we can predict:

**Dictionary size:**
$$M = kT^b \approx 44T^{0.5} \text{ terms}$$

**Example:** For $T = 10$ billion tokens:
$$M = 44 \times (10^{10})^{0.5} = 44 \times 10^5 = 4{,}400{,}000 \text{ distinct terms}$$

**Memory requirements:**
- If each dictionary entry is 20 bytes (term pointer + metadata): $4.4M \times 20 = 88$ MB
- Actual index with postings lists: typically 20-40% of raw text size

**2. Memory Planning**

When building an inverted index in memory:
```python
# Estimate memory for dictionary during SPIMI
T_total = 1_000_000_000  # 1 billion tokens
M = 44 * (T_total ** 0.49)  # ~95,000 distinct terms
memory_per_term = 50  # bytes (term string + postings pointer)
dictionary_memory = M * memory_per_term  # ~4.75 MB

# Plus postings lists memory
postings_memory = T_total * 8  # 8 bytes per posting (docID + frequency)
total_memory = dictionary_memory + postings_memory
```

**3. Scaling Predictions**

**Question:** If we have an index for 1 million documents, how much bigger will the index be for 10 million documents?

**Answer using Heap's Law:**
- Assume $T_1 = 10^9$ tokens (1M docs) → $M_1 = 95{,}000$ terms
- For $T_2 = 10^{10}$ tokens (10M docs) → $M_2 = 300{,}000$ terms
- Dictionary grows by factor of 3.16, not 10
- Postings lists grow by factor of 10 (linear in $T$)

**Total index size:**
$$\text{Index size} \propto M \log(T) + T$$
where the first term is dictionary (with compression) and second is postings.

**4. Dynamic Indexing**

For streaming data, Heap's Law tells us the rate of new terms:

**Derivative of Heap's Law:**
$$\frac{dM}{dT} = kbT^{b-1}$$

For $k = 44$, $b = 0.49$:
$$\frac{dM}{dT} = 44 \times 0.49 \times T^{-0.51} \approx \frac{21.5}{\sqrt{T}}$$

**Interpretation:** After processing $T = 1{,}000{,}000$ tokens, we encounter approximately:
$$\frac{dM}{dT} \approx \frac{21.5}{1{,}000} = 0.0215 \text{ new terms per token}$$

That is, about 1 new term every 46 tokens.

**5. Distributed Indexing**

When partitioning a collection across $n$ machines:

**By document (each machine indexes $T/n$ tokens):**
- Each machine's vocabulary: $M_i = k(T/n)^b$
- Total vocabulary (union): $M \approx k T^b$ (same as centralized)
- Dictionary merging overhead: minimal

**By term (vocabulary partitioned):**
- Each machine handles $M/n$ terms
- No vocabulary overhead
- But postings lists are scattered

##### Heap's Law in Different Scenarios

**Scenario 1: Adding documents to existing collection**
- Current: $T_1 = 1B$ tokens, $M_1 = 95K$ terms
- Add: $\Delta T = 100M$ tokens
- New total: $T_2 = 1.1B$ tokens
- New vocabulary: $M_2 = 44 \times (1.1 \times 10^9)^{0.49} \approx 99.5K$ terms
- New terms: $\Delta M = 4.5K$ terms

**Scenario 2: Multilingual collections**
- Each language has its own $(k, b)$ parameters
- Total vocabulary is nearly additive: $M_{\text{total}} \approx M_{\text{lang1}} + M_{\text{lang2}} + \ldots$
- Cross-language term overlap is minimal (~1-5% for related languages)

**Scenario 3: Temporal collections (news archives)**
- Older text: stable vocabulary, lower $k$
- Recent text: growing vocabulary (new named entities, neologisms), higher $k$
- Need to periodically re-estimate $(k, b)$ parameters

##### Limitations and Extensions

**When Heap's Law breaks down:**

1. **Very small collections** ($T < 10{,}000$ tokens): High variance, poor fit
2. **Artificially constrained vocabulary**: Controlled vocabularies (medical codes, product catalogs)
3. **Highly formulaic text**: Legal boilerplate, generated reports
4. **Multilingual mixed text**: Different $(k, b)$ per language

**Generalized Heap's Law:**
For non-uniform text, some researchers use a more flexible form:
$$M = k_1 T^{b_1} + k_2 T^{b_2}$$
where the first term models common vocabulary and second models rare/new terms.

##### Visualizing Heap's Law

**Log-log plot:**
```
log(M)
   ^
   |                        •  (web corpus)
   |                    •
   |                •
   |            •
   |        •  (Reuters corpus)
   |    •
   |•
   +---------------------------------> log(T)

   Slope = b ≈ 0.49
```

**Growth comparison:**
```
M (vocabulary size)
   ^
   | Heap's Law: M = kT^0.5
   |             ___---•••
   |        ___---
   |    ___--
   | •--
   +---------------------------------> T (tokens)

   Compare to linear: M = kT
                      |                      •
                      |                 •
                      |            •
                      |       •
                      |  •
                      +----------------------> T
```

##### Summary: Key Takeaways

**Zipf's Law:**
- Frequency ∝ 1/rank
- Most terms are rare; few terms are very frequent
- Impacts: compression, caching, query processing, stop word selection

**Heap's Law:**
- Vocabulary size $M = kT^b$ where $b < 1$
- Sublinear growth: vocabulary grows slower than collection
- Impacts: index size estimation, memory planning, scalability analysis

**Together, these laws:**
- Explain why indexes are smaller than raw text (Zipf enables compression)
- Predict scaling behavior (Heap predicts dictionary growth)
- Guide system design decisions (caching, partitioning, memory allocation)

---

## Part 3: Inverted Index Construction

### 3.1 Term-Document Incidence Matrix

**Naive representation:** Matrix where:

- Rows = terms
- Columns = documents
- Cell (t,d) = 1 if term t in document d, else 0

**Example:**


|           | Antony | Julius | The | Hamlet | Othello | Macbeth |
| --------- | ------ | ------ | --- | ------ | ------- | ------- |
| Antony    | 1      | 1      | 0   | 0      | 0       | 1       |
| Brutus    | 1      | 1      | 0   | 1      | 0       | 0       |
| Caesar    | 1      | 1      | 0   | 1      | 1       | 1       |
| Calpurnia | 0      | 1      | 0   | 0      | 0       | 0       |
| mercy     | 1      | 0      | 1   | 1      | 1       | 1       |
| worser    | 1      | 0      | 1   | 1      | 1       | 0       |


**Query:** `Brutus AND Caesar AND NOT Calpurnia`

**Boolean algebra:**

```
110100 AND 110111 AND NOT 010000
= 110100 AND 110111 AND 101111
= 100100
```

**Answer:** Antony and Cleopatra, Hamlet

**Problem:** Matrix is huge and sparse!

- N = 1M documents, M = 500K terms
- Matrix size: 1M × 500K = 500 billion cells
- If each cell is 1 bit: 62.5 GB
- But **99.8%+ are zeros!** (sparse)

**Solution:** Inverted Index (store only 1s)

### 3.2 Inverted Index Structure

**Idea:** For each term, store list of documents it appears in

**Components:**

1. **Dictionary (Lexicon):** Terms in sorted order
2. **Postings Lists:** For each term, list of document IDs

**Example:**

```
Brutus    → [1, 2, 4, 11, 31, 45, 173, 174]
Caesar    → [1, 2, 4, 5, 6, 16, 57, 132, ...]
Calpurnia → [2, 31, 54, 101]
```

**Dictionary stored in memory, Postings on disk**

**Space Savings:**

- Only store non-zero entries
- Typical compression: 1:4 ratio (vs dense matrix)

**Document Frequency (df):**

- df(Brutus) = 8 (appears in 8 documents)
- Store df in dictionary for query optimization

### 3.3 Sort-Based Indexing

**Goal:** Build inverted index from document collection

**Steps:**

1. **Tokenize** documents
2. **Create** (term, docID) pairs
3. **Sort** pairs by term (then by docID)
4. **Group** by term to create postings lists

**Example:**

**Doc 1:** "I did enact Julius Caesar"
**Doc 2:** "So let it be with Caesar"

**Step 1: Tokenization and pairs**

```
[(I,1), (did,1), (enact,1), (julius,1), (caesar,1),
 (so,2), (let,2), (it,2), (be,2), (with,2), (caesar,2)]
```

**Step 2: Sort by term**

```
[(be,2), (caesar,1), (caesar,2), (did,1), (enact,1),
 (I,1), (it,2), (julius,1), (let,2), (so,2), (with,2)]
```

**Step 3: Group and create postings**

```
be      → [2]
caesar  → [1, 2]
did     → [1]
enact   → [1]
I       → [1]
it      → [2]
julius  → [1]
let     → [2]
so      → [2]
with    → [2]
```

**Complexity:**

- Time: O(T log T) where T = total tokens
- Space: O(T) (all pairs in memory)

**Problem:** What if collection doesn't fit in memory?

### 3.4 Blocked Sort-Based Indexing (BSBI)

**Idea:** Process collection in blocks that fit in memory

**Algorithm:**

```
BSBI(collection):
    blocks = []

    while (documents remain):
        block = read_next_block(collection)  # Fit in memory
        pairs = []

        for doc in block:
            for term in tokenize(doc):
                pairs.append((term, docID))

        pairs.sort()  # In-memory sort
        postings = group_into_postings(pairs)
        block_index = write_to_disk(postings)
        blocks.append(block_index)

    # Merge all block indexes
    final_index = merge_blocks(blocks)
    return final_index
```

**Merge Phase:**

- Like merge sort: merge sorted postings lists
- k-way merge (k = number of blocks)
- Output: single sorted inverted index

**Example Merge:**

Block 1:

```
brutus  → [1, 2, 4]
caesar  → [1, 5, 6]
noble   → [4]
```

Block 2:

```
brutus  → [11, 31, 45]
caesar  → [16, 57, 132]
the     → [2, 11]
```

Merged:

```
brutus  → [1, 2, 4, 11, 31, 45]
caesar  → [1, 5, 6, 16, 57, 132]
noble   → [4]
the     → [2, 11]
```

**Complexity:**

- Time: O(T log T) sorting + O(T) merging = O(T log T)
- Space: O(B) where B = block size (not T!)
- Disk I/O: O(T) (2 passes: write blocks, read for merge)

**Parameters:**

- Block size: Fit in available memory (e.g., 1 GB)
- For Reuters-RCV1 (6 GB): Need ~6-10 blocks

### 3.5 Single-Pass In-Memory Indexing (SPIMI)

**Problem with BSBI:** Must store all (term, docID) pairs before sorting

**SPIMI Idea:** Build dictionary incrementally, write blocks when memory full

**Key Difference:**

- BSBI: Create all pairs → sort → group
- SPIMI: Create postings lists on-the-fly

**Algorithm:**

```
SPIMI(collection):
    output_blocks = []
    dictionary = {}

    while (documents remain):
        while (memory available):
            doc = next_document()
            for term in tokenize(doc):
                if term not in dictionary:
                    dictionary[term] = create_postings_list()
                postings = dictionary[term]
                postings.append(docID)

        # Memory full, write block
        sorted_terms = sort(dictionary.keys())
        block_file = write_block(sorted_terms, dictionary)
        output_blocks.append(block_file)
        dictionary = {}  # Clear for next block

    final_index = merge_blocks(output_blocks)
    return final_index
```

**Advantages:**

1. **Faster:** No large sort (only sort terms, not pairs)
2. **Dynamic memory:** Postings lists grow as needed
3. **Larger blocks:** Can process more before writing

**Memory Management:**

- Allocate postings lists dynamically (linked lists or arrays)
- When full: double size (amortized O(1) append)
- Total memory: O(terms in block + postings)

**Comparison BSBI vs SPIMI:**


|            | BSBI                 | SPIMI                 |
| ---------- | -------------------- | --------------------- |
| Sort       | All pairs O(T log T) | Just terms O(M log M) |
| Memory     | Fixed (term, docID)  | Dynamic postings      |
| Block size | Smaller              | Larger                |
| Speed      | Slower               | Faster (2-3x)         |


### 3.6 Distributed Indexing (MapReduce)

**Problem:** Web-scale collections (billions of documents)

**Solution:** Distribute across many machines

**MapReduce Framework:**

```
Input → Map → Shuffle → Reduce → Output
```

**Index Construction:**

**Map Phase:**

```python
def map(docID, doc_content):
    for term in tokenize(doc_content):
        emit(term, docID)
```

Output: Stream of (term, docID) pairs

**Shuffle Phase:**

- Group all pairs by term
- Route to reducers (hash(term) % num_reducers)

**Reduce Phase:**

```python
def reduce(term, list_of_docIDs):
    postings = sorted(set(list_of_docIDs))
    write_to_index(term, postings)
```

**Example:**

**Mappers:**

```
Mapper 1: (brutus, 1), (caesar, 1), (noble, 1)
Mapper 2: (brutus, 3), (brutus, 5), (caesar, 3)
Mapper 3: (caesar, 7), (the, 7), (noble, 8)
```

**After Shuffle (group by term):**

```
brutus → [1, 3, 5]
caesar → [1, 3, 7]
noble  → [1, 8]
the    → [7]
```

**Reducers write final postings**

**Partitioning Strategies:**

1. **Term Partitioning:**
  - Each reducer handles subset of terms
  - Unbalanced: frequent terms on one machine
2. **Document Partitioning:**
  - Each mapper handles subset of documents
  - Better load balancing
  - Must merge at the end

**Google's Approach (circa 2004):**

- 5000+ mappers
- 50+ reducers
- Process 20 TB in 2-3 hours
- Produce 1 TB compressed index

---

## Part 4: Boolean Retrieval

### 4.1 Boolean Queries

**Boolean Model:** Documents either match or don't match

**Operators:**

- **AND:** Both terms must appear
- **OR:** At least one term appears
- **NOT:** Term must not appear

**Examples:**

- `information AND retrieval`
- `information OR retrieval`
- `information AND NOT database`
- `(information OR data) AND retrieval`

**Query Processing:**

Given query: `Brutus AND Calpurnia`

1. Locate `Brutus` in dictionary → get postings
2. Locate `Calpurnia` in dictionary → get postings
3. **Intersect** postings lists
4. Return document IDs

### 4.2 Postings List Intersection

**Given two sorted postings lists, find common documents**

**Algorithm (Linear Merge):**

```python
def intersect(p1, p2):
    """
    p1, p2: sorted lists of docIDs
    Returns: sorted list of common docIDs
    """
    answer = []
    i, j = 0, 0

    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            answer.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1

    return answer
```

**Example:**

```
Brutus    → [1, 2, 4, 11, 31, 45, 173, 174]
Calpurnia → [2, 31, 54, 101]

Intersection:
i=0, j=0: 1 < 2, advance i
i=1, j=0: 2 = 2, add 2, advance both
i=2, j=1: 4 < 31, advance i
i=3, j=1: 11 < 31, advance i
i=4, j=1: 31 = 31, add 31, advance both
i=5, j=2: 45 < 54, advance i
i=6, j=2: 173 > 54, advance j
i=6, j=3: 173 > 101, advance j
j=4: done

Result: [2, 31]
```

**Complexity:**

- Time: O(x + y) where x, y are lengths of postings lists
- Space: O(min(x, y)) for result
- **Linear time!** Much better than O(xy) naive approach

**Multi-Term AND:**

For `term1 AND term2 AND ... AND termN`:

```python
def intersect_multiple(postings_lists):
    # Sort by length (shortest first)
    postings_lists.sort(key=len)

    result = postings_lists[0]
    for postings in postings_lists[1:]:
        result = intersect(result, postings)
        if not result:  # Early termination
            return []
    return result
```

**Why process shortest first?**

- Result size ≤ min(all lists)
- Intermediate results smaller
- Fewer comparisons overall

### 4.3 Query Optimization

**Problem:** Order of operations matters for efficiency

**Example:**

Query: `Brutus AND Caesar AND Calpurnia`

**Option 1:** (Brutus AND Caesar) AND Calpurnia

```
|Brutus| = 100,000
|Caesar| = 50,000
|Brutus ∩ Caesar| ≈ 10,000
|Calpurnia| = 50
|Result| ≈ 5
```

Work: 100,000 + 50,000 + 10,000 + 50 = 160,050 comparisons

**Option 2:** Calpurnia AND (Brutus AND Caesar)

```
|Calpurnia| = 50
|Brutus| = 100,000
|Calpurnia ∩ Brutus| ≈ 40
|Caesar| = 50,000
|Result| ≈ 5
```

Work: 50 + 100,000 + 40 + 50,000 ≈ 150,090 comparisons

**Best Option:** Process in increasing order of df

```
Calpurnia (50) AND Brutus (100,000) AND Caesar (50,000)
```

Work: 50 + 40 + 5 ≈ 95 comparisons (much better!)

**Heuristic:** Process terms in order of increasing document frequency

**Implementation:**

- Store df in dictionary
- Sort query terms by df before processing
- Process smallest postings list first

### 4.4 Skip Pointers

**Problem:** Linear intersection can be slow for large lists

**Idea:** Add pointers to "skip ahead" in postings list

**Structure:**

```
Brutus → [1] → [2] → [4] → [11] → [31] → [45] → [173] → [174]
          ↓           ↓             ↓              ↓
         [4]         [31]          [173]         [NULL]
```

Skip pointers point √n positions ahead (approximately)

**Modified Intersection:**

```python
def intersect_with_skips(p1, p2):
    answer = []
    i, j = 0, 0

    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            answer.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            # Try to skip
            if has_skip(p1, i) and skip_to(p1, i) <= p2[j]:
                i = skip_pointer(p1, i)
            else:
                i += 1
        else:
            if has_skip(p2, j) and skip_to(p2, j) <= p1[i]:
                j = skip_pointer(p2, j)
            else:
                j += 1

    return answer
```

**Example:**

```
List 1: [1, 4, 11, 31, 45, 173]
        skip: 1→11, 4→31, 11→173

List 2: [2, 31, 54, 101]
        skip: 2→54, 31→[end]

Compare 1 vs 2: 1 < 2, skip to 11
Compare 11 vs 2: 11 > 2, advance to 31
Compare 11 vs 31: 11 < 31, skip to 173
Compare 173 vs 31: 173 > 31, skip to 54
Compare 173 vs 54: 173 > 54, advance to 101
Compare 173 vs 101: 173 > 101, done

(Missed 31! Skip was too aggressive)
```

**Trade-offs:**

**Benefits:**

- Fewer comparisons (skip over non-matches)
- Especially helpful when lists have different lengths

**Costs:**

- Extra storage (skip pointers)
- More complex code
- Might miss matches if skip too far

**Optimal Skip Distance:**

**Theorem:** For list of length n, optimal skip distance is √n

**Proof sketch:**

- Number of skips: n/s (where s = skip distance)
- Storage for skips: O(n/s)
- Expected comparisons saved: O(s)
- Total work: n/s + n - ks (k = constant)
- Minimize: d/ds[n/s + n - ks] = 0
- Solve: -n/s² - k = 0 → s = √(n/k) ≈ √n

**Practical choices:**

- √n (theoretical optimum)
- Powers of 2 (easier to compute)
- Fixed intervals (e.g., every 128 docIDs)

**When to use skip pointers:**

- Large postings lists (> 10,000 entries)
- Queries with disparate term frequencies
- NOT worth it for small lists (<1000)

---

## Part 5: Phrase Queries & Positional Indexes

### 5.1 Biword Indexes

**Problem:** User searches for exact phrase "information retrieval"

**Naive approach:**

- Find docs with "information" AND "retrieval"
- Post-process: verify they're adjacent

**Better:** Index word pairs (biwords)

**Biword Index:**

- Index every consecutive pair of words
- "Friends, Romans, Countrymen" →
  - "friends romans"
  - "romans countrymen"

**Example:**

Document: "The quick brown fox jumps over the lazy dog"

Biwords:

```
"the quick"
"quick brown"
"brown fox"
"fox jumps"
"jumps over"
"over the"
"the lazy"
"lazy dog"
```

**Query:** "brown fox jumps"

Process as: "brown fox" AND "fox jumps"

**Advantages:**

- Simple
- Fast phrase matching

**Disadvantages:**

- Index size: ~2× larger (every adjacent pair)
- Can't handle arbitrary proximity (e.g., within 5 words)
- Can't handle long phrases efficiently

**Extended Biwords:**

- Use nouns and content words only
- "The cost of living" → "cost living"
- Reduces index size, loses some precision

### 5.2 Positional Indexes

**Idea:** Store positions of each term in each document

**Structure:**

```
term → docID: [pos1, pos2, ...]
```

**Example:**

Doc 1: "The cat sat on the mat"

Positional index:

```
cat → 1: [2]
mat → 1: [6]
on  → 1: [4]
sat → 1: [3]
the → 1: [1, 5]
```

**Full inverted index:**

```
cat → 1:[2], 17:[5], 25:[1], ...
mat → 1:[6], 8:[3], ...
on  → 1:[4], 2:[1,8,12], 5:[3], ...
sat → 1:[3], 13:[2], ...
the → 1:[1,5], 2:[4,9], 3:[1,2,6], ...
```

**Space:**

- Basic index: O(T) where T = total tokens
- Positional index: O(T) + O(T) = O(2T)
- Typically 2-4× larger than non-positional

### 5.3 Positional Intersection Algorithm

**Goal:** Find documents where term1 immediately precedes term2

**Algorithm:**

```python
def positional_intersect(p1, p2, k):
    """
    p1, p2: postings with positions {docID: [pos1, pos2, ...]}
    k: maximum allowed gap (k=1 for adjacent)
    Returns: docIDs where terms appear within k positions
    """
    answer = []

    # Iterate over document IDs in both postings
    docs1 = set(p1.keys())
    docs2 = set(p2.keys())

    for docID in docs1.intersection(docs2):
        positions1 = p1[docID]
        positions2 = p2[docID]

        # Check if any position in p1 is k positions before p2
        matched = False
        for pos1 in positions1:
            for pos2 in positions2:
                if 0 < pos2 - pos1 <= k:
                    matched = True
                    break
            if matched:
                break

        if matched:
            answer.append(docID)

    return answer
```

**Optimized version (using sorted positions):**

```python
def positional_intersect_optimized(p1, p2, k):
    answer = []

    # Get common documents
    docs = set(p1.keys()).intersection(p2.keys())

    for docID in docs:
        pos1_list = p1[docID]  # sorted
        pos2_list = p2[docID]  # sorted

        i, j = 0, 0
        matched = False

        while i < len(pos1_list) and j < len(pos2_list):
            if pos2_list[j] - pos1_list[i] == k:
                # Found match!
                matched = True
                break
            elif pos2_list[j] - pos1_list[i] < k:
                j += 1
            else:
                i += 1

        if matched:
            answer.append(docID)

    return answer
```

**Complexity:**

- Worst case: O(x × y) where x, y = number of positions
- Expected: Much better (positions sparsely distributed)

**Phrase Query:** "information retrieval"

1. Get positional postings for "information"
2. Get positional postings for "retrieval"
3. Find docs where "retrieval" at position p+1 after "information" at position p

**Longer Phrases:** "information retrieval systems"

Process as:

- "information retrieval" (positions p, p+1)
- Constrain "systems" at p+2

Or incrementally:

1. Find "information" AND "retrieval" adjacent
2. For those docs, check if "systems" follows

### 5.4 Proximity Queries

**Beyond exact phrases:** Terms within k words

**Example:** "bill" NEAR/5 "gates"

- Find documents where "bill" and "gates" appear within 5 words

**Using Positional Index:**

```python
def proximity_search(term1, term2, k):
    """Find docs where term1 and term2 within k positions"""
    p1 = positional_postings(term1)
    p2 = positional_postings(term2)
    answer = []

    for docID in set(p1.keys()).intersection(p2.keys()):
        pos1 = p1[docID]
        pos2 = p2[docID]

        for p in pos1:
            for q in pos2:
                if abs(q - p) <= k:
                    answer.append(docID)
                    break

    return answer
```

**Applications:**

- Names: "Bill Clinton" (adjacent) vs "Bill and Hillary Clinton" (within 3)
- Concepts: "machine" NEAR "learning"
- Avoid false matches: "Bill Gates" vs "gates bill" (different!)

**Ordered vs Unordered:**

- Ordered: term1 before term2
- Unordered: Either order OK

---

## Part 6: Index Compression

### 6.1 Why Compression?

**Benefits:**

1. **Reduce storage costs**
  - Disk: Cheaper but slower
  - SSD: Faster but more expensive
  - Compression: 4:1 ratio typical
2. **Faster data transfer**
  - Disk → Memory bottleneck
  - Transfer 1 MB compressed < 4 MB uncompressed
  - Decompression in CPU cache (fast!)
3. **Fit more in memory**
  - Cache more postings lists
  - Fewer disk seeks

**What to compress:**

1. **Dictionary** (terms)
2. **Postings lists** (docIDs)

### 6.2 Dictionary Compression

**Naive:** Fixed-width strings (20 bytes per term)

**Problems:**

- Wastes space on short terms ("a", "the")
- Can't store long terms (URLs, compounds)

**Strategy 1: Dictionary-as-String**

Concatenate all terms, store pointers:

```
Terms: aachen, abacus, abbey, able
Storage: "aachenabacusabbeyable"
Pointers: [0, 6, 12, 17]  (each 4 bytes)
```

**Space:**

- Strings: Sum of term lengths
- Pointers: 4 bytes × M (M = vocabulary size)
- Saves ~60% over fixed-width

**Strategy 2: Blocked Storage**

Group terms (e.g., 4 per block), store common prefix once:

```
Block 1:
  Prefix: "ab"
  Suffixes: "acus", "bey", "le"

Block 2:
  Prefix: "aca"
  Suffixes: "demic", "demy"
```

**Strategy 3: Front Coding**

Store prefix length + suffix:

```
aardvark
3ark     (aar + vark)
4wolf    (aard + wolf)
```

### 6.3 Postings Compression (Gap Encoding)

**Key Insight:** PostingsLists are sorted, store gaps instead of absolute IDs

**Example:**

Absolute: `[3, 10, 17, 25, 32, 101, 150, 159]`

Gaps: `[3, 7, 7, 8, 7, 69, 49, 9]`

**Why gaps?**

- Smaller numbers
- More compressible (more zeros in binary)

**Average gap:**

```
average_gap = N / df(term)
```

For frequent terms, gaps are small!

### 6.4 Variable Byte (VB) Encoding

**Idea:** Use variable number of bytes per integer

**Encoding:**

- 7 bits for data
- 1 bit (high bit) as continuation flag
  - 1 = more bytes follow
  - 0 = last byte

**Example:**

Number 5:

```
Binary: 101
VB: 00000101 (1 byte)
```

Number 130:

```
Binary: 10000010
VB: 10000001 00000010 (2 bytes)
     └─ continue  └─ last
```

Number 214577:

```
Binary: 110100011000110001
Split into 7-bit chunks: 0110100 0110001 10001
VB: 10000110 10100011 00010001
    └─more   └─more   └─last
```

**Decoding:**

```python
def vb_decode(bytes):
    n = 0
    for byte in bytes:
        n = (n << 7) | (byte & 0x7F)  # Add 7 data bits
        if (byte & 0x80) == 0:         # Last byte?
            yield n
            n = 0
```

**Properties:**

- Simple, fast
- Byte-aligned (CPU-friendly)
- Wastes some bits (only 7/8 used)
- Good for numbers < 2^21 (3 bytes)

### 6.5 Gamma Codes

**Idea:** Unary length + binary offset

**Encoding:**

1. Find length: L = ⌊log₂(N)⌋
2. Unary encode L: L ones + 1 zero
3. Binary encode N - 2^L (L bits)

**Example:**

Number 13:

```
Binary: 1101
L = floor(log₂(13)) = 3
Offset = 13 - 2^3 = 5 = 101₂

Gamma code:
  1110    (unary for L=3: three 1s + one 0)
  101     (binary for offset 5)
Result: 1110101
```

**More examples:**


| N   | Binary | L   | Unary | Offset | Gamma Code |
| --- | ------ | --- | ----- | ------ | ---------- |
| 1   | 1      | 0   | 0     | -      | 0          |
| 2   | 10     | 1   | 10    | 0      | 100        |
| 3   | 11     | 1   | 10    | 1      | 101        |
| 4   | 100    | 2   | 110   | 00     | 11000      |
| 5   | 101    | 2   | 110   | 01     | 11001      |
| 9   | 1001   | 3   | 1110  | 001    | 1110001    |


**Length of gamma code:**

```
length = 2⌊log₂(N)⌋ + 1
```

**Properties:**

- Prefix-free (self-delimiting)
- Good for small numbers (N < 10)
- Wastes bits for large numbers

**Decoding:**

```python
def gamma_decode(bits):
    # Read unary length
    length = 0
    while read_bit(bits) == 1:
        length += 1

    # Read binary offset
    if length == 0:
        return 1
    offset = read_bits(bits, length)
    return (1 << length) + offset
```

### 6.6 Delta Codes

**Idea:** Improvement over gamma, better for larger numbers

**Encoding:**

1. Find L = ⌊log₂(N)⌋
2. **Gamma encode L** (not unary!)
3. Binary encode N - 2^L (L bits)

**Example:**

Number 13:

```
L = floor(log₂(13)) = 3
Gamma(3) = 101 (from table above)
Offset = 13 - 8 = 5 = 101₂

Delta code: 101 101
```

**Comparison:**


| N   | Gamma       | Delta       |
| --- | ----------- | ----------- |
| 1   | 0           | 0           |
| 2   | 100         | 1000        |
| 5   | 11001       | 10101       |
| 9   | 1110001     | 10001001    |
| 13  | 1110101     | 101101      |
| 100 | 11111000100 | 11001100100 |


**Delta is better for N > 16**

### 6.7 Compression Comparison

**For Reuters-RCV1:**


| Method        | Bits/gap | Compression Ratio |
| ------------- | -------- | ----------------- |
| Uncompressed  | 32       | 1.0               |
| Variable Byte | 8.5      | 3.8               |
| Gamma         | 5.5      | 5.8               |
| Delta         | 5.0      | 6.4               |


**Trade-offs:**


|             | VB     | Gamma   | Delta   |
| ----------- | ------ | ------- | ------- |
| Speed       | Fast   | Slow    | Medium  |
| Compression | Good   | Better  | Best    |
| Simplicity  | Simple | Complex | Complex |


**Best Practice:**

- Use **VB** for general purpose (good balance)
- Use **Gamma/Delta** when storage critical
- Modern: **PForDelta** (advanced, not covered here)

---

## Summary & Key Takeaways

### From Text to Searchable Index

```
Raw Text
   ↓ Tokenization
Tokens
   ↓ Normalization, Stemming
Terms
   ↓ Inverted Index Construction
Dictionary + Postings
   ↓ Compression
Efficient Index
   ↓ Boolean Retrieval
Search Results
```

### Essential Concepts

1. **Tokenization matters:** Language-specific, handle edge cases
2. **Stemming vs Lemmatization:** Speed vs accuracy
3. **Statistical laws:** Zipf (frequency), Heap's (vocabulary growth)
4. **Inverted index:** Core data structure (dictionary + postings)
5. **BSBI vs SPIMI:** Sort all vs incremental, memory trade-offs
6. **Boolean queries:** Fast with sorted postings, O(x+y) intersection
7. **Query optimization:** Process shortest lists first
8. **Skip pointers:** √n spacing, help for large lists
9. **Positional indexes:** Enable phrase queries, 2-4× storage
10. **Compression:** Gap encoding + VB/Gamma/Delta codes

### What's Next?

Module 1 covered **how to build an index** and **exact matching** (Boolean).

**Module 2** will cover **ranking** (which documents are most relevant?):

- TF-IDF weighting
- Vector Space Model
- Cosine similarity
- BM25 (the most important ranking function!)

---

## References & Further Reading

**Manning, Raghavan, Schütze:**

- Chapter 1: Boolean Retrieval (pages 1-18)
- Chapter 2: The Term Vocabulary and Postings Lists (pages 19-45)
- Chapter 4: Index Construction (pages 67-84)
- Chapter 5: Index Compression (pages 85-108)

**Key Papers:**

- Porter, M.F. (1980). "An algorithm for suffix stripping." Program, 14(3), 130-137.
- Zipf, G.K. (1949). Human Behavior and the Principle of Least Effort.
- Dean, J., & Ghemawat, S. (2004). "MapReduce: Simplified data processing on large clusters."

**Tools:**

- NLTK: Natural Language Toolkit (Python)
- spaCy: Industrial-strength NLP
- Lucene/Elasticsearch: Production IR systems

---

## Exercises

See `example.ipynb` for hands-on implementation exercises!

**By hand:**

1. Stem 20 words using Porter algorithm (step-by-step)
2. Calculate Zipf's Law parameters for toy corpus
3. Build inverted index for 5 documents (manually)
4. Perform Boolean query intersection (trace algorithm)
5. Add skip pointers, show intersection saves comparisons
6. Compress postings list with VB and Gamma codes

**Programming:**

1. Implement Porter Stemmer from scratch
2. Build BSBI indexer
3. Build SPIMI indexer, compare performance
4. Implement positional index
5. Phrase query processor
6. Compression algorithms (VB, Gamma, Delta)

**Analysis:**

1. Plot Zipf's Law for your data
2. Verify Heap's Law prediction
3. Measure compression ratios
4. Benchmark: Boolean query speed with/without skip pointers

---

**End of Module 1 Theory**

Next: Work through `example.ipynb` to implement these concepts!