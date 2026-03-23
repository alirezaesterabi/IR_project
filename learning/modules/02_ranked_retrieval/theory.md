# Module 2: Ranked Retrieval — TF-IDF, Vector Space Model, BM25

**Duration:** 7–8 days  
**Manning Chapters:** 6, 11  
**Slides:** Week 2, Week 3, Week 5  
**Lab:** Week 3 (VSM → TF-IDF → BM25), Week 5 (Elasticsearch / BM25 in production)

---

## Table of Contents

### Part 1: From Boolean to Ranked Retrieval
1.1 Why Boolean Retrieval is Not Enough  
1.2 The Ranking Problem  
1.3 Relevance as a Score  

### Part 2: Term Weighting — TF, IDF, TF-IDF
2.1 Term Frequency (TF) and Its Variants  
2.2 Document Frequency and Inverse Document Frequency (IDF)  
2.3 TF-IDF Weighting Schemes (SMART Notation)  
2.4 The IDF Derivation — Where Does log Come From?  

### Part 3: Vector Space Model (VSM)
3.1 Documents and Queries as Vectors  
3.2 Cosine Similarity — Derivation and Geometry  
3.3 Why Not Euclidean Distance?  
3.4 Efficient Cosine Ranking via Inverted Index  
3.5 Limitations of VSM  

### Part 4: Probabilistic Information Retrieval
4.1 The Probability Ranking Principle (PRP)  
4.2 Binary Independence Model (BIM)  
4.3 Estimating Term Weights — Robertson-Sparck Jones (RSJ)  
4.4 From RSJ to BM25: Bridging the Gap  

### Part 5: BM25 — The Most Important Ranking Function
5.1 Full Derivation of BM25  
5.2 TF Saturation — the k₁ Parameter  
5.3 Document Length Normalisation — the b Parameter  
5.4 IDF Component in BM25  
5.5 The Complete BM25 Formula  
5.6 BM25+ — Fixing the Lower Bound Problem  
5.7 Worked Example by Hand  

### Part 6: BM25F — Multi-Field BM25
6.1 Motivation: Fields Have Different Importance  
6.2 Naive Concatenation and Why It Fails  
6.3 BM25F Formula and Field Weights  
6.4 BM25F in Elasticsearch (Week 5 Lab)  

### Part 7: Language Models for IR
7.1 Query Likelihood Approach  
7.2 The Zero-Probability Problem and Smoothing  
7.3 Jelinek-Mercer Smoothing  
7.4 Dirichlet Smoothing  
7.5 Relationship to BM25  

### Part 8: Model Comparison and Practical Guidance
8.1 Summary: When to Use Which Model  
8.2 What BM25 Parameters to Set  
8.3 Connection to the OpenSanctions Project  

---

## Part 1: From Boolean to Ranked Retrieval

### 1.1 Why Boolean Retrieval is Not Enough

Boolean retrieval, as we built in Module 1, has a fundamental problem: it returns a **set**, not a **ranking**.

Consider a compliance analyst searching OpenSanctions for `"Russian sanctions"`:

- Boolean AND returns 847 documents — all equally "relevant" or "not relevant"
- The analyst has no idea which of the 847 to look at first
- A near-miss that omits one term returns **nothing**

**Two failure modes:**

1. **Too many results:** `"sanction"` → 1.2M docs (all of them). No guidance on where to start.
2. **Too few results:** `"Russian oil tanker evading OFAC sanctions 2019"` → 0 docs (exact phrase never appears)

This is called the **feast-or-famine problem** of Boolean retrieval.

**The solution:** compute a real-valued **score** for every document and return them ranked highest first. The analyst gets:
- the most relevant documents at the top
- a graceful degradation (partial matches still appear, lower)

### 1.2 The Ranking Problem

**Formal statement:**

Given:
- A collection $\mathcal{D}$ of $N$ documents
- A query $q$ consisting of terms $t_1, t_2, \ldots, t_k$

Compute a score $s(q, d) \in \mathbb{R}$ for each $d \in \mathcal{D}$, then return the top-$K$ documents sorted by score descending.

The score function must be:
1. **Efficient** — computable in milliseconds, not hours
2. **Meaningful** — higher score = more likely relevant
3. **Comparable** — scores across different queries must be interpretable

### 1.3 Relevance as a Score

We need to quantify relevance without human labels (which would require reading all 1.2M documents per query). The key insight, which every model in this module exploits:

> **A document is likely relevant to a query if it contains the query terms frequently, and those terms are rare in the overall collection.**

This single sentence motivates TF, IDF, and their combination TF-IDF. Everything else in this module is a refinement of this idea.

---

## Part 2: Term Weighting — TF, IDF, TF-IDF

### 2.1 Term Frequency (TF) and Its Variants

**Raw term frequency:** the count of how many times term $t$ appears in document $d$:

$$\text{tf}(t, d) = \text{count of } t \text{ in } d$$

**Problem with raw TF:** a document that mentions "sanctions" 10 times is probably more relevant than one mentioning it once, but is it 10× more relevant? No — relevance is not linear in count.

This is called the **TF saturation problem**: the 10th occurrence of a term adds less new evidence than the 1st.

**Solution variants:**

| Variant | Formula | When to use |
|---------|---------|-------------|
| Raw count | $\text{tf}(t,d)$ | Rarely used directly |
| Binary | $\mathbf{1}[\text{tf}(t,d) > 0]$ | When presence/absence is enough |
| Log-scaled | $1 + \log\text{tf}(t,d)$ if $\text{tf}>0$, else 0 | Most common in VSM |
| Normalized | $\frac{\text{tf}(t,d)}{\max_{t'} \text{tf}(t',d)}$ | When doc length varies greatly |
| Augmented | $0.5 + 0.5 \cdot \frac{\text{tf}(t,d)}{\max_{t'} \text{tf}(t',d)}$ | Long docs, prevents 0 for present terms |
| BM25 saturation | $\frac{\text{tf}(t,d) \cdot (k_1+1)}{\text{tf}(t,d) + k_1}$ | BM25 (see Part 5) |

**Log-scaled TF visualised:**

For raw TF = {0, 1, 2, 5, 10, 100}:
- Raw TF:    {0, 1, 2, 5, 10, 100}
- Log TF:    {0, 1, 1.3, 1.7, 2.0, 3.0}

The log transformation **compresses** the range — the 100th occurrence is worth 3 units, not 100. This matches human intuition: seeing a word 100 times is not 100× more relevant than seeing it once.

**Derivation of log-scaled TF from utility theory:**

Assume the "utility" of the $k$-th occurrence of a term decreases as $1/k$ (diminishing returns):

$$U(\text{tf}) = \sum_{k=1}^{\text{tf}} \frac{1}{k} \approx \ln(\text{tf}) + \gamma \approx \log(\text{tf})$$

The harmonic series approximation gives us the log function directly. This is why log-scaling is theoretically motivated, not arbitrary.

### 2.2 Document Frequency and Inverse Document Frequency (IDF)

**Document frequency** $\text{df}(t)$: the number of documents in the collection containing term $t$.

Observation:

| Term | df | Meaning |
|------|----|---------|
| `the` | 1,240,000 | Appears in nearly every doc → useless |
| `sanction` | 1,180,000 | Very common in our corpus → low info |
| `proliferation` | 12,400 | Moderately rare → medium info |
| `imo9553359` | 1 | Unique → maximum info |

Terms with high $\text{df}$ carry little **discriminative power** — they don't help distinguish relevant from non-relevant documents.

**Inverse Document Frequency:**

$$\text{idf}(t) = \log\frac{N}{\text{df}(t)}$$

where $N$ is the total number of documents.

**Why log?** Without log:

| Term | df | $N/\text{df}$ | $\log(N/\text{df})$ |
|------|----|--------------|-------------------|
| `the` | $N$ | 1 | 0 |
| `proliferation` | $N/100$ | 100 | 2 |
| `imo9553359` | 1 | $N$ | $\log N$ |

The log keeps scores on a human-scale range. Without it, a rare term would dominate by a factor of millions.

**IDF table for our OpenSanctions corpus ($N = 1{,}245{,}931$):**

| Term | df | idf |
|------|----|-----|
| `sanction` | 1,180,000 | $\log(1.06) \approx 0.02$ |
| `vessel` | 98,000 | $\log(12.7) \approx 1.10$ |
| `proliferation` | 12,400 | $\log(100) \approx 2.00$ |
| `imo9553359` | 1 | $\log(1{,}245{,}931) \approx 6.10$ |

**IDF smoothing variants:**

The basic formula has a problem: if $\text{df}(t) = N$ then $\text{idf}(t) = 0$ — the term is completely ignored. Common fixes:

$$\text{idf}_{\text{smooth}}(t) = \log\left(\frac{N}{\text{df}(t) + 1}\right) + 1$$

$$\text{idf}_{\text{max}}(t) = \log\left(\frac{\max_t \text{df}(t)}{\text{df}(t)}\right)$$

$$\text{idf}_{\text{probabilistic}}(t) = \log\frac{N - \text{df}(t)}{\text{df}(t)}$$

The probabilistic variant (Manning eq. 6.11) is what BM25 uses — we'll return to it in Part 5.

### 2.3 TF-IDF Weighting Schemes (SMART Notation)

The **SMART system** (Salton, 1971) introduced a systematic notation for weighting schemes: `ddd.qqq` where the first triple applies to documents and the second to queries. Each triple has three characters: TF variant, DF variant, normalisation variant.

**TF variants (first character):**
- `n` — natural (raw): $\text{tf}(t,d)$
- `l` — logarithm: $1 + \log\text{tf}(t,d)$
- `a` — augmented: $0.5 + 0.5 \cdot \text{tf}/\max\text{tf}$
- `b` — binary: $\mathbf{1}[\text{tf}>0]$

**DF variants (second character):**
- `n` — none: 1
- `t` — idf: $\log(N/\text{df})$
- `p` — prob. idf: $\log\frac{N-\text{df}}{\text{df}}$

**Normalisation (third character):**
- `n` — none
- `c` — cosine: divide by $L_2$ norm of vector
- `u` — pivoted unique normalisation

The most common scheme in practice is **lnc.ltc**:
- Documents: log TF, no IDF, cosine normalised
- Queries: log TF, IDF weighted, cosine normalised

$$w_{t,d}^{\text{lnc}} = \frac{(1 + \log\text{tf}_{t,d})}{\sqrt{\sum_{t'} (1 + \log\text{tf}_{t',d})^2}}$$

$$w_{t,q}^{\text{ltc}} = \frac{(1 + \log\text{tf}_{t,q}) \cdot \log(N/\text{df}_t)}{\sqrt{\sum_{t'} [(1 + \log\text{tf}_{t',q}) \cdot \log(N/\text{df}_{t'})]^2}}$$

### 2.4 The IDF Derivation — Where Does log Come From?

This derivation is Manning Box 6.2 and is worth understanding deeply.

Consider a document collection. A term with $\text{df}(t) = p \cdot N$ (fraction $p$ of all documents) provides:

$$\text{information}(t) = -\log p = \log\frac{1}{p} = \log\frac{N}{\text{df}(t)}$$

This is the **Shannon information content** of observing the term in a document — if a term appears in every document, observing it gives zero bits of information ($\log(1)=0$). If it appears in only 1 document, observing it gives $\log N$ bits.

IDF is exactly the Shannon self-information of term $t$ under the uniform document model. It is not arbitrary — it is the theoretically correct measure of a term's discriminative power.

## Part 3: Vector Space Model (VSM)

### 3.1 Documents and Queries as Vectors

The **Vector Space Model** (Salton, 1975; Manning Chapter 6) represents every document and query as a vector in $\mathbb{R}^{|V|}$, where $|V|$ is the vocabulary size.

Each dimension corresponds to one term. The value at dimension $i$ is the TF-IDF weight of term $i$.

$$\vec{d} = (w_{t_1,d},\ w_{t_2,d},\ \ldots,\ w_{t_{|V|},d}) \in \mathbb{R}^{|V|}$$

$$\vec{q} = (w_{t_1,q},\ w_{t_2,q},\ \ldots,\ w_{t_{|V|},q}) \in \mathbb{R}^{|V|}$$

**For OpenSanctions:**
- $|V| \approx 500{,}000$ terms
- $N = 1{,}245{,}931$ documents
- Matrix would be $500K \times 1.25M = 6.25 \times 10^{11}$ cells — impossible to store explicitly
- In practice: sparse representation (only non-zero weights stored)

### 3.2 Cosine Similarity — Derivation and Geometry

The **cosine similarity** between query $\vec{q}$ and document $\vec{d}$ is:

$$\text{score}(q, d) = \cos(\theta) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}|\ |\vec{d}|}$$

where:

$$\vec{q} \cdot \vec{d} = \sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}$$

$$|\vec{q}| = \sqrt{\sum_{t} w_{t,q}^2}, \quad |\vec{d}| = \sqrt{\sum_{t} w_{t,d}^2}$$

**Geometric interpretation:**

The cosine of the angle between two vectors measures their **directional similarity** — it is 1 when vectors point in the same direction, 0 when orthogonal, −1 when opposite.

Two documents about the same topic will have similar term distributions → similar direction → high cosine.

**Key property:** cosine is **length-invariant**. A document with twice as many words but the same proportional term distribution gets the same score as the shorter one. This is the desired behaviour.

**Worked example (3 documents, 3 terms):**

Terms: `russian`, `sanction`, `vessel`

| | russian | sanction | vessel |
|-|---------|----------|--------|
| E001 (Viktor Petrov) | 2 | 1 | 0 |
| E003 (DONG CHANG) | 0 | 1 | 3 |
| E005 (Black Sea Corp) | 1 | 1 | 0 |
| query: "russian sanction" | 1 | 1 | 0 |

Step 1: IDF weights (toy with $N=3$):
$$\text{idf(russian)} = \log(3/2) = 0.41, \quad \text{idf(sanction)} = \log(3/3) = 0, \quad \text{idf(vessel)} = \log(3/1) = 0.48$$

Step 2: TF-IDF weights for E001: $(2 \times 0.41,\ 1 \times 0,\ 0) = (0.82,\ 0,\ 0)$

Step 3: Query vector (binary TF): $(1 \times 0.41,\ 1 \times 0) = (0.41,\ 0)$

Step 4: Cosine similarity:
$$\cos(\vec{q}, \vec{d}_{\text{E001}}) = \frac{0.41 \times 0.82 + 0 \times 0 + 0 \times 0}{|\vec{q}|\ |\vec{d}|} = \frac{0.336}{0.41 \times 0.82} = 1.0$$

The two vectors are parallel — perfect directional match.

### 3.3 Why Not Euclidean Distance?

An obvious alternative to cosine similarity is **Euclidean distance**:

$$d_{\text{Eucl}}(\vec{q}, \vec{d}) = |\vec{q} - \vec{d}| = \sqrt{\sum_t (w_{t,q} - w_{t,d})^2}$$

**Problem:** Euclidean distance is sensitive to vector length (document length). Consider:

- $\vec{d}_1 = (1, 1, 0, \ldots)$ (short document, 2 relevant terms)
- $\vec{d}_2 = (3, 3, 0, \ldots)$ (long document, same terms × 3)
- $\vec{q} = (1, 1, 0, \ldots)$

Euclidean: $d(\vec{q}, \vec{d}_2) = \sqrt{4+4} = 2.83 > d(\vec{q}, \vec{d}_1) = 0$  
So $\vec{d}_1$ is ranked higher — but $\vec{d}_2$ is likely more relevant (more content about the topic).

Cosine: both score 1.0 — same directional alignment regardless of length.

**Second problem:** Euclidean distance has the "curse of dimensionality" — in high-dimensional spaces, all distances converge to the same value, making them uninformative.

### 3.4 Efficient Cosine Ranking via Inverted Index

Computing cosine similarity naively requires iterating over all $|V|$ dimensions for every document — $O(N \times |V|)$ per query, which is billions of operations.

**Efficient approach:** only compute over terms that appear in both query and document (the non-zero intersection):

$$\vec{q} \cdot \vec{d} = \sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}$$

**Algorithm** (Manning Algorithm 6.14):

```
scores = dict()  # doc_id → accumulated score

for each query term t:
    for each (doc_id, tf_t_d) in INDEX[t].postings:
        scores[doc_id] += w(t, q) × tf_idf(t, doc_id)

for each doc_id in scores:
    scores[doc_id] /= lengths[doc_id]     # divide by |d| for cosine normalisation

return top-K by scores
```

Time complexity: $O(\sum_{t \in q} \text{df}(t))$ — proportional to the sum of query term document frequencies, not $N \times |V|$.

For a query with 5 terms, each with df ≈ 1000, this is 5000 operations — not 600 billion.

### 3.5 Limitations of VSM

1. **Term independence assumption:** Each term is treated independently. "New York" ≠ "New" + "York" in VSM.

2. **No term proximity:** A document with query terms adjacent scores the same as one with them scattered.

3.**Bag-of-words:** Word order is completely ignored.

4. **IDF is a heuristic:** It works empirically but has no principled probabilistic justification within VSM.

5. **TF saturation is not properly handled:** Log-scaling helps but doesn't model diminishing returns optimally.

6. **No document length normalisation that handles very long/short documents well:** Cosine normalisation over-penalises long documents.

These limitations motivate the probabilistic approach — leading to BM25.

---

## Worked Example: TF-IDF + VSM from Scratch

This example builds a complete TF-IDF index, scores one query two different ways, and explains why the results differ — every number shown explicitly.

---

### The Collection (4 documents)

```
D1: "Russian oligarch evading sanctions energy"
D2: "North Korean vessel crude oil sanctions"
D3: "Russian shell company sanctions evasion"
D4: "Iranian vessel suspected arms smuggling"
```

$N = 4$ documents.

---

### Step 1: Tokenize and Build Vocabulary

Lowercase, remove stop words, apply Porter stemming:

```
D1 tokens: russian, oligarch, evad, sanction, energi
D2 tokens: north, korean, vessel, crude, oil, sanction
D3 tokens: russian, shell, compani, sanction, evas
D4 tokens: iranian, vessel, suspect, arms, smuggl
```

*(Porter stemming: `evading→evad`, `evasion→evas`, `company→compani`)*

**Vocabulary:** 17 unique terms.

---

### Step 2: Raw Term Frequency Matrix

Rows = documents, columns = terms (0 or 1 in this small example):

|    | russian | sanction | vessel | crude | oil | oligarch | evad | energi | north | korean | shell | compani | evas | iranian | arms | suspect | smuggl |
|----|---------|----------|--------|-------|-----|----------|------|--------|-------|--------|-------|---------|------|---------|------|---------|--------|
| D1 | **1** | **1** | 0 | 0 | 0 | **1** | **1** | **1** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D2 | 0 | **1** | **1** | **1** | **1** | 0 | 0 | 0 | **1** | **1** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D3 | **1** | **1** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** | **1** | **1** | 0 | 0 | 0 | 0 |
| D4 | 0 | 0 | **1** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** | **1** | **1** | **1** |

---

### Step 3: Document Frequency and IDF

$$\text{df}(t) = \text{number of documents containing } t$$

$$\text{idf}(t) = \log_{10}\frac{N}{\text{df}(t)} = \log_{10}\frac{4}{\text{df}(t)}$$

| Term | df | idf = log₁₀(4/df) |
|------|----|-------------------|
| sanction | 3 | $\log_{10}(4/3) = \mathbf{0.125}$ |
| russian | 2 | $\log_{10}(4/2) = \mathbf{0.301}$ |
| vessel | 2 | $\log_{10}(4/2) = \mathbf{0.301}$ |
| crude | 1 | $\log_{10}(4/1) = \mathbf{0.602}$ |
| oil | 1 | $\log_{10}(4/1) = \mathbf{0.602}$ |
| oligarch | 1 | $\log_{10}(4/1) = \mathbf{0.602}$ |
| iranian | 1 | $\log_{10}(4/1) = \mathbf{0.602}$ |

**Key observation:** `sanction` appears in 3 of 4 docs → idf = 0.125 (weak discriminator).  
`crude` appears in only 1 doc → idf = 0.602 (strong discriminator).

---

### Step 4: TF-IDF Weights

$$w(t, d) = \text{tf}(t, d) \times \text{idf}(t)$$

Since all tf values here are 0 or 1, each weight is simply the idf when the term is present:

|    | russian | sanction | vessel | crude | oil | oligarch | energi | iranian |
|----|---------|----------|--------|-------|-----|----------|--------|---------|
| D1 | **0.301** | **0.125** | 0 | 0 | 0 | **0.602** | **0.602** | 0 |
| D2 | 0 | **0.125** | **0.301** | **0.602** | **0.602** | 0 | 0 | 0 |
| D3 | **0.301** | **0.125** | 0 | 0 | 0 | 0 | 0 | 0 |
| D4 | 0 | 0 | **0.301** | 0 | 0 | 0 | 0 | **0.602** |

Each document is now a **sparse vector** in 17-dimensional space.

---

### Step 5: The Query

**Query:** `"Russian sanctions"`

Tokenised + stemmed: `russian`, `sanction`

Query TF-IDF weights (binary tf = 1 for present terms):

$$w(\text{russian}, q) = 1 \times 0.301 = 0.301 \qquad w(\text{sanction}, q) = 1 \times 0.125 = 0.125$$

All other 15 dimensions = 0.

---

### Step 6a: Method 1 — Sum of TF-IDF Weights

The simplest scoring: add up the TF-IDF weights of the query terms that appear in each document.

$$\text{score}_{\text{sum}}(q, d) = \sum_{t \in q} w(t, d)$$

| Doc | w(russian, d) | w(sanction, d) | Sum score |
|-----|--------------|----------------|-----------|
| D1 | 0.301 | 0.125 | **0.426** |
| D2 | 0 | 0.125 | **0.125** |
| D3 | 0.301 | 0.125 | **0.426** |
| D4 | 0 | 0 | **0.000** |

**Ranked list (Method 1):**

| Rank | Document | Score |
|------|----------|-------|
| **1=** | D1: "Russian oligarch evading sanctions energy" | 0.426 |
| **1=** | D3: "Russian shell company sanctions evasion" | 0.426 |
| **3** | D2: "North Korean vessel crude oil sanctions" | 0.125 |
| **4** | D4: "Iranian vessel suspected arms smuggling" | 0.000 |

**Problem:** D1 and D3 are tied. Method 1 ignores all terms in the document that are *not* in the query, so it cannot distinguish a focused document from a verbose one.

---

### Step 6b: Method 2 — Cosine Similarity (VSM)

To break the tie and correct for document length, we represent both query and documents as full vectors and measure the **angle** between them (Part 3 above).

$$\text{score}_{\text{cos}}(q, d) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}|\ |\vec{d}|}$$

**Query magnitude:**

$$|\vec{q}| = \sqrt{0.301^2 + 0.125^2} = \sqrt{0.0906 + 0.0156} = \sqrt{0.1062} = 0.326$$

**D1** (4 terms: `russian`, `sanction`, `oligarch`, `energi`):

$$\vec{q} \cdot \vec{d_1} = (0.301)(0.301) + (0.125)(0.125) = 0.0906 + 0.0156 = 0.1062$$

$$|\vec{d_1}| = \sqrt{0.301^2 + 0.125^2 + 0.602^2 + 0.602^2} = \sqrt{0.0906+0.0156+0.362+0.362} = 0.911$$

$$\text{score}(q, D1) = \frac{0.1062}{0.326 \times 0.911} = \frac{0.1062}{0.297} = \mathbf{0.357}$$

**D2** (6 terms, only `sanction` from the query):

$$\vec{q} \cdot \vec{d_2} = (0.125)(0.125) = 0.0156$$

$$|\vec{d_2}| = \sqrt{0.125^2 + 0.301^2 + 4 \times 0.602^2} = \sqrt{0.0156+0.0906+1.448} = 1.247$$

$$\text{score}(q, D2) = \frac{0.0156}{0.326 \times 1.247} = \frac{0.0156}{0.406} = \mathbf{0.038}$$

**D3** (5 terms: `russian`, `sanction`, `shell`, `compani`, `evas`):

$$\vec{q} \cdot \vec{d_3} = 0.1062 \quad \text{(same dot product as D1 — same two query terms)}$$

$$|\vec{d_3}| = \sqrt{0.301^2 + 0.125^2 + 3 \times 0.602^2} = \sqrt{0.1062 + 1.086} = 1.092$$

$$\text{score}(q, D3) = \frac{0.1062}{0.326 \times 1.092} = \frac{0.1062}{0.356} = \mathbf{0.298}$$

**D4** (no query terms) → score = **0**.

---

### Result: Side-by-Side Comparison

| Doc | Method 1 (sum) | Rank | Method 2 (cosine) | Rank |
|-----|----------------|------|-------------------|------|
| D1 | 0.426 | **1=** | 0.357 | **1** |
| D3 | 0.426 | **1=** | 0.298 | **2** |
| D2 | 0.125 | **3** | 0.038 | **3** |
| D4 | 0.000 | **4** | 0.000 | **4** |

---

### Why Did Cosine Break the Tie?

Both D1 and D3 contain the **same two query terms with the same weights** → identical dot products (0.1062).

The difference is in the **denominator** $|\vec{d}|$, which accounts for **all** terms in the document:

- D1 has 4 terms → $|\vec{d_1}| = 0.911$ (shorter vector)
- D3 has 5 terms → $|\vec{d_3}| = 1.092$ (longer vector, pulled further from query direction)

D3's extra non-query term (`evas`) adds mass to the vector, reducing the cosine angle with the query.  
**Cosine rewards focused, on-topic documents over verbose ones.**

---

### The Two Scoring Methods at a Glance

```
Steps 1–5 are identical for both methods:

1. Tokenize + stem all documents              → token lists per doc
2. Build vocabulary (dictionary)              → sorted unique terms
3. Compute tf(t, d) = raw count              → term-frequency matrix
4. Compute idf(t) = log(N / df(t))           → one value per term
5. Compute w(t, d) = tf × idf               → TF-IDF weight matrix

Then choose a scoring method:

6a. score_sum(q, d)  = Σ w(t,d) for t ∈ q   → simple, fast, length-biased (ties possible)
6b. cosine_sim(q, d) = (q⃗ · d⃗) / (|q⃗| |d⃗|) → length-normalised, breaks ties (standard VSM)
```

---

## Part 4: Probabilistic Information Retrieval

### 4.1 The Probability Ranking Principle (PRP)

**Robertson (1977) showed** that the optimal retrieval strategy is to rank documents in decreasing order of their probability of relevance:

$$P(R = 1 \mid d, q)$$

where $R$ is a relevance random variable. This is the **Probability Ranking Principle (PRP)**.

Using Bayes' theorem:

$$P(R = 1 \mid d, q) = \frac{P(d, q \mid R = 1) \cdot P(R = 1)}{P(d, q)}$$

Since $P(d, q)$ is constant across documents, ranking by $P(R=1 \mid d,q)$ is equivalent to ranking by the **Likelihood Ratio**:

$$\frac{P(d, q \mid R = 1)}{P(d, q \mid R = 0)}$$

This ratio is what the **Binary Independence Model** attempts to estimate.

### 4.2 Binary Independence Model (BIM)

The **Binary Independence Model** (Manning §11.3) makes two simplifying assumptions:

1. **Binary representation:** documents are represented as binary vectors $\vec{x} \in \{0,1\}^{|V|}$ — only presence/absence of each term, not counts.

2. **Independence:** terms are conditionally independent of each other given relevance status.

Under these assumptions, the likelihood ratio becomes:

$$\text{RSV}(d, q) = \sum_{t \in q \cap d} \log \frac{p_t(1 - u_t)}{u_t(1 - p_t)}$$

where:
- $p_t = P(x_t = 1 \mid R = 1)$ — probability term $t$ appears in a **relevant** document
- $u_t = P(x_t = 1 \mid R = 0)$ — probability term $t$ appears in a **non-relevant** document

This is called the **Retrieval Status Value (RSV)** — the fundamental score in probabilistic IR.

### 4.3 Estimating Term Weights — Robertson-Sparck Jones (RSJ)

The challenge is estimating $p_t$ and $u_t$ without explicit relevance labels.

**Without relevance feedback** (the common case):

Assume most documents are non-relevant. Then $u_t \approx \text{df}_t / N$, and in the absence of relevance information, assume $p_t = 0.5$.

Substituting:

$$\text{RSV}_t = \log \frac{0.5 \cdot (1 - \text{df}_t/N)}{(\text{df}_t/N) \cdot 0.5} = \log \frac{N - \text{df}_t}{\text{df}_t}$$

This is the **Robertson-Sparck Jones (RSJ) weight** — and it's exactly the **probabilistic IDF** we mentioned in Part 2.

With smoothing to avoid division by zero:

$$\text{RSV}_t = \log \frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5}$$

**This is the IDF component of BM25.** The probabilistic foundation motivates the formula that was previously introduced as a heuristic.

### 4.4 From RSJ to BM25: Bridging the Gap

The BIM+RSJ model has two remaining weaknesses:

1. **Binary representation:** It ignores how many times a term appears (tf). A document mentioning "sanctions" 10 times scores the same as one mentioning it once.

2. **No document length normalisation:** Long documents are unfairly advantaged (more chance of containing query terms).

BM25 addresses both. It is the result of empirical experimentation at TREC (Text REtrieval Conference) in the early 1990s by Robertson, Walker, and colleagues.

---

## Part 5: BM25 — The Most Important Ranking Function

BM25 (**Best Match 25**, Robertson & Walker 1994; Robertson & Zaragoza 2009) is the state-of-the-art for **sparse retrieval** and the standard baseline in all IR research.

### 5.1 Full Derivation of BM25

Starting from the BIM RSV:

$$\text{RSV}(d, q) = \sum_{t \in q \cap d} \log \frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5}$$

**Problem 1: Only binary term presence.** We want to incorporate term counts.

Extend the binary model: instead of $p_t = P(x_t=1 \mid R)$, model $P(TF_t = tf \mid R)$ using a two-component mixture:

- Component 1: Poisson distribution with mean $\mu_{\text{rel}}$ (relevant docs)
- Component 2: Poisson with mean $\mu_{\text{nonrel}}$ (non-relevant docs)

The **log-likelihood ratio** for term counts becomes:

$$\text{contribution}(t, tf) \approx \frac{tf \cdot (\mu_{\text{rel}} - \mu_{\text{nonrel}})}{tf + K}$$

where $K$ is related to the ratio of Poisson means. This gives the TF saturation function — the 2-Poisson model.

**Problem 2: Document length affects tf.** A long document naturally contains more occurrences of any term. We need to normalise.

Let $L_d = $ document length, $\bar{L} = $ average document length across the corpus. Define a **normalised term frequency**:

$$\text{tf}_{\text{norm}} = \frac{tf_{t,d}}{(1 - b) + b \cdot \frac{L_d}{\bar{L}}}$$

where $b \in [0, 1]$ controls the degree of normalisation.

Substituting $\text{tf}_{\text{norm}}$ for $tf$ in the 2-Poisson contribution and defining $k_1$ as the saturation parameter:

$$\text{BM25\_TF}(t, d) = \frac{\text{tf}_{t,d} \cdot (k_1 + 1)}{\text{tf}_{t,d} + k_1 \cdot \left((1-b) + b \cdot \frac{L_d}{\bar{L}}\right)}$$

Combining with the RSJ weight:

$$\boxed{\text{BM25}(d, q) = \sum_{t \in q \cap d} \underbrace{\log\frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5}}_{\text{IDF}} \times \underbrace{\frac{\text{tf}_{t,d} \cdot (k_1+1)}{\text{tf}_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{L_d}{\bar{L}}\right)}}_{\text{normalised TF with saturation}}}$$

This is the **complete BM25 formula** (Manning eq. 11.32).

### 5.2 TF Saturation — the k₁ Parameter

The normalised TF factor is:

$$\text{BM25\_TF} = \frac{\text{tf} \cdot (k_1 + 1)}{\text{tf} + k_1 \cdot \text{norm\_factor}}$$

**Behaviour analysis** (holding document length fixed, so norm\_factor = 1):

$$\lim_{\text{tf} \to 0} \text{BM25\_TF} = 0$$
$$\lim_{\text{tf} \to \infty} \text{BM25\_TF} = k_1 + 1$$

The function is **bounded above by** $k_1 + 1$. This is the saturation ceiling.

| $k_1$ | At tf=1 | At tf=5 | At tf=10 | Ceiling |
|--------|---------|---------|---------|---------|
| 0.0 | 1.0 | 1.0 | 1.0 | 1.0 (binary) |
| 0.5 | 1.33 | 1.45 | 1.48 | 1.5 |
| 1.2 | 1.18 | 1.75 | 1.89 | 2.2 |
| 2.0 | 1.0 | 1.67 | 1.91 | 3.0 |
| ∞ | 1.0 | 5.0 | 10.0 | ∞ (linear) |

**Extreme values:**
- $k_1 = 0$: completely binary — tf is ignored, BM25 reduces to the RSJ model
- $k_1 \to \infty$: linear TF — no saturation, equivalent to raw TF weighting
- $k_1 \in [1.2, 2.0]$: typical range used in practice (TREC experiments)

**Intuition:** The 10th occurrence of "sanctions" is still evidence of relevance, but far less than the 1st. $k_1$ controls exactly how quickly the marginal value of additional occurrences decays.

### 5.3 Document Length Normalisation — the b Parameter

The length normalisation factor in the denominator is:

$$\text{norm} = (1 - b) + b \cdot \frac{L_d}{\bar{L}}$$

**Behaviour analysis:**

- **$b = 0$:** norm = 1 for all documents → no length normalisation → long documents advantaged
- **$b = 1$:** norm = $L_d/\bar{L}$ → full proportional normalisation → scores proportional to term density
- **$b = 0.75$:** typical value (TREC experiments)

**Why not always use $b=1$ (full normalisation)?**

Consider two documents:
- **Short:** 50 words, mentions "vessel" once → $\text{tf/L} = 0.02$ (dense use)
- **Long:** 500 words, mentions "vessel" 8 times → $\text{tf/L} = 0.016$ (slightly sparser)

Full normalisation ($b=1$) would rank the short document higher. But the long document contains far more evidence about vessels — it just also covers other topics. Partial normalisation ($b=0.75$) correctly balances this.

**Two reasons a document might be long:**
1. **Verbose style:** Repeats content many times — should be penalised
2. **More content:** Genuinely covers more topics — should not be penalised

The parameter $b$ lets the practitioner calibrate between these two extremes.

### 5.4 IDF Component in BM25

The IDF in BM25 uses the probabilistic variant:

$$\text{IDF}_{\text{BM25}}(t) = \log\frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5}$$

**Comparison with classical IDF:**

$$\text{IDF}_{\text{classical}} = \log\frac{N}{\text{df}_t}$$

The 0.5 additive smoothing prevents division by zero and dampens extreme values. The term $N - \text{df}_t$ in the numerator reflects the probabilistic origin: it estimates the number of non-relevant documents (recall $u_t \approx \text{df}_t/N$ from the BIM).

**Important:** When $\text{df}_t > N/2$ (term appears in more than half of documents), $\text{IDF}_{\text{BM25}}$ becomes **negative**. This means the term is so common it is actually evidence **against** relevance. In practice, implementations often floor IDF at 0 to avoid subtracting from the score.

### 5.5 The Complete BM25 Formula

Bringing together all components:

$$\text{BM25}(d, q) = \sum_{t \in q \cap d} \log\frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5} \cdot \frac{\text{tf}_{t,d} \cdot (k_1+1)}{\text{tf}_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{L_d}{\bar{L}}\right)}$$

**Optional query term frequency weighting:**

If the query itself contains repeated terms (rare in practice but included in some formulations):

$$\text{BM25}(d, q) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \frac{\text{tf}_{t,d} \cdot (k_1+1)}{\text{tf}_{t,d} + k_1(\ldots)} \cdot \frac{\text{tf}_{t,q} \cdot (k_3+1)}{\text{tf}_{t,q} + k_3}$$

where $k_3 \in [0, 1000]$ controls query TF saturation. Setting $k_3 = 0$ gives binary query term matching (standard usage).

**Default parameters (from TREC experiments):**
- $k_1 = 1.2$ to $2.0$
- $b = 0.75$
- $k_3 = 0$ (binary query term treatment)

### 5.6 BM25+ — Fixing the Lower Bound Problem

Standard BM25 has a subtle flaw identified by Lv and Zhai (2011):

A term that appears in a document gets **score 0** when $b=1$ and the document is infinitely long. This means an extremely long relevant document that contains the query term might score lower than an irrelevant document that doesn't contain it at all.

**BM25+ adds a small additive term $\delta$ to prevent this:**

$$\text{BM25+}(d, q) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \left(\delta + \frac{\text{tf}_{t,d} \cdot (k_1+1)}{\text{tf}_{t,d} + k_1 \cdot \text{norm}}\right)$$

Typical value: $\delta = 1.0$. This ensures every matching term always contributes a positive score.

### 5.7 Worked Example by Hand

**Collection (3 documents):**
- D1: "Russian oligarch evading OFAC sanctions linked to energy"  ($L=9$)
- D2: "North Korean vessel transporting crude oil sanctions violation"  ($L=8$)
- D3: "Russian shell company sanctions evasion money laundering"  ($L=7$)

$N=3$, $\bar{L} = 8$

**Query:** `"russian sanctions"`

**Step 1: Document frequencies**
- $\text{df}(\text{russian}) = 2$ (D1, D3)
- $\text{df}(\text{sanction}) = 3$ (D1, D2, D3)

**Step 2: IDF (BM25 variant)**

$$\text{IDF}(\text{russian}) = \log\frac{3 - 2 + 0.5}{2 + 0.5} = \log\frac{1.5}{2.5} = \log(0.6) = -0.22$$

$$\text{IDF}(\text{sanction}) = \log\frac{3 - 3 + 0.5}{3 + 0.5} = \log\frac{0.5}{3.5} = \log(0.143) = -0.85$$

Both negative — both terms appear in ≥ half the documents. Flooring at 0: both get IDF = 0.

*Note: in a larger collection the IDFs would be positive and meaningful. This toy example illustrates the small-N behaviour.*

**Step 3: BM25 TF ($k_1 = 1.2$, $b = 0.75$)**

For D1 and term "russian" ($\text{tf}=1$, $L=9$):

$$\text{norm} = 1 - 0.75 + 0.75 \times \frac{9}{8} = 0.25 + 0.844 = 1.094$$

$$\text{BM25\_TF} = \frac{1 \times (1.2 + 1)}{1 + 1.2 \times 1.094} = \frac{2.2}{1 + 1.313} = \frac{2.2}{2.313} = 0.951$$

For D3 and term "russian" ($\text{tf}=1$, $L=7$):

$$\text{norm} = 0.25 + 0.75 \times \frac{7}{8} = 0.25 + 0.656 = 0.906$$

$$\text{BM25\_TF} = \frac{2.2}{1 + 1.2 \times 0.906} = \frac{2.2}{2.087} = 1.054$$

D3 scores slightly higher for "russian" because it is shorter, meaning "russian" is relatively more prominent.

---

## Part 6: BM25F — Multi-Field BM25

### 6.1 Motivation: Fields Have Different Importance

Real documents have structure. For our OpenSanctions entities:

- `name` field — the primary identity of the entity
- `alias` field — alternative names
- `notes` field — description of activity
- `country` field — short metadata

A match in the `name` field is far more significant than a match in the `notes` field. How do we express this in BM25?

### 6.2 Naive Concatenation and Why It Fails

The simplest approach: concatenate all fields into one text blob and run standard BM25.

**Problem:** field lengths differ dramatically.

- `name`: 2–5 tokens
- `notes`: 5–50 tokens

If we concatenate, the `name` field becomes a tiny fraction of the total document. A match in `name` gets the same weight as a match in the much longer `notes`, which is wrong.

Additionally: a 500-token `notes` field with 1 mention of "sanctions" gets the same raw TF as a 10-token `name` with 1 mention. The BM25 length normalisation now normalises over the concatenated length, distorting both.

### 6.3 BM25F Formula and Field Weights

**BM25F** (Robertson et al., 2004) normalises each field **independently**, then combines:

$$\text{BM25F}(d, q) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \frac{\tilde{tf}_{t,d} \cdot (k_1 + 1)}{\tilde{tf}_{t,d} + k_1}$$

where the **pseudo-TF** $\tilde{tf}$ is the weighted sum of per-field normalised TFs:

$$\tilde{tf}_{t,d} = \sum_{f \in \text{fields}} w_f \cdot \frac{\text{tf}_{t,d,f}}{(1 - b_f) + b_f \cdot \frac{L_{d,f}}{\bar{L}_f}}$$

Parameters:
- $w_f$: weight of field $f$ (tunable: e.g. $w_{\text{name}} = 5$, $w_{\text{alias}} = 3$, $w_{\text{notes}} = 1$)
- $b_f$: length normalisation for field $f$ (can differ per field)
- $\bar{L}_f$: average length of field $f$ across all documents

**Key difference from concatenation:** BM25F normalises each field relative to the average length of **that specific field**, not the total document length. This correctly handles the mismatch between short name fields and long notes fields.

### 6.4 BM25F in Elasticsearch

Elasticsearch (Week 5 Lab) implements BM25F natively. The `request_body` in the lab configures:
- `similarity.type = BM25`
- `k1`, `b` parameters per index

Multiple fields are queried via `multi_match` with field boosting:

```json
{
  "query": {
    "multi_match": {
      "query": "Russian sanctions",
      "fields": ["name^5", "alias^3", "notes^1"],
      "type": "cross_fields"
    }
  }
}
```

The `^5` notation sets the field weight $w_f = 5$. The `cross_fields` type implements BM25F-style normalisation across fields.

---

## Part 7: Language Models for IR

A different probabilistic approach: instead of modelling relevance directly, model the **probability that the query was generated by the document's language model**.

### 7.1 Query Likelihood Approach

**Idea:** each document $d$ defines a probability distribution $P(\cdot \mid \theta_d)$ over terms (a **language model**). Score a document by:

$$\text{score}(d, q) = P(q \mid \theta_d) = \prod_{t \in q} P(t \mid \theta_d)^{\text{tf}_{t,q}}$$

Using the maximum likelihood estimate:

$$P(t \mid \theta_d) = \frac{\text{tf}_{t,d}}{L_d}$$

In log space:

$$\log P(q \mid \theta_d) = \sum_{t \in q} \text{tf}_{t,q} \cdot \log\frac{\text{tf}_{t,d}}{L_d}$$

**Problem:** if any query term does not appear in $d$, then $P(t \mid \theta_d) = 0$ and the entire product is zero. This is the **zero-probability problem**.

### 7.2 The Zero-Probability Problem and Smoothing

The zero-probability problem is solved by **smoothing** — mixing the document model with a background corpus model:

$$P_{\text{smooth}}(t \mid d) = (1-\lambda) \cdot \frac{\text{tf}_{t,d}}{L_d} + \lambda \cdot P(t \mid \mathcal{C})$$

where $P(t \mid \mathcal{C}) = \frac{\text{cf}_t}{\sum_{t'} \text{cf}_{t'}}$ is the collection language model (term relative frequency across all documents).

This guarantees $P(t \mid d) > 0$ for all $t$.

### 7.3 Jelinek-Mercer Smoothing

**Jelinek-Mercer (JM) smoothing** mixes with a fixed coefficient $\lambda$:

$$P_{\text{JM}}(t \mid d) = (1-\lambda) \cdot \frac{\text{tf}_{t,d}}{L_d} + \lambda \cdot P(t \mid \mathcal{C})$$

Substituting into the log-likelihood score:

$$\log P(q \mid d) = \sum_{t \in q} \log\left[(1-\lambda) \cdot \frac{\text{tf}_{t,d}}{L_d} + \lambda \cdot P(t \mid \mathcal{C})\right]$$

Typical value: $\lambda \in [0.1, 0.5]$. Shorter queries benefit from higher $\lambda$ (more smoothing); longer queries need less.

### 7.4 Dirichlet Smoothing

**Dirichlet smoothing** uses a Bayesian prior with parameter $\mu$:

$$P_{\text{Dir}}(t \mid d) = \frac{\text{tf}_{t,d} + \mu \cdot P(t \mid \mathcal{C})}{L_d + \mu}$$

This is equivalent to adding $\mu$ "pseudo-documents" from the collection language model to every document. The smoothing amount $\mu / (L_d + \mu)$ is larger for short documents (high uncertainty) and smaller for long ones.

Expanding the log-likelihood:

$$\log P(q \mid d) = \sum_{t \in q \cap d} \log\frac{\text{tf}_{t,d} + \mu \cdot P(t \mid \mathcal{C})}{L_d + \mu} + n_q \cdot \log\frac{\mu}{L_d + \mu} + \sum_{t \in q} \log P(t \mid \mathcal{C})$$

The last two terms are query-independent constants. The first term determines the ranking.

Typical value: $\mu \in [500, 2000]$.

### 7.5 Relationship to BM25

Zhai & Lafferty (2004) showed that the log of Dirichlet smoothing can be approximated as:

$$\log P_{\text{Dir}}(t \mid d) \approx \frac{\text{tf}_{t,d}}{\text{tf}_{t,d} + \mu} \cdot \log\frac{\text{tf}_{t,d}}{L_d \cdot P(t \mid \mathcal{C})}$$

The factor $\frac{\text{tf}_{t,d}}{\text{tf}_{t,d} + \mu}$ is structurally identical to the BM25 TF saturation term $\frac{\text{tf}}{tf + k_1(\ldots)}$ when length normalisation is set to $b=0$.

**Insight:** BM25 and language models with Dirichlet smoothing are mathematically related. BM25 can be viewed as an approximation to the Dirichlet-smoothed query likelihood model, with the advantage that it has tunable parameters.

---

## Part 8: Model Comparison and Practical Guidance

### 8.1 Summary: When to Use Which Model

| Model | Score | Ranked? | TF? | Length norm? | When to use |
|-------|-------|---------|-----|--------------|-------------|
| Boolean | {0,1} set | No | No | No | Exact compliance filtering |
| VSM + TF-IDF | cosine ∈ [0,1] | Yes | Log | Cosine | General text search, simple |
| BIM / RSJ | log-likelihood | Yes | No | No | When only presence/absence matters |
| BM25 | real-valued | Yes | Saturating | Partial | **Production default** |
| BM25+ | real-valued | Yes | Saturating+δ | Partial | Long documents present |
| BM25F | real-valued | Yes | Saturating | Per-field | Structured/multi-field docs |
| Language Model (JM) | log P | Yes | Full | Implicit | Short queries |
| Language Model (Dir) | log P | Yes | Full | Implicit | Varied length docs |

**For the OpenSanctions project:** BM25 for the classical IR component (Phase 5), BM25F as an enhancement if multi-field scoring improves results.

### 8.2 What BM25 Parameters to Set

**$k_1$ (TF saturation):**
- Start with $k_1 = 1.2$
- If your documents are short (like our OpenSanctions entities), decrease toward $k_1 = 0.5$
- If documents are long and TF variation is meaningful, increase toward $k_1 = 2.0$

**$b$ (length normalisation):**
- Start with $b = 0.75$
- If all documents are similar length, decrease toward $b = 0$ (normalisation doesn't help)
- If document lengths vary wildly, increase toward $b = 1$

**For OpenSanctions specifically:**
- Entities have highly variable text_blob lengths (vessel records are very short; company records with sanctions history can be long)
- Recommended starting point: $k_1 = 1.5$, $b = 0.75$
- For Phase 5 experiment: tune $k_1 \in \{0.5, 1.0, 1.5, 2.0\}$ and $b \in \{0.5, 0.75, 1.0\}$

### 8.3 Connection to the OpenSanctions Project

| Query Type | Best Model | Why |
|------------|-----------|-----|
| Type 1 (Exact Identifier) | Boolean / exact match | Identifier must match exactly |
| Type 2 (Name/Alias) | BM25 or BM25F | Name field weighted higher |
| Type 3 (Semantic) | BM25 on notes field + dense retrieval | Free-text description matching |
| Type 5 (Cross-dataset) | BM25 + deduplication post-processing | Find all variants of same entity |
| Type 6 (Jurisdiction/Filter) | Boolean pre-filter + BM25 re-rank | programId/country as hard filter |
| Type 7 (RAG) | Dense retrieval → LLM generation | Need semantic understanding |

**Key insight for Phase 5:** BM25 handles Types 2, 3, 5, 6 well. Type 1 uses exact index lookup (not ranked retrieval at all). Types 3 and 7 will benefit from the dense retrieval module (Module 4) added to BM25 as a hybrid system.

---

## References and Further Reading

**Primary (Manning et al.):**
- Chapter 6, §6.1–6.4: TF-IDF and Vector Space Model
- Chapter 11, §11.1–11.4: Probabilistic IR, BIM, BM25
- Chapter 12: Language Models (optional)

**Key Papers:**
- Robertson, S.E. & Sparck Jones, K. (1976). Relevance weighting of search terms. *Journal of the American Society for Information Science.*
- Robertson, S.E. & Walker, S. (1994). Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval. *SIGIR.*
- Robertson, S.E. & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in IR.*
- Lv, Y. & Zhai, C. (2011). Lower-bounding term frequency normalization. *CIKM.* (BM25+)
- Robertson, S.E. et al. (2004). Simple BM25 extension to multiple weighted fields. *CIKM.* (BM25F)
- Zhai, C. & Lafferty, J. (2004). A study of smoothing methods for language models. *ACM TOIS.*

**Tools:**
- `rank-bm25` (Python): lightweight BM25 implementation
- Elasticsearch: production BM25/BM25F with full field control (Week 5 Lab)
- scikit-learn `TfidfVectorizer`: VSM with SMART-style weighting

---

## Exercises

See `01_vsm_tfidf.ipynb` and `02_bm25.ipynb` for hands-on implementation.

**Conceptual (do by hand before coding):**

1. Given 3 documents and query "vessel sanctions", compute TF-IDF for all terms and rank the documents using cosine similarity. Show every arithmetic step.

2. For the same 3 documents, compute BM25 with $k_1=1.2$, $b=0.75$. Compare the ranking to TF-IDF. Which is different? Why?

3. Draw the TF saturation curve $f(\text{tf}) = \frac{\text{tf}(k_1+1)}{\text{tf}+k_1}$ for $k_1 \in \{0, 0.5, 1.2, 2.0, \infty\}$. What is the asymptote?

4. Explain in one sentence why cosine similarity is preferred over Euclidean distance in the VSM.

5. In BM25 IDF, what happens when $\text{df}_t > N/2$? Show the calculation. Is this desirable?

6. Derive the relationship between Dirichlet smoothing ($\mu=1000$) and BM25 ($k_1=1.2$, $b=0$) on a document of length 1000 tokens. At what TF value are the two models equivalent?

7. For BM25F with `name^5` and `notes^1`, compute the pseudo-TF for the term "russian" in a document where: `name`=["Russian Federation"] (tf=1, len=2), `notes`=["Russian oligarch linked to energy sector"] (tf=1, len=7). Average lengths: $\bar{L}_{\text{name}}=3$, $\bar{L}_{\text{notes}}=15$. Use $b=0.75$ for both fields.

**Deeper questions:**

8. BM25 was derived empirically at TREC in the 1990s. What does it mean that TREC experiments converged on $k_1 \in [1.2, 2.0]$ and $b=0.75$? Are these universal or collection-specific?

9. VSM treats all terms as independent. Give a concrete example from OpenSanctions where this assumption fails, and explain how it would affect ranking.

10. Implement BM25 from scratch (no libraries). Verify your implementation against `rank-bm25` on the toy corpus.

---

**End of Module 2 Theory**

Next: `01_vsm_tfidf.ipynb` → `02_bm25.ipynb`
