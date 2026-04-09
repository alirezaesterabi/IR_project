# Evaluation of Information Retrieval Systems

## Part 1: The Evaluation Problem

### 1.1 Why Evaluation is Essential

You have built a retrieval system. You have implemented Boolean retrieval, TF-IDF, BM25. You can return a ranked list of documents for any query. But now the critical question arises:

> **Is your system any good?**

Without evaluation, this question has no answer. You cannot:

- Compare system A vs system B
- Know whether a code change improved or degraded performance
- Understand trade-offs (precision vs recall)
- Make evidence-based decisions about parameters ($k_1$, $b$ in BM25)
- Report quantitative results to stakeholders

**Information Retrieval is an empirical discipline.** Every claim about system performance must be backed by rigorous measurement on representative test collections.

### 1.2 What Makes IR Evaluation Hard

Unlike sorting (where correctness is binary) or classification (where labels are objective), IR evaluation is fundamentally **subjective**:

**Relevance is not objective.** Given query `"python programming"` and document `"Guide to Monty Python films"`:
- A film enthusiast: relevant
- A programmer: not relevant
- The system: cannot know without context

**Relevance is situational.** The same document may be:
- Relevant to a student learning Python (basic tutorial needed)
- Not relevant to an expert (too elementary)
- Relevant to a teacher (good teaching material)

**Relevance is multidimensional:**
- Topical relevance: does it discuss the query topic?
- Novelty: does it provide new information?
- Credibility: is the source trustworthy?
- Understandability: is it accessible to the user?

Despite this subjectivity, **we need repeatable, comparable measurements**. The solution: **test collections with human relevance judgements**.

---

## Part 2: Test Collections and Relevance Judgements

### 2.1 Components of a Test Collection

To evaluate an IR system in the standard way, we need three components:

1. **A document collection** — the corpus to search (e.g., 1.2M OpenSanctions entities, 800K news articles, 50M web pages)

2. **A set of information needs** — expressible as queries (e.g., `"Russian sanctions"`, `"vessel IMO9553359"`)

3. **Relevance judgements (qrels)** — for each query, which documents are relevant?

**The qrels are the gold standard.** They define ground truth. A document is relevant **not** because it contains query terms, but because a human assessor judged it relevant to the information need.

### 2.2 Information Need vs Query

A critical distinction:

**Information need:** "I want to find sanctioned Russian oligarchs with links to the energy sector who are under OFAC sanctions specifically"

**Query:** `"Russian oligarch energy OFAC"`

The query is a **proxy** for the information need. It is rarely perfect. Relevance is judged against the **information need**, not against literal query term matching.

**Example from Manning Chapter 8:**

Information need: *"Information on whether drinking red wine is more effective at reducing the risk of heart attacks than white wine"*

Query: `wine AND red AND white AND heart AND attack`

A document titled *"The health benefits of white wine"* contains all query terms but does **not** address the information need (comparative effectiveness). It would be judged **non-relevant** despite matching the query.

### 2.3 Binary vs Graded Relevance

**Binary relevance:** each document is either relevant (1) or non-relevant (0).

| Query | Doc ID | Judgement |
|-------|--------|-----------|
| Q1    | D042   | 1         |
| Q1    | D103   | 0         |
| Q1    | D871   | 1         |

**Graded relevance:** documents have degrees of relevance (e.g., 0 = not relevant, 1 = marginally relevant, 2 = relevant, 3 = highly relevant).

| Query | Doc ID | Grade |
|-------|--------|-------|
| Q1    | D042   | 3     |
| Q1    | D103   | 0     |
| Q1    | D871   | 1     |

Graded relevance enables finer-grained metrics (nDCG, discussed in Part 7). Binary relevance is simpler and used for precision, recall, MAP.

### 2.4 Qrels Format and Incompleteness

**TREC qrels format** (space-separated):

```
query_id  0  doc_id  relevance
Q1        0  D042    1
Q1        0  D103    0
Q1        0  D871    1
Q2        0  D042    0
```

The `0` is a legacy field (originally iteration number) and is ignored.

**JSON format (common in projects):**

```json
{
  "Q1": {
    "D042": 1,
    "D103": 0,
    "D871": 1
  },
  "Q2": {
    "D042": 0
  }
}
```

**Critical limitation:** qrels are **incomplete**. For a collection of 1 million documents and 50 queries, we have 50 million possible (query, document) pairs. Judging all is infeasible.

**Solution:** **pooling** (see §8). Only a subset of documents (typically top-$k$ from multiple systems) are judged. Unjudged documents are **assumed non-relevant** (with known biases — §8.3).

---

## Part 3: Unranked Retrieval Evaluation — Precision, Recall, F-Measure

### 3.1 The Contingency Table

Suppose a system retrieves a **set** of documents (unranked) for a query. We can categorize all documents in the collection:

|                      | **Relevant** | **Non-relevant** |
|----------------------|--------------|------------------|
| **Retrieved**        | TP           | FP               |
| **Not retrieved**    | FN           | TN               |

- **TP (true positives):** retrieved AND relevant — correct hits
- **FP (false positives):** retrieved but NOT relevant — junk returned
- **FN (false negatives):** NOT retrieved but relevant — misses
- **TN (true negatives):** NOT retrieved and NOT relevant — correctly ignored (usually very large)

### 3.2 Precision

**Precision** answers: *"Of the documents I returned, what fraction are relevant?"*

$$
\text{Precision} = \frac{TP}{TP + FP} = P(\text{relevant} \mid \text{retrieved})
$$

**Interpretation:** A measure of **exactness** or **purity**. High precision means few false positives — users see mostly relevant results.

**Example:** System retrieves 10 documents, 7 are relevant.
$$P = \frac{7}{10} = 0.7$$

### 3.3 Recall

**Recall** answers: *"Of all the relevant documents that exist, what fraction did I return?"*

$$
\text{Recall} = \frac{TP}{TP + FN} = P(\text{retrieved} \mid \text{relevant})
$$

**Interpretation:** A measure of **completeness** or **coverage**. High recall means few misses — users don't miss relevant information.

**Example:** 15 relevant documents exist in the collection, system retrieves 7 of them.
$$R = \frac{7}{15} \approx 0.467$$

### 3.4 The Precision-Recall Trade-off

Precision and recall trade off against each other:

**Trivial high precision:** retrieve only the single document you are most confident about. Precision can be 100%, but recall is abysmal.

**Trivial high recall:** retrieve **all** documents in the collection. Recall is 100%, but precision is terrible (nearly all documents are non-relevant in typical collections).

**The goal:** maximize both simultaneously — but in practice, you must balance them.

**A concrete analogy (OpenSanctions screening):**

- **High precision system:** flags only obvious matches → few false alarms, but misses subtle cases (low recall) → **regulatory risk**
- **High recall system:** flags every possible match → no misses, but analysts drown in false positives (low precision) → **operational cost**

The right balance depends on the **cost model** of your application.

### 3.5 F-Measure

The **F-measure** (or **F-score**) is the **harmonic mean** of precision and recall:

$$
F = \frac{2PR}{P + R} = \frac{2}{\frac{1}{P} + \frac{1}{R}}
$$

**Why harmonic mean?** The harmonic mean is **dominated by the smaller value**. If precision is 90% and recall is 10%, the arithmetic mean is 50% (misleading), but the F-measure is:

$$
F = \frac{2 \times 0.9 \times 0.1}{0.9 + 0.1} = \frac{0.18}{1.0} = 0.18
$$

This correctly reflects that the system is poor due to low recall.

**Balanced F-measure ($F_1$):** weights precision and recall equally.

**General form ($F_\beta$):** weights recall $\beta$ times as important as precision:

$$
F_\beta = \frac{(1 + \beta^2) PR}{\beta^2 P + R}
$$

- $\beta < 1$: emphasize precision
- $\beta = 1$: balanced (standard $F_1$)
- $\beta > 1$: emphasize recall (e.g., $F_2$ for recall-critical applications like OpenSanctions)

**Example computation:**

Given $P = 0.7$, $R = 0.467$:

$$
F_1 = \frac{2 \times 0.7 \times 0.467}{0.7 + 0.467} = \frac{0.6538}{1.167} \approx 0.560
$$

### 3.6 Worked Example — Unranked Retrieval

**Collection:** 1000 documents. Query `"sanctions evasion"`.

**Ground truth (qrels):** 20 relevant documents exist in the collection.

**System output:** retrieves 30 documents (unranked set).

**Inspection reveals:** of the 30 retrieved, 12 are relevant.

**Calculate metrics:**

- TP = 12 (retrieved and relevant)
- FP = 30 - 12 = 18 (retrieved but not relevant)
- FN = 20 - 12 = 8 (relevant but not retrieved)
- TN = 1000 - 30 - 8 = 962 (not retrieved and not relevant)

$$
\text{Precision} = \frac{12}{30} = 0.40 \quad (40\%)
$$

$$
\text{Recall} = \frac{12}{20} = 0.60 \quad (60\%)
$$

$$
F_1 = \frac{2 \times 0.40 \times 0.60}{0.40 + 0.60} = \frac{0.48}{1.0} = 0.48
$$

**Interpretation:** The system found 60% of the relevant documents (decent recall) but returned many irrelevant documents (only 40% precision). The $F_1$ score of 0.48 reflects this moderate performance.

---

## Part 4: Ranked Retrieval Evaluation — Precision@k, Recall@k

### 4.1 Why Ranking Matters

Modern IR systems return **ranked lists**, not unordered sets. The user sees the top 10 results, not a random 10.

**Precision and recall as defined above ignore ranking.** They treat a relevant document at rank 1 the same as a relevant document at rank 1000.

We need metrics that **reward systems for placing relevant documents early in the ranking**.

### 4.2 Precision at k (P@k)

**Precision@k:** precision computed on the **top $k$ retrieved documents only**.

$$
P@k = \frac{\text{number of relevant docs in top-}k}{k}
$$

Equivalently:

$$
P@k = \frac{1}{k} \sum_{i=1}^{k} \mathbb{1}[\text{rel}(d_i)]
$$

where $\mathbb{1}[\text{rel}(d_i)]$ is 1 if the document at rank $i$ is relevant, 0 otherwise.

**Common cut-offs:** $k = 5, 10, 20$ (matching user attention span or page size).

**Example:**

Top 10 results: `[R, N, R, R, N, N, R, N, N, N]` where R = relevant, N = non-relevant.

$$
P@10 = \frac{4}{10} = 0.4
$$

$$
P@5 = \frac{3}{5} = 0.6
$$

Note that $P@5 > P@10$ — the first 5 results are better than the full 10. This is typical: precision **decreases** as you retrieve more documents.

### 4.3 Recall at k (R@k)

**Recall@k:** fraction of **all** relevant documents that appear in the top $k$.

$$
R@k = \frac{\text{number of relevant docs in top-}k}{R}
$$

where $R$ is the **total number of relevant documents** for the query (from qrels).

$$
R@k = \frac{1}{R} \sum_{i=1}^{k} \mathbb{1}[\text{rel}(d_i)]
$$

**When $R = 0$:** recall is undefined (or conventionally set to 0 or 1 depending on implementation — be consistent).

**Example:**

Query has $R = 15$ relevant documents. Top 20 results contain 8 of them.

$$
R@20 = \frac{8}{15} \approx 0.533
$$

### 4.4 The Precision@k vs Recall@k Relationship

For a **fixed query**:

- As $k$ increases, $P@k$ typically **decreases** (more junk enters)
- As $k$ increases, $R@k$ typically **increases** (more relevant docs retrieved)

This is the **precision-recall curve**: plotting $P@k$ against $R@k$ for varying $k$.

**Example computation:**

Query has $R = 8$ relevant documents. Ranked list:

| Rank | Doc   | Relevant? | P@k  | R@k   |
|------|-------|-----------|------|-------|
| 1    | D042  | Yes       | 1.0  | 0.125 |
| 2    | D103  | No        | 0.5  | 0.125 |
| 3    | D871  | Yes       | 0.67 | 0.25  |
| 4    | D400  | Yes       | 0.75 | 0.375 |
| 5    | D522  | No        | 0.6  | 0.375 |
| 6    | D681  | No        | 0.5  | 0.375 |
| 7    | D720  | Yes       | 0.57 | 0.5   |
| 8    | D802  | No        | 0.5  | 0.5   |
| 9    | D903  | No        | 0.44 | 0.5   |
| 10   | D999  | Yes       | 0.5  | 0.625 |

At $k=4$: $P@4 = 3/4 = 0.75$, $R@4 = 3/8 = 0.375$

At $k=10$: $P@10 = 5/10 = 0.5$, $R@10 = 5/8 = 0.625$

As expected, precision dropped and recall increased.

### 4.5 R-Precision

**R-precision:** precision at rank $R$, where $R$ is the number of relevant documents.

$$
\text{R-precision} = P@R
$$

**Intuition:** If you retrieve exactly $R$ documents, how many are relevant? In the ideal ranking, $P@R = 1.0$ (all $R$ are relevant). In the worst case, $P@R \approx 0$.

**Advantage:** $R$ is a natural cut-off — query-dependent, reflects collection characteristics.

**Example:** Query has $R = 8$ relevant documents. Top 8 results contain 5 relevant docs.

$$
\text{R-precision} = \frac{5}{8} = 0.625
$$

---

## Part 5: Mean Reciprocal Rank (MRR)

### 5.1 The Single-Answer Scenario

Many queries have **one clearly correct answer**:

- `"capital of France"` → Paris (one right answer)
- `"IMO9553359"` → the vessel with that identifier (unique)
- `"who wrote Hamlet"` → Shakespeare

For such queries, we care about the **rank of the first relevant document**. If the answer is at rank 1, perfect. At rank 10, poor.

### 5.2 Reciprocal Rank

For a single query, the **reciprocal rank** is:

$$
\text{RR} = \frac{1}{r}
$$

where $r$ is the rank of the **first relevant document**. If no relevant document is retrieved, $\text{RR} = 0$.

**Examples:**

- First relevant doc at rank 1: $\text{RR} = 1.0$
- First relevant doc at rank 2: $\text{RR} = 0.5$
- First relevant doc at rank 5: $\text{RR} = 0.2$
- No relevant doc retrieved: $\text{RR} = 0$

### 5.3 Mean Reciprocal Rank (MRR)

**MRR** averages reciprocal rank over a set of queries $Q$:

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{r_q}
$$

where $r_q$ is the rank of the first relevant document for query $q$.

**Example:**

| Query | First Relevant at Rank | RR   |
|-------|------------------------|------|
| Q1    | 1                      | 1.0  |
| Q2    | 3                      | 0.33 |
| Q3    | 2                      | 0.5  |
| Q4    | none                   | 0.0  |

$$
\text{MRR} = \frac{1.0 + 0.33 + 0.5 + 0.0}{4} = \frac{1.83}{4} = 0.4575
$$

### 5.4 When to Use MRR

**Use MRR when:**
- Queries have a single target answer (navigational queries, fact lookup, identifier search)
- The user will stop at the first relevant result

**Do NOT use MRR when:**
- Multiple documents are relevant and the user needs several (informational queries)
- Relevance is graded (MRR is binary — first relevant or not)

**In the OpenSanctions project:** MRR is appropriate for **Type 1** (exact identifier) and possibly **Type 2** (name/alias) queries where the user seeks a specific entity.

---

## Part 6: Average Precision and Mean Average Precision (MAP)

### 6.1 The Limitation of P@k

Precision@k measures quality at a **single cut-off**. But consider two rankings:

**Ranking A:** `[R, R, R, N, N, N, N, N, N, N]`
**Ranking B:** `[N, N, N, N, N, N, N, R, R, R]`

Both have $P@10 = 3/10 = 0.3$. But Ranking A is **clearly better** — it places relevant documents earlier.

We need a metric that **rewards early placement** and summarizes performance **across all ranks**.

### 6.2 Average Precision (AP)

**Average Precision** for a single query is the **average of precision values at each rank where a relevant document is retrieved**.

$$
\text{AP} = \frac{1}{R} \sum_{k=1}^{n} P@k \cdot \mathbb{1}[\text{rel at rank } k]
$$

where:
- $R$ = total number of relevant documents for the query
- $n$ = number of documents retrieved (or collection size if all ranked)
- $\mathbb{1}[\text{rel at rank } k]$ = 1 if the document at rank $k$ is relevant, else 0

**Alternative formulation (equivalent):**

$$
\text{AP} = \frac{\sum_{k=1}^{n} (P@k \times \text{rel}(k))}{R}
$$

**Intuition:** AP computes precision at **every** position where a relevant document appears, then averages. Documents deep in the ranking contribute less because precision has degraded by then.

### 6.3 Worked Example — Average Precision

Query has $R = 5$ relevant documents. Ranking of 10 documents:

| Rank $k$ | Doc   | Rel? | P@k  | P@k if rel |
|----------|-------|------|------|------------|
| 1        | D1    | Yes  | 1.0  | 1.0        |
| 2        | D2    | No   | 0.5  | —          |
| 3        | D3    | Yes  | 0.67 | 0.67       |
| 4        | D4    | No   | 0.5  | —          |
| 5        | D5    | No   | 0.4  | —          |
| 6        | D6    | Yes  | 0.5  | 0.5        |
| 7        | D7    | No   | 0.43 | —          |
| 8        | D8    | Yes  | 0.5  | 0.5        |
| 9        | D9    | No   | 0.44 | —          |
| 10       | D10   | Yes  | 0.5  | 0.5        |

Relevant docs appear at ranks: 1, 3, 6, 8, 10.

Precision at those ranks: 1.0, 0.67, 0.5, 0.5, 0.5

$$
\text{AP} = \frac{1.0 + 0.67 + 0.5 + 0.5 + 0.5}{5} = \frac{3.17}{5} = 0.634
$$

**Compare to another ranking:** `[N, N, N, N, N, R, R, R, R, R]`

Relevant docs at ranks 6, 7, 8, 9, 10.
Precisions: 1/6, 2/7, 3/8, 4/9, 5/10 = 0.167, 0.286, 0.375, 0.444, 0.5

$$
\text{AP} = \frac{0.167 + 0.286 + 0.375 + 0.444 + 0.5}{5} = \frac{1.772}{5} = 0.354
$$

The first ranking (AP = 0.634) is superior because relevant documents appeared earlier.

### 6.4 Mean Average Precision (MAP)

**MAP** is the mean of AP across all queries:

$$
\text{MAP} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}(q)
$$

**Why MAP is important:**

- Single-number summary of ranked retrieval quality
- Rewards **both** early placement and completeness
- Stable across queries with different numbers of relevant documents
- **Standard metric in IR research** — most papers report MAP

**Example:**

| Query | AP   |
|-------|------|
| Q1    | 0.8  |
| Q2    | 0.6  |
| Q3    | 0.9  |
| Q4    | 0.4  |

$$
\text{MAP} = \frac{0.8 + 0.6 + 0.9 + 0.4}{4} = \frac{2.7}{4} = 0.675
$$

### 6.5 Properties of AP and MAP

1. **AP ranges from 0 to 1.** AP = 1 only if all relevant documents are ranked first (perfect ranking).

2. **AP is sensitive to rank position.** Moving a relevant document from rank 10 to rank 1 dramatically increases AP.

3. **AP handles incomplete relevance judgements.** Unjudged documents are treated as non-relevant (standard assumption).

4. **MAP is macro-averaged** — each query contributes equally regardless of how many relevant documents it has. (Alternative: micro-averaging weighs by number of judgements — less common.)

5. **MAP is the standard in TREC.** Historically, TREC collections report MAP as the primary metric for ad-hoc retrieval.

---

## Part 7: Graded Relevance — DCG and nDCG

### 7.1 Beyond Binary Relevance

Real-world relevance is not binary. Some documents are **highly relevant**, others **marginally relevant**, others **not relevant at all**.

**Example (query: "climate change impacts"):**

- Document A: IPCC report on climate impacts → **highly relevant** (grade 3)
- Document B: news article mentioning climate change briefly → **marginally relevant** (grade 1)
- Document C: article about weather (not climate) → **not relevant** (grade 0)

**Binary relevance** treats A and B the same (both = 1). This loses important information.

**Graded relevance scales (common choices):**

- **3-point:** 0 (not relevant), 1 (marginally relevant), 2 (highly relevant)
- **4-point:** 0 (not relevant), 1 (marginally), 2 (relevant), 3 (highly relevant)
- **5-point:** 0 to 4 (TREC graded relevance)

### 7.2 Discounted Cumulative Gain (DCG)

**Cumulative Gain (CG)** at position $p$ is the sum of relevance grades up to rank $p$:

$$
\text{CG}_p = \sum_{i=1}^{p} g_i
$$

where $g_i$ is the relevance grade of the document at rank $i$.

**Problem with CG:** it does not account for **position**. A highly relevant document at rank 10 contributes the same as one at rank 1.

**Discounted Cumulative Gain (DCG)** applies a **logarithmic discount** to lower ranks:

$$
\text{DCG}_p = \sum_{i=1}^{p} \frac{2^{g_i} - 1}{\log_2(i+1)}
$$

**Breaking down the formula:**

1. **Gain term:** $2^{g_i} - 1$ (exponential in grade — heavily rewards high grades)
   - $g_i = 0$: gain = $2^0 - 1 = 0$
   - $g_i = 1$: gain = $2^1 - 1 = 1$
   - $g_i = 2$: gain = $2^2 - 1 = 3$
   - $g_i = 3$: gain = $2^3 - 1 = 7$

2. **Discount term:** $\frac{1}{\log_2(i+1)}$ (logarithmic penalty for lower ranks)
   - Rank 1: discount = $1/\log_2(2) = 1.0$ (no penalty)
   - Rank 2: discount = $1/\log_2(3) \approx 0.631$
   - Rank 5: discount = $1/\log_2(6) \approx 0.387$
   - Rank 10: discount = $1/\log_2(11) \approx 0.289$

**Alternative DCG formulation (less common):**

$$
\text{DCG}_p = g_1 + \sum_{i=2}^{p} \frac{g_i}{\log_2(i)}
$$

This version uses raw grades $g_i$ instead of $2^{g_i} - 1$. Results differ numerically but rankings are similar.

### 7.3 Worked Example — DCG Computation

Query returns 5 documents with graded relevance (0–3 scale):

| Rank $i$ | Grade $g_i$ | $2^{g_i} - 1$ | $\log_2(i+1)$ | $\frac{2^{g_i} - 1}{\log_2(i+1)}$ |
|----------|-------------|---------------|---------------|-----------------------------------|
| 1        | 3           | 7             | 1.0           | 7.0                               |
| 2        | 2           | 3             | 1.585         | 1.893                             |
| 3        | 3           | 7             | 2.0           | 3.5                               |
| 4        | 0           | 0             | 2.322         | 0                                 |
| 5        | 1           | 1             | 2.585         | 0.387                             |

$$
\text{DCG}_5 = 7.0 + 1.893 + 3.5 + 0 + 0.387 = 12.78
$$

### 7.4 Normalized DCG (nDCG)

**Problem with DCG:** not comparable across queries. A query with many highly relevant documents has higher DCG than one with few relevant documents, even if both rankings are perfect.

**Solution:** normalize by the **ideal DCG (IDCG)** — the DCG of the **best possible ranking** for that query.

$$
\text{nDCG}_p = \frac{\text{DCG}_p}{\text{IDCG}_p}
$$

**IDCG computation:** sort all relevant documents by grade (descending), compute DCG on that ideal ranking.

**Interpretation:**
- nDCG = 1.0: perfect ranking (best possible)
- nDCG = 0: no relevant documents retrieved
- nDCG ∈ (0, 1): quality of ranking relative to ideal

### 7.5 Worked Example — nDCG Computation

**Query has 8 documents with grades (from qrels):**

| Doc ID | Grade |
|--------|-------|
| D1     | 3     |
| D2     | 2     |
| D3     | 3     |
| D4     | 0     |
| D5     | 1     |
| D6     | 0     |
| D7     | 2     |
| D8     | 0     |

**System ranking (top 5):**

| Rank | Doc ID | Grade |
|------|--------|-------|
| 1    | D1     | 3     |
| 2    | D4     | 0     |
| 3    | D3     | 3     |
| 4    | D5     | 1     |
| 5    | D7     | 2     |

From above, $\text{DCG}_5 = 12.78$.

**Ideal ranking (top 5):** sort by grade descending:

| Rank | Doc ID | Grade |
|------|--------|-------|
| 1    | D1     | 3     |
| 2    | D3     | 3     |
| 3    | D2     | 2     |
| 4    | D7     | 2     |
| 5    | D5     | 1     |

Compute IDCG:

| Rank $i$ | Grade $g_i$ | $2^{g_i} - 1$ | $\log_2(i+1)$ | Contribution |
|----------|-------------|---------------|---------------|--------------|
| 1        | 3           | 7             | 1.0           | 7.0          |
| 2        | 3           | 7             | 1.585         | 4.416        |
| 3        | 2           | 3             | 2.0           | 1.5          |
| 4        | 2           | 3             | 2.322         | 1.292        |
| 5        | 1           | 1             | 2.585         | 0.387        |

$$
\text{IDCG}_5 = 7.0 + 4.416 + 1.5 + 1.292 + 0.387 = 14.595
$$

$$
\text{nDCG}_5 = \frac{12.78}{14.595} \approx 0.876
$$

**Interpretation:** The system achieved 87.6% of the ideal ranking quality at rank 5. The main loss came from placing non-relevant D4 at rank 2.

### 7.6 When to Use nDCG

**Use nDCG when:**
- You have graded relevance judgements (not just binary)
- Position matters (you care about rank order)
- You want a normalized score comparable across queries

**nDCG vs MAP:**
- MAP: binary relevance, emphasizes finding all relevant documents
- nDCG: graded relevance, emphasizes ranking quality at top positions

**In the OpenSanctions project:** nDCG@10 is the **primary metric** for **Types 3 and 4** (semantic and relational queries) where relevance is graded through pooling.

---

## Part 8: Evaluation Methodology and Pitfalls

### 8.1 Train-Test Split

**Golden rule:** Never evaluate on the same data you tuned on.

If you adjust BM25 parameters ($k_1$, $b$) to maximize MAP on query set Q, then report MAP on Q, you have **overfit**. Your reported scores are **optimistically biased**.

**Proper methodology:**

1. **Training set:** tune parameters, develop features, try different models
2. **Validation set:** select the best configuration
3. **Test set:** report final scores (never look at this until final evaluation)

**Split sizes (rule of thumb):**
- 50 queries total: 25 train, 10 validation, 15 test
- 150 queries: 75 train, 25 validation, 50 test

**Cross-validation:** If queries are scarce, use $k$-fold cross-validation to maximize data usage.

### 8.2 Pooling and Judging

**The pooling problem:** For 50 queries and 1 million documents, we have 50 million (query, document) pairs to judge. Cost: $\approx$ \$500K at \$0.01/judgement. Infeasible.

**Pooling solution (TREC methodology):**

1. Run $k$ systems (e.g., 10 different retrieval models) on the query set
2. For each query, **pool** the top-$m$ documents from each system (e.g., $m = 100$)
3. **Deduplicate** the pool (union of all top-$m$ sets)
4. Judge only documents in the pool
5. Assume unjudged documents are **non-relevant**

**Pool size:** typically 50–200 documents per query (manageable).

**Bias:** Systems that retrieve documents **outside the pool** are unfairly penalized (their novel relevant documents are counted as non-relevant). This biases evaluation **toward** systems that contributed to the pool.

**Mitigation:**
- Include diverse systems in pooling (Boolean, BM25, neural, etc.)
- Use deep pooling ($m = 100+$) to capture more diversity
- Report uncertainty estimates
- For critical systems, expand the pool

### 8.3 Inter-Assessor Agreement

Relevance judgements are **subjective**. Two assessors judging the same (query, document) pair may disagree.

**Cohen's kappa:** measures inter-annotator agreement (κ ∈ [0, 1], higher = more agreement).

**Typical agreement rates in IR:**
- Binary relevance: κ ≈ 0.6–0.8 (moderate to good)
- Graded relevance: κ ≈ 0.4–0.6 (fair to moderate)

**Implications:**
- Perfect evaluation is impossible
- Small score differences between systems (MAP difference < 0.01) may be noise
- Statistical significance testing is essential (not covered in detail here — see Manning §8.6)

### 8.4 Collection Size and Query Count

**Rule of thumb (from TREC experience):**

- **Minimum 50 queries** to get stable comparative results between systems
- **Minimum 1000 documents** in collection (preferably 10K+)
- **At least 10 relevance judgements per query** (preferably 50+)

**With fewer queries:** results are noisy. A system may appear better by chance.

**Statistical power:** With 50 queries, you can detect MAP differences of ≈0.03 with 95% confidence. Smaller differences require more queries.

### 8.5 Evaluation in This Project (OpenSanctions)

**Collection:** 1.2M entities (large)

**Queries:** designed to cover 7 types (target: 10–20 queries per type, 70–140 total)

**Qrels construction:**

- **Types 1, 2, 5, 6:** Automatic or semi-automatic (gold standard entities known)
- **Types 3, 4:** **Pooling** with manual grading (0/1/2 scale)
- **Type 7:** RAGAS metrics (faithfulness, answer relevance) — not traditional IR evaluation

**Primary metrics by type (from assignment §3.4):**

| Type   | Primary Metric(s)      | Secondary          |
|--------|------------------------|--------------------|
| 1, 2   | Precision@1, MRR       | MAP                |
| 3, 4   | nDCG@10                | MAP, Recall@10     |
| 5, 6   | Recall@10, Recall@20   | MAP                |
| 7      | RAGAS faithfulness     | Answer relevance   |

**System-level metric:** **Recall@10** (averaged across all query types) — reflects regulatory risk minimization.

---

## Part 9: Interpreting Evaluation Results

### 9.1 Absolute vs Relative Scores

**Absolute scores:**
- MAP = 0.45 — is this good or bad? **Cannot tell without context.**
- On easy queries (TREC-1): MAP = 0.45 is poor
- On hard queries (TREC Robust): MAP = 0.25 is state-of-the-art

**Relative comparison is what matters:**
- System A: MAP = 0.45
- System B: MAP = 0.50
- B is **11% better** than A (relative improvement: $(0.50 - 0.45)/0.45 \approx 0.11$)

**Report both:**
- Absolute scores (for reproducibility)
- Relative improvements over baseline (for interpretation)

### 9.2 Statistical Significance

A difference in MAP of 0.45 vs 0.50 may be:
- **Statistically significant:** real improvement, not noise
- **Not significant:** could be random variation

**Significance tests (common in IR):**
- **Paired t-test:** compare per-query scores
- **Wilcoxon signed-rank test:** non-parametric alternative
- **Bootstrap test:** resample queries, compute p-value

**Conventional threshold:** $p < 0.05$ (5% significance level)

**Practical significance:** Even if statistically significant, is a 2% improvement worth the engineering cost?

### 9.3 Error Analysis

When System A beats System B, **why?**

**Drill down:**
- Which queries did A win? Which did B win?
- Are differences consistent across query types?
- Inspect top-ranked documents for failed queries
- Are failures due to: indexing, tokenization, relevance model, or qrels errors?

**Example:** System A (BM25) beats System B (Boolean) overall, but System B wins on Type 1 (exact identifiers). Why? Boolean is perfect for exact matches; BM25's soft matching is unnecessary.

**Lesson:** No single system is best for all query types. Consider query-type routing or hybrid systems.

---

## Part 10: Practical Evaluation Workflow

### 10.1 Step-by-Step Evaluation

**Step 1: Prepare qrels**

- Define relevance scale (binary or graded)
- Create qrels file (JSON or TREC format)
- Ensure quality (inter-annotator checks if manual)

**Step 2: Generate system runs**

- Run your system(s) on all queries
- Output ranked lists (TREC run format):

```
query_id  Q0  doc_id  rank  score  run_name
Q1        Q0  D042    1     15.3   BM25_k1.2
Q1        Q0  D103    2     12.1   BM25_k1.2
...
```

**Step 3: Compute metrics**

Use evaluation library:
- **Python:** `ranx`, `pytrec_eval`, `ir_measures`
- **Command-line:** `trec_eval` (C program, TREC standard)

```python
from ranx import Qrels, Run, evaluate

qrels = Qrels.from_file("qrels.json")
run = Run.from_file("run.txt")

results = evaluate(qrels, run, ["map", "ndcg@10", "recall@10"])
print(results)
# {'map': 0.456, 'ndcg@10': 0.623, 'recall@10': 0.734}
```

**Step 4: Analyze and iterate**

- Compare systems
- Identify failure cases
- Improve and re-evaluate

### 10.2 Common Pitfalls

**Pitfall 1:** Forgetting that qrels are incomplete. Unjudged ≠ non-relevant (but we assume it).

**Pitfall 2:** Reporting only aggregate scores. Per-query analysis is essential.

**Pitfall 3:** Tuning on test set. Use proper splits.

**Pitfall 4:** Ignoring statistical significance. Small differences may be noise.

**Pitfall 5:** Choosing the wrong metric. Precision@1 for multi-document queries is wrong; use MAP or nDCG.

---

## Part 11: Advanced Topics (Brief Overview)

### 11.1 User Studies

**Limitation of qrels-based evaluation:** Assumes relevance is the only factor. In reality, users care about:
- Diversity (not 10 near-duplicate results)
- Novelty (documents they haven't seen)
- Credibility (trustworthy sources)
- Presentation (snippets, highlighting)

**User studies** measure task completion time, satisfaction, clicks. Essential for deployed systems but expensive for research.

### 11.2 Online Evaluation (A/B Testing)

For web search engines:
- **Offline evaluation** (qrels): initial filtering
- **Online evaluation** (A/B tests): measure real user behavior (click-through rate, dwell time, session success)

**Metrics:**
- Click-through rate (CTR)
- Mean reciprocal rank of clicked document
- Abandonment rate (no clicks = failure)

### 11.3 Diversity and Novelty Metrics

**Problem:** A system that returns 10 near-duplicate relevant documents scores well on MAP but poorly serves users.

**Metrics for diversity:**
- α-nDCG (penalizes redundancy)
- Intent-aware metrics (subtopic recall)

Not covered in this course but important for web search.

### 11.4 Learning to Rank

**Machine learning for IR:** Instead of hand-tuning BM25, learn ranking functions from data.

**Requires:**
- Features (BM25 score, query length, document length, etc.)
- Training qrels
- Learning algorithm (RankSVM, LambdaMART, neural rankers)

**Evaluation:** Same metrics (MAP, nDCG) applied to learned rankers.

---

## Part 12: Summary and Key Takeaways

### 12.1 Core Metrics — When to Use What

| Metric         | Use Case                                              | Range   |
|----------------|-------------------------------------------------------|---------|
| **Precision**  | Unranked sets; how much junk is returned             | [0, 1]  |
| **Recall**     | Unranked sets; how many relevant docs found          | [0, 1]  |
| **F-measure**  | Balance precision and recall                         | [0, 1]  |
| **P@k**        | Ranked lists; quality of top-$k$                     | [0, 1]  |
| **R@k**        | Ranked lists; coverage in top-$k$                    | [0, 1]  |
| **MRR**        | Single-answer queries (navigational, fact lookup)    | [0, 1]  |
| **MAP**        | Ranked lists; overall quality across queries         | [0, 1]  |
| **nDCG@k**     | Graded relevance; ranking quality at top-$k$         | [0, 1]  |

### 12.2 The Golden Rules of Evaluation

1. **Use proper train-test splits** — never evaluate on tuning data
2. **Report multiple metrics** — no single number tells the whole story
3. **Perform error analysis** — understand why systems fail
4. **Test statistical significance** — small differences may be noise
5. **Choose metrics that match your task** — precision for screening, recall for discovery, nDCG for graded relevance
6. **Document your methodology** — qrels construction, pooling, parameters

### 12.3 Worked End-to-End Example

**Scenario:** Evaluate two systems on OpenSanctions Type 3 queries (semantic search).

**Test collection:**
- 10 queries
- Graded qrels (0/1/2 scale)
- 8 relevant documents per query on average

**System A (BM25):** Run generates ranked list for each query.
**System B (TF-IDF):** Run generates ranked list for each query.

**Results:**

| Query | AP (A) | AP (B) | nDCG@10 (A) | nDCG@10 (B) |
|-------|--------|--------|-------------|-------------|
| Q1    | 0.8    | 0.6    | 0.85        | 0.70        |
| Q2    | 0.7    | 0.65   | 0.78        | 0.75        |
| Q3    | 0.6    | 0.7    | 0.70        | 0.80        |
| ...   | ...    | ...    | ...         | ...         |
| Q10   | 0.75   | 0.70   | 0.82        | 0.78        |

**Aggregate:**

- MAP(A) = 0.725, MAP(B) = 0.678 → A wins by 6.9%
- nDCG@10(A) = 0.793, nDCG@10(B) = 0.758 → A wins by 4.6%

**Statistical test:** paired t-test on per-query AP scores → $p = 0.03$ (significant at $\alpha = 0.05$).

**Conclusion:** System A (BM25) significantly outperforms System B (TF-IDF) on semantic queries. Recommend BM25 for production.

---

## References and Further Reading

**Primary:**
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, Chapter 8 (Evaluation in information retrieval) — [Online PDF](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf)

**Labs:**
- [Week 4 Lab 3: Evaluation and Interface](../../raw_materials/week_4/Lab3_Evaluation%20and%20Interface.ipynb)

**Standard test collections:**
- TREC collections: https://trec.nist.gov/
- CLEF: http://www.clef-campaign.org/
- NTCIR: http://research.nii.ac.jp/ntcir/

**Tools:**
- `ranx`: Python library for IR evaluation — https://github.com/AmenRa/ranx
- `pytrec_eval`: Python interface to trec_eval
- `trec_eval`: C program (TREC standard) — https://github.com/usnistgov/trec_eval

**Key papers:**
- Voorhees, E. M. (2000). Variations in relevance judgments and the measurement of retrieval effectiveness. *Information Processing & Management.*
- Buckley, C., & Voorhees, E. M. (2004). Retrieval evaluation with incomplete information. *SIGIR.*
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM TOIS.* (nDCG paper)

**This project:**
- [Assignment 1 Submission Report](../../../docs/assignment_1_submission_report.pdf) §3.4 Evaluation Strategy

---

**End of Evaluation Theory** — Next: Apply these metrics in project evaluation ([notebooks/](../../../notebooks/)) once qrels are constructed.
