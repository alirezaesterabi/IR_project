# Detailed Lecture: Probabilistic Information Retrieval and BM25

**A Comprehensive Deep Dive into the Theory**

---

## Table of Contents

1. [Introduction to Probabilistic IR](#1-introduction-to-probabilistic-ir)
2. [Review of Basic Probability Theory](#2-review-of-basic-probability-theory)
3. [The Probability Ranking Principle](#3-the-probability-ranking-principle)
4. [The Binary Independence Model (BIM)](#4-the-binary-independence-model-bim)
5. [Deriving the Ranking Function](#5-deriving-the-ranking-function)
6. [Robertson-Sparck Jones Weights](#6-robertson-sparck-jones-weights)
7. [Probability Estimates in Practice](#7-probability-estimates-in-practice)
8. [From BIM to BM25: Incorporating Term Frequency](#8-from-bim-to-bm25-incorporating-term-frequency)
9. [The 2-Poisson Model](#9-the-2-poisson-model)
10. [BM25 Formula Derivation](#10-bm25-formula-derivation)
11. [Understanding BM25 Parameters](#11-understanding-bm25-parameters)
12. [BM25 Extensions and Variants](#12-bm25-extensions-and-variants)
13. [Comparison with Vector Space Models](#13-comparison-with-vector-space-models)
14. [Practical Implementation Considerations](#14-practical-implementation-considerations)

---

## 1. Introduction to Probabilistic IR

### 1.1 Motivation: Why Probabilistic IR?

The Vector Space Model (VSM) we studied earlier has several limitations:

- **Heuristic foundations**: TF-IDF weights are based on intuition rather than rigorous theory
- **No principled way to combine evidence**: How should we weight different factors?
- **Lack of theoretical justification**: Why logarithms? Why cosine similarity?

**Probabilistic Information Retrieval** provides a theoretically principled framework by asking:

> **Central Question**: Given a query $q$ and a document $d$, what is the probability that this document is relevant to the user's information need?

This formulation allows us to:
1. Use probability theory as our foundation
2. Make explicit assumptions about what we're modeling
3. Derive ranking functions from first principles
4. Understand what our models are actually doing

### 1.2 Historical Context

The probabilistic approach to IR was pioneered by several researchers:

- **Maron and Kuhns (1960)**: First probabilistic model for IR
- **Robertson and Sparck Jones (1976)**: Binary Independence Model
- **Robertson, Walker et al. (1990s)**: BM25 from TREC experiments
- **Ponte and Croft (1998)**: Language modeling approach

The development culminated in **BM25**, which remains the state-of-the-art baseline for sparse retrieval.

---

## 2. Review of Basic Probability Theory

Before diving into probabilistic IR, let's review the essential probability concepts.

### 2.1 Basic Probability

**Sample space** $\Omega$: The set of all possible outcomes

**Event** $A$: A subset of the sample space

**Probability** $P(A)$: A number between 0 and 1 representing the likelihood of event $A$

**Axioms of Probability**:
1. $0 \leq P(A) \leq 1$ for all events $A$
2. $P(\Omega) = 1$ (something must happen)
3. If $A$ and $B$ are mutually exclusive: $P(A \cup B) = P(A) + P(B)$

### 2.2 Conditional Probability

The probability of event $A$ given that event $B$ has occurred:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**Interpretation**: What fraction of the time does $A$ occur when we know $B$ has occurred?

### 2.3 Bayes' Theorem

One of the most important tools in probabilistic IR:

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

**In words**:
- $P(A \mid B)$: **Posterior probability** - what we want to know
- $P(B \mid A)$: **Likelihood** - how likely is our observation given the hypothesis
- $P(A)$: **Prior probability** - what we knew before observing $B$
- $P(B)$: **Evidence** - normalizing constant

### 2.4 Odds and Log Odds

**Odds** of event $A$:

$$O(A) = \frac{P(A)}{P(\neg A)} = \frac{P(A)}{1 - P(A)}$$

**Log odds** (logit):

$$\text{logit}(A) = \log O(A) = \log \frac{P(A)}{1 - P(A)}$$

**Why use log odds?**
- Converts multiplication to addition
- Maps $(0,1)$ to $(-\infty, +\infty)$
- Symmetric around 0 (odds of 1:1 gives log odds of 0)

### 2.5 Independence

Events $A$ and $B$ are **independent** if:

$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently: $P(A \mid B) = P(A)$ (knowing $B$ tells us nothing about $A$)

**Conditional independence**: $A$ and $B$ are independent given $C$ if:

$$P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)$$

This is weaker than full independence and is the key assumption in the Binary Independence Model.

---

## 3. The Probability Ranking Principle

### 3.1 The Optimal Ranking Strategy

**Probability Ranking Principle (PRP)** (Robertson, 1977):

> If a reference retrieval system's response to each request is a ranking of the documents in the collection in order of decreasing probability of relevance to the user who submitted the request, where the probabilities are estimated as accurately as possible on the basis of whatever data have been made available to the system for this purpose, the overall effectiveness of the system to its user will be the best that is obtainable on the basis of those data.

**In simpler terms**: Rank documents by $P(\text{relevant} \mid d, q)$ in decreasing order.

### 3.2 Formalization

Let:
- $R$ be a binary random variable: $R = 1$ if document is relevant, $R = 0$ otherwise
- $d$ represent a document
- $q$ represent a query

We want to rank by: $P(R = 1 \mid d, q)$

### 3.3 Derivation Using Bayes' Theorem

Apply Bayes' theorem:

$$P(R = 1 \mid d, q) = \frac{P(d, q \mid R = 1) \cdot P(R = 1)}{P(d, q)}$$

For ranking purposes, we can ignore:
- $P(d, q)$: Same for all documents (query is fixed)
- Often $P(R = 1)$: Assume constant prior

So we rank by: $P(d, q \mid R = 1)$ (likelihood of document given relevance)

### 3.4 The Likelihood Ratio

Even better, consider the **odds of relevance**:

$$\frac{P(R = 1 \mid d, q)}{P(R = 0 \mid d, q)}$$

Using Bayes' theorem on both numerator and denominator:

$$\frac{P(R = 1 \mid d, q)}{P(R = 0 \mid d, q)} = \frac{P(d, q \mid R = 1)}{P(d, q \mid R = 0)} \cdot \frac{P(R = 1)}{P(R = 0)}$$

The likelihood ratio $\frac{P(d, q \mid R = 1)}{P(d, q \mid R = 0)}$ tells us:

> How much more likely is this document-query pair under the relevant distribution compared to the non-relevant distribution?

Taking logarithms:

$$\log \frac{P(R = 1 \mid d, q)}{P(R = 0 \mid d, q)} = \log \frac{P(d, q \mid R = 1)}{P(d, q \mid R = 0)} + \log \frac{P(R = 1)}{P(R = 0)}$$

The log-likelihood ratio is what we'll compute for ranking.

### 3.5 Why This is Optimal

**Theorem**: Ranking by decreasing $P(R = 1 \mid d, q)$ minimizes the expected loss under a 0-1 loss function.

**Proof sketch**:
- If we retrieve document $d$ when it's not relevant: loss = 1
- If we don't retrieve document $d$ when it is relevant: loss = 1
- Otherwise: loss = 0

Expected loss is minimized by retrieving documents with $P(R = 1 \mid d, q) > 0.5$ and ranking them in decreasing order of this probability.

---

## 4. The Binary Independence Model (BIM)

### 4.1 Document Representation

In the Binary Independence Model, documents and queries are represented as **binary term incidence vectors**:

$$\vec{x} = (x_1, x_2, \ldots, x_{|V|}) \in \{0, 1\}^{|V|}$$

where:
- $x_i = 1$ if term $i$ appears in the document
- $x_i = 0$ if term $i$ does not appear
- $|V|$ is the vocabulary size

**Key point**: We only care about presence/absence, not frequency.

### 4.2 The Independence Assumption

The **conditional independence assumption** states:

> Given relevance or non-relevance, the presence of different terms in a document are independent events.

Formally:

$$P(\vec{x} \mid R) = \prod_{i=1}^{|V|} P(x_i \mid R)$$

**Why make this assumption?**
- Makes the mathematics tractable
- Reduces the number of parameters to estimate from $2^{|V|}$ to $2|V|$
- Empirically works reasonably well despite being obviously false

**When does it fail?**
- "New York" - presence of "New" and "York" are not independent
- "United" and "States" - highly correlated
- Phrases and collocations in general

Despite these violations, the model is surprisingly effective (like Naive Bayes in machine learning).

### 4.3 Term-Specific Probabilities

For each term $i$, define:
- $p_i = P(x_i = 1 \mid R = 1)$: probability term $i$ appears in a relevant document
- $u_i = P(x_i = 1 \mid R = 0)$: probability term $i$ appears in a non-relevant document

Also:
- $1 - p_i = P(x_i = 0 \mid R = 1)$: probability term $i$ absent from relevant doc
- $1 - u_i = P(x_i = 0 \mid R = 0)$: probability term $i$ absent from non-relevant doc

**Intuition**:
- If $p_i \gg u_i$: term $i$ is a good indicator of relevance
- If $p_i \approx u_i$: term $i$ doesn't discriminate
- If $p_i < u_i$: term $i$ is actually anti-correlated with relevance (rare)

---

## 5. Deriving the Ranking Function

### 5.1 The Likelihood Ratio Under BIM

We need to compute:

$$\frac{P(\vec{x} \mid R = 1)}{P(\vec{x} \mid R = 0)}$$

Under the independence assumption:

$$\frac{P(\vec{x} \mid R = 1)}{P(\vec{x} \mid R = 0)} = \frac{\prod_{i=1}^{|V|} P(x_i \mid R = 1)}{\prod_{i=1}^{|V|} P(x_i \mid R = 0)}$$

### 5.2 Splitting by Term Presence

For each term $i$, we have two cases:

**Case 1**: $x_i = 1$ (term present)

$$\frac{P(x_i = 1 \mid R = 1)}{P(x_i = 1 \mid R = 0)} = \frac{p_i}{u_i}$$

**Case 2**: $x_i = 0$ (term absent)

$$\frac{P(x_i = 0 \mid R = 1)}{P(x_i = 0 \mid R = 0)} = \frac{1 - p_i}{1 - u_i}$$

### 5.3 Combining All Terms

$$\frac{P(\vec{x} \mid R = 1)}{P(\vec{x} \mid R = 0)} = \prod_{i: x_i = 1} \frac{p_i}{u_i} \cdot \prod_{i: x_i = 0} \frac{1 - p_i}{1 - u_i}$$

Taking logarithms:

$$\log \frac{P(\vec{x} \mid R = 1)}{P(\vec{x} \mid R = 0)} = \sum_{i: x_i = 1} \log \frac{p_i}{u_i} + \sum_{i: x_i = 0} \log \frac{1 - p_i}{1 - u_i}$$

### 5.4 Query-Specific Simplification

**Key insight**: For ranking purposes, we only care about terms that appear in the query!

Let $q_i = 1$ if term $i$ appears in query, $q_i = 0$ otherwise.

For terms not in the query ($q_i = 0$), the contribution to the log likelihood is the same for all documents (whether $x_i = 0$ or $x_i = 1$). Since this is constant across documents, we can ignore it for ranking.

### 5.5 The Retrieval Status Value (RSV)

Focus only on query terms. The **Retrieval Status Value** is:

$$\text{RSV}(d, q) = \sum_{i: q_i = 1, x_i = 1} \log \frac{p_i(1 - u_i)}{u_i(1 - p_i)} + \sum_{i: q_i = 1, x_i = 0} \log \frac{1 - p_i}{1 - u_i}$$

The second sum is constant for all documents (depends only on query), so:

$$\text{RSV}(d, q) = \sum_{i \in q \cap d} \log \frac{p_i(1 - u_i)}{u_i(1 - p_i)} + \text{constant}$$

Define the **term weight**:

$$c_i = \log \frac{p_i(1 - u_i)}{u_i(1 - p_i)}$$

Then:

$$\boxed{\text{RSV}(d, q) = \sum_{i \in q \cap d} c_i}$$

**This is our ranking function!** Sum the weights of query terms that appear in the document.

### 5.6 Interpretation of the Term Weight

$$c_i = \log \frac{p_i(1 - u_i)}{u_i(1 - p_i)} = \log \frac{p_i}{u_i} + \log \frac{1 - u_i}{1 - p_i}$$

**Two components**:
1. $\log \frac{p_i}{u_i}$: How much more likely is term $i$ in relevant vs. non-relevant docs?
2. $\log \frac{1 - u_i}{1 - p_i}$: How much more likely is term $i$ to be absent from non-relevant vs. relevant docs?

**Special cases**:
- If $p_i = 1$ (always in relevant docs) and $u_i = 0$ (never in non-relevant): $c_i = +\infty$
- If $p_i = 0$ and $u_i = 1$: $c_i = -\infty$
- If $p_i = u_i$: $c_i = 0$ (term doesn't discriminate)

---

## 6. Robertson-Sparck Jones Weights

### 6.1 The Estimation Problem

We have a beautiful ranking function, but there's a catch: **we don't know $p_i$ and $u_i$!**

These probabilities require knowledge of which documents are relevant, but if we knew that, we wouldn't need retrieval!

### 6.2 Estimation Without Relevance Information

**Assumptions** (for initial retrieval):

1. **Constant $p_i$**: Assume $p_i = 0.5$ for all terms (maximum uncertainty)
   - Equivalent to saying: "A term is equally likely to appear or not appear in a relevant document"

2. **Estimate $u_i$ from collection statistics**:
   $$u_i \approx \frac{\text{df}_i}{N}$$
   where $\text{df}_i$ is document frequency and $N$ is collection size
   - Assumes most documents are non-relevant (usually true)

### 6.3 Substituting into the Weight Formula

$$c_i = \log \frac{p_i(1 - u_i)}{u_i(1 - p_i)}$$

With $p_i = 0.5$:

$$c_i = \log \frac{0.5(1 - u_i)}{u_i(1 - 0.5)} = \log \frac{0.5(1 - u_i)}{0.5 u_i} = \log \frac{1 - u_i}{u_i}$$

Substituting $u_i = \frac{\text{df}_i}{N}$:

$$c_i = \log \frac{1 - \text{df}_i/N}{\text{df}_i/N} = \log \frac{N - \text{df}_i}{\text{df}_i}$$

This is the **Robertson-Sparck Jones (RSJ) weight**!

### 6.4 Smoothing the RSJ Weight

To avoid division by zero and numerical issues:

$$c_i = \log \frac{N - \text{df}_i + 0.5}{\text{df}_i + 0.5}$$

**This is exactly the IDF formula used in BM25!**

### 6.5 Connection to Classical IDF

Classical IDF: $\log \frac{N}{\text{df}_i}$

RSJ weight: $\log \frac{N - \text{df}_i}{\text{df}_i}$

Relationship:

$$\log \frac{N - \text{df}_i}{\text{df}_i} = \log \left(\frac{N}{\text{df}_i} - 1\right)$$

**Differences**:
- When $\text{df}_i$ is small: $\log \frac{N - \text{df}_i}{\text{df}_i} \approx \log \frac{N}{\text{df}_i}$ (very similar)
- When $\text{df}_i > N/2$: RSJ weight becomes **negative** (term is anti-correlated with relevance)

### 6.6 Properties of RSJ Weights

**Monotonicity**: As $\text{df}_i$ increases, $c_i$ decreases
- Rare terms get higher weights
- Common terms get lower (or negative) weights

**Example** (for $N = 1,000,000$):

| Term | df | RSJ weight | Classical IDF |
|------|-----|------------|---------------|
| "the" | 990,000 | $\log(10,000/990,000) \approx -2$ | $\log(1.001) \approx 0$ |
| "information" | 100,000 | $\log(900,000/100,000) \approx 0.95$ | $\log(10) \approx 1.0$ |
| "retrieval" | 10,000 | $\log(990,000/10,000) \approx 2.0$ | $\log(100) \approx 2.0$ |
| "trec" | 100 | $\log(999,900/100) \approx 4.0$ | $\log(10,000) \approx 4.0$ |

Notice: RSJ gives negative weight to very common terms!

---

## 7. Probability Estimates in Practice

### 7.1 Relevance Feedback

When we have relevance information from the user, we can estimate $p_i$ and $u_i$ directly:

Let:
- $VR$ = set of known relevant documents
- $VNR$ = set of known non-relevant documents
- $V R_i$ = relevant documents containing term $i$
- $df_i$ = total documents containing term $i$

Then:

$$p_i = \frac{|VR_i|}{|VR|}$$

$$u_i = \frac{df_i - |VR_i|}{N - |VR|}$$

With smoothing:

$$c_i = \log \frac{(|VR_i| + 0.5)(N - |VR| - df_i + |VR_i| + 0.5)}{(df_i - |VR_i| + 0.5)(|VR| - |VR_i| + 0.5)}$$

### 7.2 Pseudo-Relevance Feedback

**Idea**: Assume top $k$ retrieved documents are relevant, estimate parameters from them, and re-rank.

Algorithm:
1. Do initial retrieval with RSJ weights
2. Assume top 10-20 documents are relevant ($VR$)
3. Re-estimate $p_i$ and $u_i$
4. Re-rank entire collection

Empirically effective, but can fail if initial results are poor (query drift).

### 7.3 The Problem with Binary Representation

The Binary Independence Model has a fundamental limitation: **it ignores term frequency!**

A document mentioning "information retrieval" 20 times gets the same score as one mentioning it once.

This is clearly suboptimal. We need to extend the model to incorporate term frequency while maintaining the probabilistic framework.

---

## 8. From BIM to BM25: Incorporating Term Frequency

### 8.1 The Need for Term Frequency

**Observation**: More occurrences of a term provide stronger evidence of relevance, but with **diminishing returns** (saturation).

- 1st occurrence: strong evidence
- 2nd occurrence: additional evidence, but less than the first
- 10th occurrence: weak additional evidence
- 100th occurrence: negligible additional evidence

We need a function $f(\text{tf})$ where:
- $f(0) = 0$
- $f$ is monotonically increasing
- $f$ has diminishing returns (concave)
- $f$ saturates at some maximum value

### 8.2 Empirical TF Normalization

Simple log: $f(\text{tf}) = 1 + \log \text{tf}$ (from VSM)

Problem: Not derived from probabilistic principles. Can we do better?

### 8.3 The Poisson Distribution

**Idea**: Model term counts using Poisson distributions!

The Poisson distribution models the number of events in a fixed interval:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

where $\lambda$ is the expected number of occurrences.

**Mean and variance**: Both equal to $\lambda$

**Application to IR**:
- Model term occurrences in documents as Poisson random variables
- Different $\lambda$ for relevant vs. non-relevant documents
- Use Poisson mixture model

---

## 9. The 2-Poisson Model

### 9.1 The Two-Poisson Mixture

**Harter's 2-Poisson model** (1975):

Term occurrences in a document come from one of two sources:

1. **Elite set**: Documents where the term is central to meaning
   - Poisson with parameter $\lambda_e$ (higher rate)

2. **Non-elite set**: Documents where term appears incidentally
   - Poisson with parameter $\lambda_n$ (lower rate)

### 9.2 Mixture for Relevant Documents

For a relevant document, term $i$ has term frequency $\text{tf}_i$ distributed as:

$$P(\text{tf}_i \mid R = 1) = \alpha \cdot \text{Poisson}(\text{tf}_i; \lambda_r) + (1-\alpha) \cdot \text{Poisson}(\text{tf}_i; \lambda_n)$$

where:
- $\alpha$ = probability document is in elite set
- $\lambda_r$ = rate for relevant elite documents
- $\lambda_n$ = rate for non-elite

Similarly for non-relevant documents (different $\alpha$ and $\lambda$).

### 9.3 Deriving the TF Component

Through complex mathematics (see Robertson & Walker 1994), the log-likelihood ratio for term frequencies can be approximated as:

$$\log \frac{P(\text{tf} \mid R = 1)}{P(\text{tf} \mid R = 0)} \approx c_i \cdot \frac{\text{tf} \cdot (k_1 + 1)}{\text{tf} + k_1}$$

where:
- $c_i$ is the RSJ IDF weight (from before)
- $k_1$ is a parameter related to $\lambda_r/\lambda_n$ (controls saturation)

**This is the BM25 TF saturation formula!**

### 9.4 Properties of the Saturation Function

$$g(\text{tf}) = \frac{\text{tf} \cdot (k_1 + 1)}{\text{tf} + k_1}$$

**Analysis**:

At $\text{tf} = 0$:
$$g(0) = 0$$

As $\text{tf} \to \infty$:
$$\lim_{\text{tf} \to \infty} g(\text{tf}) = k_1 + 1$$

**Derivative** (rate of increase):
$$g'(\text{tf}) = \frac{k_1(k_1 + 1)}{(\text{tf} + k_1)^2}$$

Always positive (monotonic increasing) but decreasing (concave).

**Effect of $k_1$**:
- $k_1 = 0$: $g(\text{tf}) = 0$ (ignores TF completely - binary)
- $k_1 \to \infty$: $g(\text{tf}) \approx \text{tf}$ (linear TF)
- $k_1 = 1.2$ (typical): good saturation curve

---

## 10. BM25 Formula Derivation

### 10.1 Document Length Normalization

**Problem**: Longer documents naturally have higher term frequencies.

A document with 1000 words mentioning "retrieval" 5 times has lower term density than a 100-word document mentioning it 5 times.

**Solution**: Normalize TF by document length.

Define:
- $L_d$ = length of document $d$ (in words or tokens)
- $\bar{L}$ = average document length in collection

**Normalized TF**:

$$\text{tf}_{\text{norm}} = \frac{\text{tf}}{(1-b) + b \cdot \frac{L_d}{\bar{L}}}$$

where $b \in [0,1]$ controls degree of normalization:
- $b = 0$: no normalization
- $b = 1$: full linear normalization by document length
- $b = 0.75$ (typical): partial normalization

**Intuition**:

The denominator $(1-b) + b \cdot \frac{L_d}{\bar{L}}$ equals:
- 1.0 for average-length document
- $> 1.0$ for longer documents (reduces effective TF)
- $< 1.0$ for shorter documents (increases effective TF)

### 10.2 Combining TF Saturation and Length Normalization

Substitute normalized TF into the saturation formula:

$$\text{BM25-TF}(\text{tf}, L_d) = \frac{\text{tf} \cdot (k_1 + 1)}{\text{tf} + k_1 \cdot \left((1-b) + b \cdot \frac{L_d}{\bar{L}}\right)}$$

Equivalently:

$$\text{BM25-TF} = \frac{\text{tf} \cdot (k_1 + 1)}{\text{tf} + k_1 \cdot (1-b) + k_1 \cdot b \cdot \frac{L_d}{\bar{L}}}$$

### 10.3 The Complete BM25 Formula

Combining the IDF component (from BIM) with the TF component (from 2-Poisson):

$$\boxed{\text{BM25}(d, q) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \frac{\text{tf}_{t,d} \cdot (k_1 + 1)}{\text{tf}_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{L_d}{\bar{L}}\right)}}$$

where:

$$\text{IDF}(t) = \log \frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5}$$

**Parameters**:
- $k_1$: Controls TF saturation (typical range: 1.2 to 2.0)
- $b$: Controls length normalization (typical value: 0.75)

### 10.4 Query Term Frequency

Some formulations include query term frequency:

$$\text{BM25}(d, q) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \frac{\text{tf}_{t,d} \cdot (k_1 + 1)}{\text{tf}_{t,d} + k_1(\cdots)} \cdot \frac{\text{tf}_{t,q} \cdot (k_3 + 1)}{\text{tf}_{t,q} + k_3}$$

where $k_3$ controls query TF saturation (often set to 0 or infinity for binary).

In most applications, queries are short and query TF is not important, so:
$$k_3 = 0 \Rightarrow \text{query term weight} = \begin{cases} 1 & \text{if } t \in q \\ 0 & \text{otherwise} \end{cases}$$

### 10.5 Worked Example

**Collection**:
- $N = 1,000,000$ documents
- $\bar{L} = 500$ tokens

**Query**: "information retrieval"

**Document $d$**:
- Length: $L_d = 400$ tokens
- "information": $\text{tf} = 3$, $\text{df} = 50,000$
- "retrieval": $\text{tf} = 2$, $\text{df} = 20,000$

**Parameters**: $k_1 = 1.5$, $b = 0.75$

**Step 1: Compute IDF weights**

$$\text{IDF}(\text{information}) = \log \frac{1,000,000 - 50,000 + 0.5}{50,000 + 0.5} = \log \frac{950,000.5}{50,000.5} \approx \log(19) \approx 1.28$$

$$\text{IDF}(\text{retrieval}) = \log \frac{1,000,000 - 20,000 + 0.5}{20,000 + 0.5} = \log \frac{980,000.5}{20,000.5} \approx \log(49) \approx 1.69$$

**Step 2: Compute length normalization factor**

$$\text{norm} = (1 - 0.75) + 0.75 \cdot \frac{400}{500} = 0.25 + 0.75 \cdot 0.8 = 0.25 + 0.6 = 0.85$$

**Step 3: Compute TF components**

For "information":
$$\text{BM25-TF} = \frac{3 \cdot (1.5 + 1)}{3 + 1.5 \cdot 0.85} = \frac{3 \cdot 2.5}{3 + 1.275} = \frac{7.5}{4.275} \approx 1.75$$

For "retrieval":
$$\text{BM25-TF} = \frac{2 \cdot 2.5}{2 + 1.275} = \frac{5.0}{3.275} \approx 1.53$$

**Step 4: Compute final score**

$$\text{BM25}(d, q) = 1.28 \cdot 1.75 + 1.69 \cdot 1.53 = 2.24 + 2.59 = 4.83$$

---

## 11. Understanding BM25 Parameters

### 11.1 The $k_1$ Parameter (TF Saturation)

$$k_1$$ controls how quickly the TF component saturates.

**At different values**:

For $\text{tf} = 5$, $L_d = \bar{L}$ (average length), $b = 0.75$:

| $k_1$ | BM25-TF value | Interpretation |
|-------|---------------|----------------|
| 0.0 | 0 | Binary (presence/absence only) |
| 0.5 | $\frac{5 \cdot 1.5}{5 + 0.5} = 1.36$ | Strong saturation |
| 1.2 | $\frac{5 \cdot 2.2}{5 + 1.2} = 1.77$ | Moderate saturation (typical) |
| 2.0 | $\frac{5 \cdot 3.0}{5 + 2.0} = 2.14$ | Weak saturation |
| $\infty$ | 5.0 | No saturation (linear TF) |

**When to adjust**:
- **Short documents** (titles, abstracts): Lower $k_1$ (more saturation)
- **Long documents** (full text): Higher $k_1$ (less saturation)
- **Verbose text** (with repetition): Lower $k_1$
- **Diverse vocabularies**: Higher $k_1$

**Typical range**: $k_1 \in [1.2, 2.0]$

### 11.2 The $b$ Parameter (Length Normalization)

$$b$$ controls how much we penalize long documents.

**At different values**:

For $\text{tf} = 2$, $k_1 = 1.5$:

| $b$ | Short doc ($L = 100$) | Average ($L = 500$) | Long doc ($L = 2000$) |
|-----|---------------------|---------------------|---------------------|
| 0.0 | $\frac{2 \cdot 2.5}{2 + 1.5} = 1.43$ | $1.43$ | $1.43$ (same!) |
| 0.5 | $\frac{5}{2 + 1.5 \cdot 0.9} = 1.53$ | $1.43$ | $\frac{5}{2 + 1.5 \cdot 3.5} = 0.70$ |
| 0.75 | $\frac{5}{2 + 1.5 \cdot 0.85} = 1.56$ | $1.43$ | $\frac{5}{2 + 1.5 \cdot 3.25} = 0.67$ |
| 1.0 | $\frac{5}{2 + 1.5 \cdot 0.8} = 1.61$ | $1.43$ | $\frac{5}{2 + 1.5 \cdot 4.0} = 0.63$ |

**When to adjust**:
- **Similar-length documents**: Lower $b$ (less normalization needed)
- **Highly variable lengths**: Higher $b$ (more normalization)
- **Verbose collections**: Higher $b$ (penalize length more)
- **Controlled-length documents**: Lower $b$

**Typical range**: $b \in [0.5, 1.0]$, most commonly $b = 0.75$

### 11.3 Why $b \neq 1$ Usually Optimal?

**Two reasons for document length**:

1. **Verbosity**: Document repeats content without adding information
   - Should be penalized → favor higher $b$

2. **Scope**: Document covers more topics with more information
   - Should NOT be heavily penalized → favor lower $b$

Setting $b = 0.75$ is a compromise that partially normalizes for verbosity while not fully penalizing scope.

### 11.4 Tuning BM25 Parameters

**Grid search**:
```python
for k1 in [0.5, 1.0, 1.5, 2.0, 2.5]:
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        evaluate_on_test_set(k1, b)
```

**Practical advice**:
- Start with defaults: $k_1 = 1.2$, $b = 0.75$
- If performance is good, don't over-optimize
- If tuning: use validation set, not test set
- TREC experiments suggest defaults work across many collections

---

## 12. BM25 Extensions and Variants

### 12.1 BM25+ (Lv & Zhai, 2011)

**Problem**: Standard BM25 can assign very low scores to long documents.

For very long documents, the length normalization term $k_1 \cdot b \cdot \frac{L_d}{\bar{L}}$ becomes very large, making the TF component approach 0.

**Solution**: Add a small constant $\delta$:

$$\text{BM25+}(d, q) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \left(\delta + \frac{\text{tf}_{t,d} \cdot (k_1 + 1)}{\text{tf}_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{L_d}{\bar{L}}\right)}\right)$$

**Typical value**: $\delta = 1.0$

**Effect**: Every matching term contributes at least $\delta \cdot \text{IDF}(t)$ to the score.

### 12.2 BM25F (Robertson et al., 2004)

**Problem**: Documents have structure (title, body, abstract, etc.). How to weight fields differently?

**Solution**: Field-specific parameters and weights.

$$\text{BM25F}(d, q) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \frac{\tilde{\text{tf}}_{t,d} \cdot (k_1 + 1)}{\tilde{\text{tf}}_{t,d} + k_1}$$

where **weighted TF** is:

$$\tilde{\text{tf}}_{t,d} = \sum_{f \in \text{fields}} w_f \cdot \frac{\text{tf}_{t,d,f}}{(1 - b_f) + b_f \cdot \frac{L_{d,f}}{\bar{L}_f}}$$

**Parameters**:
- $w_f$: Weight for field $f$ (e.g., $w_{\text{title}} = 3$, $w_{\text{body}} = 1$)
- $b_f$: Length normalization for field $f$ (can vary by field)
- $L_{d,f}$: Length of field $f$ in document $d$
- $\bar{L}_f$: Average length of field $f$ across collection

**Example** (title + body):

$$\tilde{\text{tf}}_t = 3 \cdot \frac{\text{tf}_{\text{title}}}{0.25 + 0.75 \cdot \frac{L_{\text{title}}}{\bar{L}_{\text{title}}}} + 1 \cdot \frac{\text{tf}_{\text{body}}}{0.25 + 0.75 \cdot \frac{L_{\text{body}}}{\bar{L}_{\text{body}}}}$$

**Use cases**:
- Web search (title, URL, anchor text have different importance)
- Academic search (title, abstract, body)
- Product search (title, description, reviews)

### 12.3 BM25L (Language Model Variant)

Combines BM25 with language modeling smoothing:

$$\text{BM25L}(d, q) = \sum_{t \in q \cap d} \log(1 + \text{IDF}(t)) \cdot \frac{\text{tf}_{t,d} + \delta}{...}$$

Less commonly used than standard BM25.

### 12.4 BM25-Adpt (Adaptive BM25)

Uses document-specific $k_1$ and $b$ parameters learned from features:

$$k_1^{(d)} = f_1(\text{features}(d))$$
$$b^{(d)} = f_2(\text{features}(d))$$

Requires training data but can improve performance.

---

## 13. Comparison with Vector Space Models

### 13.1 Similarities

Both BM25 and TF-IDF VSM:
- Sum contributions from matching query terms
- Weight terms by inverse document frequency
- Use sublinear term frequency
- Can be computed efficiently via inverted index

### 13.2 Differences

| Aspect | TF-IDF VSM | BM25 |
|--------|-----------|------|
| **Foundation** | Heuristic / geometric | Probabilistic |
| **TF saturation** | $1 + \log \text{tf}$ | $\frac{\text{tf}(k_1+1)}{\text{tf}+k_1(\cdots)}$ |
| **IDF** | $\log \frac{N}{\text{df}}$ | $\log \frac{N-\text{df}+0.5}{\text{df}+0.5}$ |
| **Length norm** | Cosine (all terms) | Integrated into TF (query terms only) |
| **Negative weights** | Rare | Common for very frequent terms |
| **Parameters** | Few or none | $k_1$, $b$ (tunable) |
| **Theoretical justification** | Weak | Strong (probabilistic framework) |

### 13.3 Empirical Performance

**Typical findings**:
- BM25 outperforms TF-IDF on most TREC collections
- Margin varies by collection (sometimes small, sometimes large)
- BM25 more robust across different collection types
- TF-IDF sometimes competitive when well-tuned

**Why BM25 wins**:
1. Better TF saturation (derived from 2-Poisson model)
2. Better length normalization (field-specific in BM25F)
3. Tuneable parameters (can adapt to collection)
4. Probabilistic IDF handles extreme df values better

### 13.4 Cosine Normalization vs BM25 Length Normalization

**VSM Cosine**:

$$\text{score} = \frac{\sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}}{\sqrt{\sum_{t \in d} w_{t,d}^2}}$$

Normalizes by **all** terms in document (even those not in query).

**BM25**:

Length normalization only affects **query terms**:

$$\text{score} = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \frac{\text{tf}_{t,d}}{\text{tf}_{t,d} + k_1 \cdot b \cdot L_d/\bar{L}}$$

**Consequence**: BM25 length norm is "softer" and more appropriate for retrieval.

---

## 14. Practical Implementation Considerations

### 14.1 Efficient Computation

BM25 can be computed efficiently using an inverted index:

```
scores = {}
for each query_term t:
    idf_t = compute_idf(t)
    for each (doc_id, tf) in inverted_index[t]:
        norm_factor = 1 - b + b * (doc_lengths[doc_id] / avg_doc_length)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * norm_factor)
        scores[doc_id] += idf_t * tf_component

return top_k(scores)
```

**Time complexity**: $O\left(\sum_{t \in q} \text{df}_t\right)$ — same as Boolean retrieval.

### 14.2 Precomputation

**Precompute and store**:
- Document lengths (needed for every query)
- Average document length (global constant)
- IDF values (can precompute for all terms)

**Do NOT precompute**:
- Query-document scores (query-dependent)
- TF components (query-dependent through length norm)

### 14.3 Index Organization

**Document-at-a-time (DAAT)**:
```
for each document d:
    score[d] = 0
    for each query term t in d:
        score[d] += BM25_term(t, d)
```

**Term-at-a-time (TAAT)** (more efficient):
```
for each query term t:
    for each document d containing t:
        score[d] += BM25_term(t, d)
```

TAAT allows better disk access patterns and early termination.

### 14.4 Top-K Retrieval

**Optimization**: Don't compute scores for all documents.

**Techniques**:
1. **Max score**: Upper bound score for each query term, prune documents that can't make top-K
2. **WAND**: Dynamic pruning using maximum possible scores
3. **Block-max indexing**: Store maximum TF and maximum IDF in index blocks

These can provide 10-100× speedups for retrieving top-10 from millions of documents.

### 14.5 Handling Edge Cases

**Very common terms** ($\text{df} > N/2$):
- IDF becomes negative
- Option 1: Floor IDF at 0
- Option 2: Use negative IDF (term is anti-relevant)
- Most systems: floor at 0 or small positive value

**Missing statistics**:
- Unknown $\bar{L}$: estimate from sample
- Unknown $N$: use collection size or estimate
- Zero TF: shouldn't happen for matching documents

**Numerical issues**:
- Very large collections: log IDF can overflow → use appropriate data types
- Very small probabilities: work in log space throughout

### 14.6 BM25 in Modern Search Systems

**Elasticsearch**:
- Default similarity function (since v5.0)
- Configurable $k_1$ and $b$ per index
- Supports BM25F for field boosting

**Apache Lucene**:
- BM25Similarity class
- Default in recent versions
- Configurable parameters

**Apache Solr**:
- Based on Lucene, same BM25 implementation
- BM25 available since Solr 6

**Academic implementations**:
- Terrier (Java): Full BM25, BM25F support
- Anserini (Java): Built on Lucene
- PyTerrier (Python): Research-friendly interface
- rank-bm25 (Python): Lightweight pure Python

---

## Summary

### Key Takeaways

1. **Probabilistic IR provides theoretical foundation**: BM25 is derived from probability theory, not heuristics

2. **The Probability Ranking Principle**: Optimal strategy is to rank by $P(R=1|d,q)$

3. **Binary Independence Model**: Makes independence assumption, allows tractable computation

4. **RSJ weights**: Derived from BIM without relevance information, equivalent to probabilistic IDF

5. **2-Poisson model**: Extends BIM to incorporate term frequency with saturation

6. **BM25 formula**: Combines probabilistic IDF with saturating TF and length normalization

7. **Parameters matter**: $k_1$ and $b$ should be tuned for your collection

8. **BM25 dominates**: State-of-the-art for sparse retrieval, beats TF-IDF empirically

9. **Extensions**: BM25+, BM25F extend the model for specific scenarios

10. **Efficient implementation**: Can be computed as efficiently as Boolean retrieval

### The BM25 Formula (One More Time)

$$\boxed{\text{BM25}(d, q) = \sum_{t \in q \cap d} \underbrace{\log \frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5}}_{\text{Probabilistic IDF from BIM}} \cdot \underbrace{\frac{\text{tf}_{t,d} \cdot (k_1 + 1)}{\text{tf}_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{L_d}{\bar{L}}\right)}}_{\text{Saturating TF with length norm from 2-Poisson}}}$$

### Further Reading

**Essential papers**:
1. Robertson & Sparck Jones (1976) - Binary Independence Model
2. Robertson & Walker (1994) - Okapi BM25
3. Robertson & Zaragoza (2009) - BM25 tutorial
4. Lv & Zhai (2011) - BM25+
5. Robertson et al. (2004) - BM25F

**Books**:
- Manning, Raghavan, Schütze - *Introduction to Information Retrieval* (Chapter 11)
- Croft, Metzler, Strohman - *Search Engines: Information Retrieval in Practice*
- Büttcher, Clarke, Cormack - *Information Retrieval: Implementing and Evaluating Search Engines*

---

## Appendix: Why VSM Limitations Led to BM25

This appendix provides additional context on the specific shortcomings of the Vector Space Model that motivated the development of BM25.

### A.1 The Six Core Limitations of VSM

#### Limitation 1: Arbitrary TF Scaling

**VSM Problem:**
- Uses `1 + log(tf)` for term frequency weighting
- Why logarithm specifically? Because it "feels right" empirically
- No theoretical or probabilistic justification

**Example:**
```
Term appears 10 times:  TF weight = 1 + log(10) = 2.0
Term appears 100 times: TF weight = 1 + log(100) = 3.0
```

Is this the *right* saturation curve? VSM provides no answer.

**BM25 Solution:**
- Derives TF saturation from the **2-Poisson mixture model** (Section 9)
- Formula: `tf * (k1 + 1) / (tf + k1)`
- Has theoretical probabilistic backing
- Includes tunable parameter k1 to adjust saturation rate for different collections

---

#### Limitation 2: Over-Aggressive Length Normalization

**VSM Problem:**
- Uses **cosine normalization**: divides by the L2 norm of the entire document vector
- This **over-penalizes long documents**

**Concrete Example:**
```
Short doc (50 words):  mentions "sanctions" 1 time → term density = 2%
Long doc (500 words):  mentions "sanctions" 8 times → term density = 1.6%

After cosine normalization:
- Short doc receives HIGHER score (more dense)
- But long doc has 8× more evidence about sanctions!
```

**The Fundamental Issue:**

Documents can be long for two distinct reasons:
1. **Verbosity**: Repeats same content unnecessarily → should be penalized
2. **Broader scope**: Covers more topics with genuine information → should NOT be heavily penalized

Cosine normalization cannot distinguish these cases. It treats all length uniformly.

**BM25 Solution:**
- Partial normalization controlled by parameter `b`
- Formula: `(1 - b) + b * (doc_length / avg_length)`
- When b = 0.75 (typical), only 75% of length difference matters
- Balances verbosity penalty against genuine content breadth

---

#### Limitation 3: Heuristic IDF Without Probabilistic Foundation

**VSM Problem:**
- IDF = `log(N / df)` works well in practice
- But WHY does it work?
- What does the logarithm represent conceptually?
- No connection to relevance probability

**BM25 Solution:**
- Derives IDF from **Binary Independence Model** (Section 4)
- IDF = `log((N - df + 0.5) / (df + 0.5))`
- This is the **Robertson-Sparck Jones weight** (Section 6)
- Represents the log-likelihood ratio of relevance given term presence
- Probabilistically grounded in Bayes' theorem

---

#### Limitation 4: Naive Combination of TF and IDF

**VSM Problem:**
```
weight(t, d) = (1 + log(tf)) * log(N / df)
```

This is just multiplication with no principled reason for this specific combination.

**BM25 Solution:**
- Derives the combination from the **likelihood ratio** of relevance (Section 5)
- The multiplicative structure emerges naturally from:
  - Log-odds of relevance (IDF component)
  - Term frequency contribution (TF component from 2-Poisson model)
- Formula follows from Bayes' theorem, not intuition

---

#### Limitation 5: No Tunable Parameters

**VSM Problem:**
- Fixed formula: `1 + log(tf)`
- Cannot adjust for different collection characteristics
- News articles, tweets, legal documents, and medical records all use identical formula
- One-size-fits-all approach

**BM25 Solution:**
- **k1**: controls TF saturation (typical range: 1.2 to 2.0)
- **b**: controls length normalization (typical range: 0.5 to 1.0)
- Can be tuned via grid search or learned from validation data
- Adapts to collection-specific characteristics

---

#### Limitation 6: Breaks Down for Extreme Document Lengths

**VSM Problem:**
- Very short documents (1-2 words): normalized to unit vector, loses signal
- Very long documents (10,000+ words): also normalized to unit vector
- The magnitude information (how much evidence exists) is completely discarded

**BM25+ Solution:**
- Adds lower bound constant `delta` to prevent zero contribution (Section 12.1)
- Even in infinitely long documents, matching terms contribute at least `delta * IDF(t)`
- Ensures every relevant term provides some score

---

### A.2 Side-by-Side Comparison

| Aspect | VSM + TF-IDF | BM25 | Winner |
|--------|-------------|------|--------|
| **Foundation** | Heuristic / geometric intuition | Probabilistic (BIM + 2-Poisson) | BM25 |
| **TF Saturation** | `1 + log(tf)` | `tf*(k1+1)/(tf+k1)` | BM25 |
| **TF Justification** | "Looks right empirically" | Derived from 2-Poisson model | BM25 |
| **Length Norm** | Cosine (all terms) | Partial (parameter b, query terms only) | BM25 |
| **IDF Formula** | `log(N/df)` | `log((N-df+0.5)/(df+0.5))` | BM25 |
| **IDF Justification** | Information theory intuition | BIM probability + RSJ weights | BM25 |
| **Handles long docs** | Over-penalizes | Graceful partial normalization | BM25 |
| **Tunable?** | No | Yes (k1, b, and variants) | BM25 |
| **Negative weights?** | Rare | Common for high-df terms (intended) | BM25 |

---

### A.3 Concrete Example: Where VSM Fails

**Query:** "russian sanctions"

**Document A (short, focused):**
"Russian oligarch under sanctions" (4 words, each term appears once)

**Document B (long, comprehensive):**
"The Russian Federation has imposed various economic sanctions on multiple entities. Russian companies have been affected by these sanctions, leading to widespread economic impact across Russian industries. Sanctions compliance remains a major challenge..." (50 words, "russian" appears 3×, "sanctions" appears 3×)

---

#### VSM + TF-IDF Scoring:

**Document A:**
- TF: `1 + log(1) = 1.0` for each term
- IDF: assume both terms have `idf ≈ 2.0`
- Raw score: `1.0 * 2.0 + 1.0 * 2.0 = 4.0`
- After cosine normalization (4 total terms): **HIGH score** (very dense)

**Document B:**
- TF: `1 + log(3) = 1.48` for each term
- IDF: same `2.0`
- Raw score: `1.48 * 2.0 + 1.48 * 2.0 = 5.92`
- After cosine normalization (50 total terms): **LOWER score** (diluted by other content)

**Result:** VSM might rank A > B, despite B containing 3× more evidence!

---

#### BM25 Scoring:

Assume `k1 = 1.2`, `b = 0.75`, `avg_length = 30`

**Document A:**
- Length norm: `(1 - 0.75) + 0.75 * (4/30) = 0.25 + 0.1 = 0.35`
- TF component for each term: `1 * 2.2 / (1 + 1.2 * 0.35) = 2.2 / 1.42 ≈ 1.55`
- Score: `2.0 * 1.55 + 2.0 * 1.55 = 6.2`

**Document B:**
- Length norm: `0.25 + 0.75 * (50/30) = 0.25 + 1.25 = 1.5`
- TF component for each term: `3 * 2.2 / (3 + 1.2 * 1.5) = 6.6 / 4.8 ≈ 1.375`
- Score: `2.0 * 1.375 + 2.0 * 1.375 = 5.5`

**Result:** BM25 ranks A > B by a small margin. The longer document is not over-penalized, and if it had tf=4 instead of 3, it would rank higher. The ranking is more nuanced and sensitive to actual evidence.

---

### A.4 The Historical Development

**1970s: VSM Era**
- Salton develops Vector Space Model
- TF-IDF emerges from empirical experimentation
- Works remarkably well for the time
- But foundation is intuitive, not rigorous

**1976: Probabilistic Turn**
- Robertson & Sparck Jones publish Binary Independence Model
- Introduces probabilistic framework
- Derives RSJ weights (early form of BM25 IDF)
- But still binary representation (no term frequency)

**1990s: TREC Experiments**
- Text REtrieval Conference provides standard test collections
- Okapi team (Robertson, Walker, et al.) systematically experiments
- Develops BM25 through empirical testing + probabilistic theory
- Discovers optimal parameter ranges: k1 ∈ [1.2, 2.0], b ≈ 0.75

**2000s-Present: BM25 Dominance**
- BM25 becomes the standard baseline for IR research
- Adopted by Elasticsearch, Lucene, Solr
- Extensions: BM25+, BM25F, BM25-Adpt
- Still unbeaten by sparse methods

---

### A.5 Why BM25 is Not Just "Better TF-IDF"

It's tempting to view BM25 as an incremental improvement over TF-IDF. This misses the deeper point:

**BM25 is a fundamentally different approach:**

1. **Generative model**: Assumes documents are generated by a probabilistic process
2. **Explicit model of relevance**: Models P(relevant | document, query)
3. **Derived, not designed**: Formula emerges from probability theory
4. **Interpretable parameters**: k1 and b have clear statistical meanings
5. **Extensible**: Can incorporate relevance feedback, field weights, etc.

**TF-IDF is a heuristic scoring function:**

1. **No underlying model**: Just "reasonable" weighting choices
2. **No notion of relevance**: Geometric similarity, not probability
3. **Designed, not derived**: Someone decided log and cosine were good
4. **Fixed formula**: No principled way to adapt
5. **Hard to extend**: Adding features requires ad-hoc modifications

---

### A.6 When VSM Might Still Be Preferred

Despite BM25's theoretical and empirical superiority, VSM + TF-IDF has niches:

**Use VSM when:**
1. **Computational simplicity matters**: VSM is slightly simpler to implement from scratch
2. **Interpretability for non-experts**: "Cosine similarity" is easier to explain than "log-likelihood ratio"
3. **Document similarity tasks**: Finding similar documents (not query-document matching)
4. **Embeddings downstream**: If you're using vectors for other ML tasks
5. **Legacy systems**: Already deployed and working "well enough"

**Use BM25 when:**
1. **Building a new search system**: Start with the best baseline
2. **Production IR system**: Proven effectiveness across domains
3. **Need tunability**: Parameters let you adapt to your data
4. **Research baseline**: Required for credible IR research
5. **Multi-field documents**: BM25F handles structured data elegantly

---

### A.7 The Key Insight

**The progression from VSM to BM25 mirrors a broader pattern in computer science:**

```
Heuristic methods → Probabilistic models → Machine learning
```

- **1970s VSM**: Smart heuristics, empirically validated
- **1990s BM25**: Probabilistic foundations, theoretically grounded
- **2020s Neural IR**: Deep learning, learned representations

Each stage builds on the previous. BM25 didn't replace VSM by accident — it formalized the intuitions that made VSM work, fixed its bugs, and added theoretical rigor.

Understanding *why* BM25 improves on VSM makes you a better IR practitioner. You'll know:
- When to tune k1 vs. b
- Why field boosting (BM25F) matters
- How to interpret BM25 scores
- When BM25 assumptions break down

---

**End of Appendix**

---

**End of Lecture**
