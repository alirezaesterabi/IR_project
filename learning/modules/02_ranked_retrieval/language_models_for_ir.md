# Language Models for Information Retrieval

**Module 2 — Extended notes (query likelihood, smoothing, connection to BM25)**

This document expands **Part 7** of [theory.md](./theory.md). It follows the narrative of *Introduction to Information Retrieval* (Manning, Raghavan, and Schütze) **Chapter 12**, and aligns with the classical IR thread in [week 3 lab materials](../../raw_materials/week_3/) (Boolean → VSM → TF‑IDF → BM25) and **week 3 slides** (`slide.pdf`), where language models are usually introduced after lexical scoring functions.

**Companion reading:** [detailed_probabilistic_ir_bm25.md](./detailed_probabilistic_ir_bm25.md) (BM25 derivation) — LM and BM25 are different parameterisations of related ideas (see §9).

---

## Table of contents

1. [Why language models in IR?](#1-why-language-models-in-ir)
2. [The query likelihood retrieval model](#2-the-query-likelihood-retrieval-model)
3. [Multinomial (unigram) language models](#3-multinomial-unigram-language-models)
4. [Maximum likelihood estimation and zero probability](#4-maximum-likelihood-estimation-and-zero-probability)
5. [The collection language model](#5-the-collection-language-model)
6. [Smoothing: Jelinek–Mercer](#6-smoothing-jelinekmercer)
7. [Smoothing: Dirichlet (Bayesian)](#7-smoothing-dirichlet-bayesian)
8. [Other ranking views (brief)](#8-other-ranking-views-brief)
9. [Relationship to BM25](#9-relationship-to-bm25)
10. [Worked example](#10-worked-example)
11. [Practical notes and OpenSanctions](#11-practical-notes-and-opensanctions)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Why language models in IR?

**Vector space models** represent documents and queries as vectors and score by geometric similarity (e.g. cosine). **Probabilistic relevance** models (BIM → BM25) try to estimate $P(\text{relevant} \mid d, q)$ under explicit independence assumptions.

**Language models for IR** take a **generative** view:

> Pick a document $d$. It implies a probability distribution over terms — a **language model** $\theta_d$. How likely is it that a user who “had this document in mind” would type query $q$?

Documents that assign **high probability** to the query terms are ranked higher. This is **query likelihood** retrieval (Ponte & Croft, 1998; refined in later work).

**Advantages:**

- Principled handling of uncertainty via smoothing (no hard zero probabilities after smoothing).
- Close connection to **statistical NLP** and speech (n-gram LMs).
- Strong empirical performance when smoothing is chosen well (often **Dirichlet**).

**Not** the same as **large language models (LLMs)** used in modern RAG: here “language model” means a **small, classical** distribution (typically **unigram**) over terms estimated per document.

---

## 2. The query likelihood retrieval model

Let $q = (t_1, t_2, \ldots, t_{n_q})$ be a query as a sequence of $n_q$ terms (after the same tokenisation as the index). Let $\theta_d$ denote the language model associated with document $d$.

The **query likelihood** score is:

$$
\text{score}(d, q) = P(q \mid \theta_d)
$$

Under the usual **unigram** assumption, terms are generated **independently** given $\theta_d$ (bag-of-words at generation time):

$$
P(q \mid \theta_d) = \prod_{i=1}^{n_q} P(t_i \mid \theta_d) = \prod_{t \in \mathcal{V}} P(t \mid \theta_d)^{\text{tf}_{t,q}}
$$

where $\mathcal{V}$ is the vocabulary and $\text{tf}_{t,q}$ is the count of term $t$ in $q$.

**Log score** (monotonic, numerically stable):

$$
\log P(q \mid \theta_d) = \sum_{t \in \mathcal{V}} \text{tf}_{t,q} \cdot \log P(t \mid \theta_d) = \sum_{i=1}^{n_q} \log P(t_i \mid \theta_d)
$$

**Ranking:** sort documents by $\log P(q \mid \theta_d)$ in **descending** order.

---

## 3. Multinomial (unigram) language models

For each document $d$, treat it as a sample from a **multinomial** over the vocabulary with parameters $\theta_d$. The **unigram** model is:

$$
\sum_{t \in \mathcal{V}} P(t \mid \theta_d) = 1, \qquad P(t \mid \theta_d) \geq 0
$$

Only **unigrams** are used in the standard IR formulation (Chapter 12 of IIR); bigrams or trigrams are possible extensions but explode vocabulary and sparsity.

---

## 4. Maximum likelihood estimation and zero probability

A **maximum likelihood estimate (MLE)** of the document model from the document itself treats the document as a sequence of $L_d$ tokens (length $L_d$):

$$
P_{\text{MLE}}(t \mid d) = \frac{\text{tf}_{t,d}}{L_d}
$$

where $\text{tf}_{t,d}$ is the within-document term frequency.

**Zero-probability problem:** If $t$ appears in the query but $\text{tf}_{t,d} = 0$, then $P_{\text{MLE}}(t \mid d) = 0$, so $P(q \mid \theta_d) = 0$ for any such query. One missing term **zeros out** the entire score.

**Smoothing** fixes this by **borrowing** probability mass from the rest of the collection (or from a prior), so every query term has **strictly positive** probability under every document model.

---

## 5. The collection language model

Let $\mathcal{C}$ denote the **whole collection**. The **collection language model** is a unigram distribution:

$$
P(t \mid \mathcal{C}) = \frac{\text{cf}_t}{\sum_{t' \in \mathcal{V}} \text{cf}_{t'}}
$$

where $\text{cf}_t$ is the **total count** of term $t$ across all documents (collection frequency). The denominator is the **total number of token positions** in the collection.

**Implementation note:** Store $\log P(t \mid \mathcal{C})$ for frequent query terms; use log sums for scores. For terms never seen in the collection, use a floor probability or skip (they are rare with a closed vocabulary).

---

## 6. Smoothing: Jelinek–Mercer

**Jelinek–Mercer (JM)** smoothing uses **linear interpolation** between the document MLE and the collection model:

$$
P_{\text{JM}}(t \mid d) = (1 - \lambda) \cdot \frac{\text{tf}_{t,d}}{L_d} + \lambda \cdot P(t \mid \mathcal{C})
$$

with **interpolation parameter** $\lambda \in [0, 1]$. Larger $\lambda$ pulls estimates toward the collection (more smoothing for rare or missing terms).

**Query log-likelihood:**

$$
\log P(q \mid d) = \sum_{t \in \mathcal{V}} \text{tf}_{t,q} \cdot \log\left( (1-\lambda) \cdot \frac{\text{tf}_{t,d}}{L_d} + \lambda \cdot P(t \mid \mathcal{C}) \right)
$$

Only terms with $\text{tf}_{t,q} > 0$ contribute.

**Typical values:** $\lambda$ in roughly $[0.1, 0.5]$; tune on development data. Very **short queries** often benefit from **more** smoothing (higher $\lambda$) because the document MLE is peaky.

**Behaviour:** JM uses a **fixed** mix $\lambda$ for all documents regardless of length. **Short documents** and **long documents** get the same interpolation weight, which is not always ideal — **Dirichlet** addresses this differently.

---

## 7. Smoothing: Dirichlet (Bayesian)

**Dirichlet smoothing** corresponds to a **Dirichlet prior** on the multinomial parameters. A common closed form for the **smoothed** term probability is:

$$
P_{\text{Dir}}(t \mid d) = \frac{\text{tf}_{t,d} + \mu \, P(t \mid \mathcal{C})}{L_d + \mu}
$$

where $\mu > 0$ is a **prior strength** hyperparameter (sometimes described as pseudo-count mass from the collection).

**Interpretation:** Imagine adding $\mu$ “pseudo-tokens” distributed according to $P(t \mid \mathcal{C})$ to the document. Long documents ($L_d \gg \mu$) stay close to the MLE; short documents rely more on the collection.

**Query log-likelihood (standard form):**

$$
\log P(q \mid d) = \sum_{t \in \mathcal{V}} \text{tf}_{t,q} \cdot \log \frac{\text{tf}_{t,d} + \mu \, P(t \mid \mathcal{C})}{L_d + \mu}
$$

Again, only terms appearing in the query need to be evaluated.

**Typical values:** $\mu$ often in **hundreds to a few thousand** (e.g. 500–2000), **collection-dependent**.

**JM vs Dirichlet (rule of thumb):** Dirichlet’s smoothing amount **adapts to document length**; JM’s is **fixed**. Many TREC-style ad hoc tasks favoured **Dirichlet** when tuned well.

---

## 8. Other ranking views (brief)

Chapter 12 of IIR also discusses related formulations:

- **Document likelihood** $P(d \mid q)$ — less common as a direct ranking score; Bayes’ rule relates it to $P(q \mid d)$ with a document prior $P(d)$.
- **KL-divergence** between a **query language model** $\theta_q$ and the document model $\theta_d$:

$$
D_{\mathrm{KL}}(\theta_q \,\|\, \theta_d) = \sum_t P(t \mid \theta_q) \log \frac{P(t \mid \theta_q)}{P(t \mid \theta_d)}
$$

  With a suitable estimate of $\theta_q$ from the query (and smoothing), **minimising** KL (or maximising **negative** KL) is related to query likelihood and gives a **symmetric** view of “how well the document explains the query model.”

For this module, **master query likelihood + JM + Dirichlet** first; treat KL as optional reading in IIR §12.

---

## 9. Relationship to BM25

BM25 is not derived as a language model, but **Zhai & Lafferty (2004)** analyse smoothing and show **qualitative parallels** between **Dirichlet-smoothed** query likelihood and **BM25-like** weighting:

- A **saturating** function of $\text{tf}_{t,d}$ appears in both frameworks.
- Terms that are rare in the collection (small $P(t \mid \mathcal{C})$) behave similarly to **high IDF** terms in BM25 when scores are written in log form.

A useful **intuition** (not a literal equality): Dirichlet smoothing’s term $\frac{\text{tf}_{t,d}}{\text{tf}_{t,d} + \mu}$ resembles BM25’s TF saturation $\frac{\text{tf}}{\text{tf} + k_1(1 - b + b\, L_d/\bar{L})}$ in simplified settings.

**Practical takeaway:** BM25 remains the default **sparse** ranker in many systems; LM retrieval is an **alternative** with different tuning knobs ($\lambda$ or $\mu$). Hybrid and **linear fusion** of BM25 + LM scores appear in research systems.

Full BM25 mathematics: [detailed_probabilistic_ir_bm25.md](./detailed_probabilistic_ir_bm25.md).

---

## 10. Worked example

**Collection:** three documents (tokens lowercased, no stopword removal for simplicity).

| Doc | Tokens | $L_d$ |
|-----|--------|-------|
| $d_1$ | [ sanctions, vessel, russian ] | 3 |
| $d_2$ | [ sanctions, company, vessel, vessel ] | 4 |
| $d_3$ | [ oil, company, sanctions ] | 3 |

**Collection frequencies** (total tokens = 10):

| Term | $\text{cf}_t$ | $P(t \mid \mathcal{C})$ |
|------|---------------|-------------------------|
| sanctions | 3 | 0.3 |
| vessel | 3 | 0.3 |
| russian | 1 | 0.1 |
| company | 2 | 0.2 |
| oil | 1 | 0.1 |

**Query:** $q =$ `sanctions vessel` → $\text{tf}_{\text{sanctions},q} = 1$, $\text{tf}_{\text{vessel},q} = 1$, $n_q = 2$.

**MLE** for $d_1$: $P(\text{vessel}\mid d_1) = 1/3$, $P(\text{sanctions}\mid d_1) = 1/3$ → $\log P(q\mid d_1) = \log(1/3) + \log(1/3) \approx -2.197$.

**Dirichlet** with $\mu = 6$ for $d_1$ ($L_{d_1}=3$):

$$
P_{\text{Dir}}(\text{sanctions}\mid d_1) = \frac{1 + 6 \cdot 0.3}{3 + 6} = \frac{2.8}{9}, \quad
P_{\text{Dir}}(\text{vessel}\mid d_1) = \frac{1 + 6 \cdot 0.3}{9} = \frac{2.8}{9}
$$

$$
\log P(q\mid d_1) = 2 \log(2.8/9) \approx -2.303
$$

For $d_2$ (vessel count 2):

$$
P_{\text{Dir}}(\text{sanctions}\mid d_2) = \frac{1 + 1.8}{4+6} = \frac{2.8}{10}, \quad
P_{\text{Dir}}(\text{vessel}\mid d_2) = \frac{2 + 1.8}{10} = \frac{3.8}{10}
$$

$$
\log P(q\mid d_2) \approx \log(0.28) + \log(0.38) \approx -2.218
$$

So **$d_2$ ranks above $d_1$** under this Dirichlet model (higher vessel frequency helps). Recompute **JM** with $\lambda = 0.4$ as an exercise (Exercise 2).

---

## 11. Practical notes and OpenSanctions

- **Query preprocessing** should **match** the index (tokenisation, normalisation). Your project uses lemmatised `text_blob` / `tokens` for classical retrieval — LM scores should use the **same** representation.
- **Short entities** (many OpenSanctions records) have **small** $L_d$: Dirichlet smoothing pulls scores toward the collection; tune $\mu$ on a dev query set if you implement LM.
- LM does **not** replace **exact identifiers** (Type 1) or structured filters (Type 6); it is in the same **lexical** family as BM25 for descriptive queries (Types 2–3).

---

## 12. Exercises

1. **Pen and paper:** For the worked example, compute $\log P(q\mid d_3)$ with Dirichlet $\mu=6$ and rank all three documents.
2. **JM:** With $\lambda = 0.4$, compute $P_{\text{JM}}(t\mid d_1)$ for $t \in \{\text{sanctions}, \text{vessel}\}$ and $\log P(q\mid d_1)$.
3. **Zero probability:** Show that if one query term has MLE probability 0 in $d$, then $P(q\mid d)=0$ without smoothing.
4. **Implementation:** On the Lab 2 toy `documents` list (week 3 notebook), build collection counts, implement Dirichlet query likelihood for a one-word and two-word query, and rank documents.
5. **Concept:** Explain in two sentences why Dirichlet smoothing depends on $L_d$ but JM smoothing (fixed $\lambda$) does not.
6. **BM25 link:** Read Zhai & Lafferty (2004) abstract and summarise one finding about Dirichlet vs JM in one paragraph.

---

## 13. References

- Manning, C. D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*. **Chapter 12: Language models for information retrieval.** Cambridge University Press, 2008. ([online](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf))
- Ponte, J. M., & Croft, W. B. (1998). A language modeling approach to information retrieval. *SIGIR.*
- Zhai, C., & Lafferty, J. (2004). A study of smoothing methods for language models applied to ad hoc information retrieval. *ACM Transactions on Information Systems*, 22(2), 179–214.
- Hiemstra, D. (2001). Using language models for information retrieval. *PhD thesis*, University of Twente.
- Course materials: [week 3 slides](../../raw_materials/week_3/slide.pdf), [Lab 2 — Classical IR models](../../raw_materials/week_3/Lab%202%20Classifical%20IR%20Models.ipynb).

If you keep a local copy of **`intro_to_ir.pdf`** under `learning/raw_materials/`, align notation (e.g. $\mu$, $\lambda$, $P(t\mid\mathcal{C})$) with your edition when studying.

---

**End of language models notes** — Return to [theory.md](./theory.md) Part 8 for model comparison and module exercises.
