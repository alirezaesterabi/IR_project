# Module 3: Evaluation of Information Retrieval Systems

**Duration:** 4–5 days  
**Primary text:** Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, **Chapter 8** (Evaluation in information retrieval). Align notation with your copy of [intro_to_ir.pdf](../../raw_materials/intro_to_ir.pdf) if present locally.

**Course hands-on:** [Week 4 — Lab 3: Evaluation and Interface](../../raw_materials/week_4/Lab3_Evaluation%20and%20Interface.ipynb) (precision@k, recall@k, F1, nDCG on a small annotated collection).

**Prerequisite:** Classical ranking models from [Week 3](../../raw_materials/week_3/) and [Module 2: Ranked Retrieval](../02_ranked_retrieval/theory.md) — you evaluate *systems* that produce ranked lists.

**Project alignment:** [documents/implementation_phases.md](../../../documents/implementation_phases.md) (Phase 4–6), [documents/project_structure.md](../../../documents/project_structure.md), and **Assignment 1** design ([documents/assignment_1.pdf](../../../documents/assignment_1.pdf)) — especially §3.4 Evaluation Strategy.

---

## Table of contents

1. [Why evaluation matters](#1-why-evaluation-matters)
2. [Relevance judgements and qrels](#2-relevance-judgements-and-qrels)
3. [Unranked evaluation: precision, recall, F1](#3-unranked-evaluation-precision-recall-f1)
4. [Ranked evaluation: Precision@k, Recall@k, R-precision](#4-ranked-evaluation-precisionk-recallk-r-precision)
5. [Mean Reciprocal Rank (MRR)](#5-mean-reciprocal-rank-mrr)
6. [Average precision and MAP](#6-average-precision-and-map)
7. [Graded relevance: DCG and nDCG](#7-graded-relevance-dcg-and-ndcg)
8. [Evaluation methodology: splits, pooling, bias](#8-evaluation-methodology-splits-pooling-bias)
9. [User-oriented measures and significance (overview)](#9-user-oriented-measures-and-significance-overview)
10. [This project: TREC-style runs, qrels JSON, ranx](#10-this-project-trec-style-runs-qrels-json-ranx)
11. [Recommended metrics by query type (OpenSanctions)](#11-recommended-metrics-by-query-type-opensanctions)
12. [References and further reading](#12-references-and-further-reading)

---

## 1. Why evaluation matters

Building a retriever (Boolean, TF–IDF, BM25, dense, …) is not enough: we need **measurable** answers to:

- Is system A better than system B for our users?
- Did a parameter change or new feature **improve** ranking?
- Are we trading precision for recall in an acceptable way?

IR evaluation is **task- and user-dependent**. For sanctions search, the design document stresses **high recall**: missing a relevant sanctioned entity can carry **regulatory cost**; that shapes which metrics you emphasise at **system** level vs **query-type** level (see §11).

---

## 2. Relevance judgements and qrels

**Relevance** is typically defined by **assessors** against an **information need** (often represented by a query).

- **Binary relevance:** document is relevant (1) or non-relevant (0).
- **Graded relevance:** e.g. 0 = not relevant, 1 = somewhat, 2 = highly relevant — needed for **nDCG** and finer-grained diagnosis.

**Qrels (query–relevance file):** for each query id, a mapping from document id to relevance label (or grade).

In TREC-style benchmarks, qrels are built from **pooling** (see §8). In this project, Types 1, 2, 5, 6 use **automatic** qrels where possible; Types 3, 4 use **pooled manual** judgements; Type 7 uses **generation** metrics (see §11).

---

## 3. Unranked evaluation: precision, recall, F1

For a **fixed retrieved set** (or the whole answer set treated as unordered):

|                    | **Relevant** | **Non-relevant** |
|--------------------|-------------|------------------|
| **Retrieved**      | TP          | FP               |
| **Not retrieved**  | FN          | TN               |

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1** (harmonic mean, balances precision and recall):

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Interpretation:** Precision answers “what fraction of what we returned is good?” Recall answers “what fraction of all relevant items did we return?” For large collections, recall is often computed **up to depth k** using qrels (see §4).

---

## 4. Ranked evaluation: Precision@k, Recall@k, R-precision

For a **ranked list** $d_1, d_2, \ldots$, consider only the **top k** positions.

Let $R$ = total number of relevant documents for the query in the qrels (for binary qrels).

**Precision@k** — fraction of the top-$k$ that are relevant:

$$
P@k = \frac{1}{k} \sum_{i=1}^{k} \mathbb{1}[\text{rel}(d_i)]
$$

**Recall@k** — fraction of **all** relevant documents that appear in the top-$k$:

$$
R@k = \frac{1}{R} \sum_{i=1}^{k} \mathbb{1}[\text{rel}(d_i)]
$$

(If $R=0$, recall is usually undefined or defined as 0 — implement consistently.)

**R-precision:** precision at rank $R$ (if you retrieve $R$ documents, how many are relevant?). Useful when the number of true relevant documents is known and small.

**Cut-offs:** $k$ is chosen to match **user patience** or **UI** (e.g. first page = 10). This project uses **Recall@10** and **Recall@20** as organisational metrics where **completeness** matters ([implementation_phases.md](../../../documents/implementation_phases.md)).

---

## 5. Mean Reciprocal Rank (MRR)

For a single query, the **reciprocal rank** is $1/r$ where $r$ is the rank of the **first** relevant document (if none, 0).

**MRR** averages over queries:

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{r_q}
$$

**When it helps:** **one** clearly correct answer (e.g. “the” entity for an exact identifier) — complements **Precision@1** (which is 1 if rank 1 is relevant, else 0).

---

## 6. Average precision and MAP

**Average precision (AP)** summarises a **full ranked list** for one query (binary relevance):

$$
\text{AP} = \frac{1}{R} \sum_{k=1}^{n} P@k \cdot \mathbb{1}[\text{rel at rank } k]
$$

Equivalently: average of $P@k$ taken at each rank $k$ where the $k$-th document is relevant, then averaged over the $R$ relevant docs (careful with definitions when $R=0$).

**MAP (mean average precision):** mean of AP over all queries.

**Properties:** Rewards **early** placement of **all** relevant documents — stricter than a single $P@1$ or $R@10$.

**In this project:** MAP is used as a **secondary aggregate** across query types ([assignment_1.pdf](../../../documents/assignment_1.pdf) §3.4.3).

---

## 7. Graded relevance: DCG and nDCG

When judgements are **ordinal** (e.g. 0, 1, 2), **DCG** (discounted cumulative gain) at position $p$:

$$
\text{DCG}_p = \sum_{i=1}^{p} \frac{2^{g_i} - 1}{\log_2(i+1)}
$$

where $g_i$ is the **gain** (relevance grade) of the document at rank $i$. (Other gain/discount conventions exist; be consistent in code.)

**Ideal DCG (IDCG):** DCG computed on the **best possible** ordering for that query.

**nDCG@p** (normalised DCG):

$$
\text{nDCG}_p = \frac{\text{DCG}_p}{\text{IDCG}_p}
$$

**When to use:** **Pooled** queries with **graded** labels (Types 3, 4 in this project) — captures that rank 2 “highly relevant” should beat rank 2 “somewhat relevant.” **nDCG@10** is the agreed **primary** metric for those types in Assignment 1.

---

## 8. Evaluation methodology: splits, pooling, bias

**Train vs test:** Do not tune on the same queries you report as final scores. Hold out a **test** query set for unbiased comparison.

**Pooling (TREC-style):** For large collections, judges cannot label every document for every query. **Pool** = union of top-$m$ results from several **runs** (e.g. BM25, TF–IDF, dense). Assess only inside the pool; documents **never pooled** are often treated as **non-relevant** (optimistic bias possible for systems that retrieve “odd” relevant docs outside the pool — a known limitation).

**This project:** Types 3 and 4 — merge top results from multiple retrievers, manual **3-point** grades, see [notebooks/04_pooling_type_3_4.ipynb](../../../notebooks/04_pooling_type_3_4.ipynb) and Phase 4 in [implementation_phases.md](../../../documents/implementation_phases.md).

---

## 9. User-oriented measures and significance (overview)

**Beyond batch metrics:** IIR also discusses **interactive** evaluation: time to complete task, clicks, user satisfaction — important for deployment, not always reducible to P@k.

**Statistical significance:** When comparing two systems on the same queries, small MAP differences may be **noise**. Paired tests (bootstrap, sign test, t-test on per-query differences) are used in IR papers; a full treatment is outside this note — consult IIR §8 and specialised methodology texts when writing a rigorous report.

---

## 10. This project: TREC-style runs, qrels JSON, ranx

**Qrels in-repo:** JSON under `data/qrels/` (e.g. `qrels_type_1.json`, …) — query id → `{ doc_id: relevance }` (binary or integer grades per your convention).

**Runs:** Ranked lists per query (TREC “run” format or equivalent) can be evaluated with **[ranx](https://github.com/AmenRa/ranx)** in Python — as specified in [implementation_phases.md](../../../documents/implementation_phases.md) Phase 6.

**Planned code:** `src/evaluation/metrics.py` (see [project_structure.md](../../../documents/project_structure.md)) — centralise metric computation so notebooks and scripts stay consistent.

---

## 11. Recommended metrics by query type (OpenSanctions)

This section matches **Assignment 1 — §3.4 Evaluation Strategy** ([documents/assignment_1.pdf](../../../documents/assignment_1.pdf)) and the **ground-truth construction** described there. **System-level primary metric:** **Recall@10** — regulatory cost of **missing** a sanctioned entity.

### Summary table (design document)

| Query types | Ground truth | Primary metric(s) | Secondary metric(s) |
|-------------|--------------|-------------------|---------------------|
| **1, 2** | Auto-extracted | **Precision@1** | **MAP** |
| **5, 6** | Auto-extracted | **Recall@10**, **Recall@20** | **MAP** |
| **3, 4** | Pooling (manual, graded 0/1/2) | **nDCG@10** | **MAP**, **Recall@10** |
| **7** | RAG / generation (not ranked IR) | **Faithfulness** (RAGAS) | **Answer relevance** (RAGAS), **expert review** |

### Rationale by type

- **Type 1 (Exact identifier):** One **objective** target entity (e.g. IMO). Success = correct hit at **rank 1** → **Precision@1**; **MRR** is also informative. **MAP** aggregates with other queries.
- **Type 2 (Name / alias):** Still often a **single** operational “hit” for screening, but aliases can yield **multiple** defensible relevants — Assignment 1 groups Type 2 with Type 1 for **P@1** + **MAP**; consider reporting **R@10** if multiple matches matter for your analyst workflow.
- **Type 3 (Semantic / descriptive):** Graded pooling → **nDCG@10** as primary; **MAP** and **R@10** as secondary (breadth + ranking quality).
- **Type 4 (Relational / graph):** Same as Type 3 in the design table — multiple linked entities possible; graded **nDCG@10** plus **MAP** / **R@10**.
- **Type 5 (Cross-dataset deduplication):** Need **all** records for the same real-world entity → **recall-oriented** cut-offs (**R@10**, **R@20**) primary; **MAP** secondary.
- **Type 6 (Jurisdiction / filter):** Often a **subset** of many relevant entities — same grouping as Type 5 in Assignment 1 (**R@10**, **R@20**, **MAP**). Choose $k$ to match how many results an analyst can review.
- **Type 7 (RAG summarisation):** **Not** standard ad-hoc retrieval metrics on a ranked list. Assignment 1 specifies **RAGAS** **faithfulness** (grounding in retrieved evidence) and **answer relevance**, plus **qualitative expert review**. See also RAGAS citations in the assignment bibliography.

### Optional extras (not replacing the table above)

- **MRR** for Types 1–2 when the first relevant rank is the main operational question.
- **Success@k** (any relevant in top-$k$) as a simple binary usability metric for screening.

---

## 12. References and further reading

- Manning, C. D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval.* **Chapter 8.** [Online](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf)
- Course lab: [Lab 3 — Evaluation and Interface](../../raw_materials/week_4/Lab3_Evaluation%20and%20Interface.ipynb)
- Project: [implementation_phases.md](../../../documents/implementation_phases.md), [assignment_1.pdf](../../../documents/assignment_1.pdf) §3.4
- Pooling: TREC methodology (e.g. Voorhees & Harman, *TREC: Experiment and Evaluation in Information Retrieval*)
- RAG evaluation: RAGAS framework (cited in Assignment 1)

---

**End of Module 3 Theory** — Next: [Module 4 — Dense retrieval](../OVERVIEW.md) (overview) or apply metrics in the project evaluation notebook once `src/evaluation/metrics.py` exists.
