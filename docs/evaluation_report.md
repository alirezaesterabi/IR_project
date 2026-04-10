# Types 1–6 Evaluation Report

**DATE:** 2026-04-10
**SAMPLE:** Full dataset
**Systems evaluated:** bm25, dense_bge_m3, dense_minilm, rrf_bge_m3, rrf_minilm

Metric policy: `TYPE_METRICS` / `OVERALL_METRICS` in `src/evaluation/utils.py`.

---

## 1. Overall pipeline performance

| System | MAP | Recall@10 |
|--------|-----|-----------|
| BM25 | 0.2269 | 0.2278 |
| RRF (BGE-M3) | 0.2225 | 0.2265 |

---

## 2. Per query type — MAP (BM25 vs RRF)

| Type | Description | n | BM25 MAP | RRF MAP | RRF − BM25 |
|------|-------------|---|----------|---------|------------|
| 1 | Identifier | 50 | 0.0000 | 0.0018 | 0.0018 |
| 2 | Name search | 50 | 1.0000 | 0.9822 | -0.0178 |
| 3 | Thematic | 14 | 0.0718 | 0.0220 | -0.0498 |
| 4 | Relational | 14 | 0.0315 | 0.0218 | -0.0097 |
| 5 | Programme | 50 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Semantic | 50 | 0.0058 | 0.0182 | 0.0124 |

---

## 3. Per query type — full metrics (relevant per type)

Types 1–2: P@1, MRR, MAP. Types 3–4: nDCG@10, MAP, Recall@10. Types 5–6: Recall@10, Recall@20, MAP.

### Type 1 — Identifier (50 queries)

| Metric | BM25 | RRF | RRF − BM25 |
|--------|------|-----|------------|
| `precision@1` (P@1) | 0.0000 | 0.0000 | 0.0000 |
| `mrr` (MRR) | 0.0000 | 0.0018 | 0.0018 |
| `map` (MAP) | 0.0000 | 0.0018 | 0.0018 |

### Type 2 — Name search (50 queries)

| Metric | BM25 | RRF | RRF − BM25 |
|--------|------|-----|------------|
| `precision@1` (P@1) | 1.0000 | 0.9800 | -0.0200 |
| `mrr` (MRR) | 1.0000 | 0.9822 | -0.0178 |
| `map` (MAP) | 1.0000 | 0.9822 | -0.0178 |

### Type 3 — Thematic (14 queries)

| Metric | BM25 | RRF | RRF − BM25 |
|--------|------|-----|------------|
| `ndcg@10` (nDCG@10) | 0.1355 | 0.0394 | -0.0961 |
| `map` (MAP) | 0.0718 | 0.0220 | -0.0498 |
| `recall@10` (Recall@10) | 0.0758 | 0.0574 | -0.0184 |

### Type 4 — Relational (14 queries)

| Metric | BM25 | RRF | RRF − BM25 |
|--------|------|-----|------------|
| `ndcg@10` (nDCG@10) | 0.0550 | 0.0328 | -0.0222 |
| `map` (MAP) | 0.0315 | 0.0218 | -0.0097 |
| `recall@10` (Recall@10) | 0.0568 | 0.0357 | -0.0211 |

### Type 5 — Programme (50 queries)

| Metric | BM25 | RRF | RRF − BM25 |
|--------|------|-----|------------|
| `recall@10` (Recall@10) | 0.0000 | 0.0000 | 0.0000 |
| `recall@20` (Recall@20) | 0.0000 | 0.0000 | 0.0000 |
| `map` (MAP) | 0.0000 | 0.0000 | 0.0000 |

### Type 6 — Semantic (50 queries)

| Metric | BM25 | RRF | RRF − BM25 |
|--------|------|-----|------------|
| `recall@10` (Recall@10) | 0.0016 | 0.0070 | 0.0054 |
| `recall@20` (Recall@20) | 0.0043 | 0.0129 | 0.0086 |
| `map` (MAP) | 0.0058 | 0.0182 | 0.0124 |

---

## 4. Per query type analysis

### Type 1 — Identifier (50 queries)

**Current MAP:** BM25 0.0000, RRF 0.0018, delta 0.0018.

Identifier queries are a deliberate architectural case: exact identifier lookup is best handled by a dedicated identifier index, not by lexical or dense semantic matching. BM25 and dense retrievers are expected to add little here, while fusion only helps if the identifier signal is already present.

### Type 2 — Name search (50 queries)

**Current MAP:** BM25 1.0000, RRF 0.9822, delta -0.0178.

Name-search queries are dominated by lexical evidence and benefit from query normalisation. A common failure mode is partial-name collision, where several entities share one token and fusion amplifies that shared evidence.

### Type 3 — Thematic (14 queries)

**Current MAP:** BM25 0.0718, RRF 0.0220, delta -0.0498.

Thematic queries remain strongly lexical in this setup, so BM25 often stays competitive. Interpretation should be cautious because pooling for these queries was built from lexical systems, which means dense-retriever scores can be lower bounds.

### Type 4 — Relational (14 queries)

**Current MAP:** BM25 0.0315, RRF 0.0218, delta -0.0097.

Relational queries benefit most when embeddings include richer context around names, ownership, and programme terms. This is where dense retrieval is most likely to complement BM25 rather than simply duplicate lexical matches.

### Type 5 — Programme (50 queries)

**Current MAP:** BM25 0.0000, RRF 0.0000, delta 0.0000.

Programme-membership queries usually have strong lexical anchors, so BM25 does a lot of the work. Dense retrieval still helps when the programme relationship is described semantically instead of through a single distinctive keyword.

### Type 6 — Semantic (50 queries)

**Current MAP:** BM25 0.0058, RRF 0.0182, delta 0.0124.

Semantic and cross-lingual queries remain the hardest group. Many of these are not pure semantic-search problems: they bundle structured constraints such as programme, country, and entity type that embeddings alone cannot enforce.

---

## 5. Dense BGE-M3 (by type)

| query_type | n_queries | map | mrr | ndcg@10 | precision@1 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|---|
| 1 | 50 | 0.0018 | 0.0018 |  | 0.0000 |  |  |
| 2 | 50 | 0.8636 | 0.8636 |  | 0.8200 |  |  |
| 3 | 14 | 0.0004 |  | 0.0000 |  | 0.0000 |  |
| 4 | 14 | 0.0089 |  | 0.0176 |  | 0.0179 |  |
| 5 | 50 | 0.0000 |  |  |  | 0.0000 | 0.0000 |
| 6 | 50 | 0.0211 |  |  |  | 0.0096 | 0.0152 |

---

## 6. Next steps

1. Keep reporting overall MAP and Recall@10 as the headline retrieval summary, then use the per-type tables to explain where fusion helps and where it hurts.
2. For type 2, reduce partial-name collisions with stricter lexical constraints such as AND-style matching, phrase matching, or reranking focused on exact full-name agreement.
3. For types 3 and 4, keep the pooling-bias caveat explicit because dense metrics may underestimate true usefulness when judgments come mainly from lexical pools.
4. For type 6, treat metadata-aware retrieval as the main next step: filtering or hybrid retrieval over fields such as programme, schema, and country is more promising than relying on embeddings alone.
5. Continue treating type 7 separately, since answer generation and RAGAS evaluation are not directly comparable to ranked retrieval metrics for types 1–6.

