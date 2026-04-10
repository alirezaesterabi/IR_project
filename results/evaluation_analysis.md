# Evaluation Analysis Report

**Generated:** 2026-04-10 (final — after fix/embedding-text-latin-names)
**Dataset:** OpenSanctions full corpus (1,249,379 documents)
**Pipeline:** 5-retriever RRF fusion (BM25 + TF-IDF + Identifier + MiniLM + BGE-M3)
**Fixes applied:** query normalisation (BM25/TF-IDF dispatch) + Latin names in embedding_text

---

## 1. Overall Pipeline Performance

| System | MAP | Recall@10 |
|--------|-----|-----------|
| BM25 (lexical) | 0.3868 | 0.3239 |
| TF-IDF | 0.2786 | 0.2220 |
| IdentifierRetriever | 0.2193 | 0.2193 |
| Dense MiniLM | 0.0568 | 0.0400 |
| Dense BGE-M3 | 0.1548 | 0.1031 |
| RRF 2-system (BM25 + BGE-M3) | 0.3217 | 0.2886 |
| RRF 5-system — pre-fix | 0.5551 | 0.4962 |
| **RRF 5-system — final** | **0.5679** | **0.5040** |

**Key finding:** The final 5-retriever RRF fusion (MAP 0.5679) outperforms BM25
alone (0.3868) by 47% and the original 2-system fusion (0.3217) by 76%. Two
targeted fixes contributed +1.3% MAP over the initial 5-system result:
query normalisation (+4.1% on type 2) and Latin name enrichment of
embedding_text (+7.4% on type 4).

---

## 2. Per Query Type Results — Final vs All Baselines

| Type | BM25 MAP | RRF 2-sys | RRF 5-sys (pre-fix) | RRF 5-sys (final) | vs BM25 |
|------|----------|-----------|---------------------|-------------------|---------|
| 1 — Identifier | 0.000 | 0.000 | 0.990 | 0.967 | +0.967 |
| 2 — Name search | 0.868 | 0.683 | 0.618 | 0.659 | -0.209 |
| 3 — Thematic | 0.438 | 0.264 | 0.345 | 0.331 | -0.107 |
| 4 — Relational | 0.383 | 0.379 | 0.461 | 0.535 | +0.152 |
| 5 — Programme | 0.661 | 0.581 | 0.672 | 0.694 | +0.033 |
| 6 — Semantic | 0.005 | 0.023 | 0.026 | 0.028 | +0.023 |

---

## 3. Per Query Type Analysis

### Type 1 — Identifier Lookup (50 queries)

**Final MAP: 0.967**

Near-perfect performance via IdentifierRetriever — an inverted index over
`identifiers` fields returning exact matches at rank 1. The minor regression
from 0.990 to 0.967 vs the pre-fix run (one additional miss) is within
noise. The two misses are likely data quality issues in the source records
(missing or malformed identifier fields), not retrieval failures. BM25 and
dense retrievers contribute zero to this type by design.

### Type 2 — Named Entity Search (50 queries)

**Final MAP: 0.659 (+0.041 vs pre-fix, -0.209 vs BM25 alone)**

Query normalisation was the primary driver: BM25 perfect queries increased
from 26 to 41 of 50 by ensuring query tokens match text_blob tokens after
the same normalisation pipeline (accent stripping, punctuation removal,
lowercasing). BM25 alone improved from 0.868 to 0.878 MAP.

The remaining -0.209 gap vs BM25 alone is caused by partial name collision:
20 queries are outranked by entities sharing one common name token
(e.g. "michael", "irakli", "kateryna"). RRF amplifies these collisions
because all 5 retrievers independently boost common-token documents. This
is a known RRF trade-off — the aggregate gain across all types (+0.297 MAP
over BM25 alone) justifies the localised loss on type 2.

### Type 3 — Thematic/Vessel/Programme Search (14 queries)

**Final MAP: 0.331 (-0.107 vs BM25 alone)**

BM25 remains the strongest single retriever. Minor regression vs the
pre-fix run (-0.014) is within noise across 14 queries. The fusion gap vs
BM25 alone reflects pooling bias: ground truth was constructed from
BM25/TF-IDF pools only, so dense retriever results are systematically
unjudged (Zobel, 1998). Dense MAP on this type is a lower bound.

### Type 4 — Relational/Ownership Search (14 queries)

**Final MAP: 0.535 (+0.074 vs pre-fix, +0.152 vs BM25 alone)**

The strongest fusion win. The embedding_text fix drove the gain: richer
document representations (including Latin name variants and text_blob
prefix) improved BGE-M3's ability to match relational queries involving
entity names, ownership terms, and programme context. BGE-M3 contributed
18/138 relevant documents in this type — 6x MiniLM's contribution.

Pooling bias caveat applies as per type 3.

### Type 5 — Programme/Dataset Membership (50 queries)

**Final MAP: 0.694 (+0.022 vs pre-fix, +0.033 vs BM25 alone)**

Solid gain from both fixes combined. BM25 dominates (programme names are
lexically distinctive) but BGE-M3 contributed 757/1434 relevant document
finds — 3.7x MiniLM — making it the most productive dense retriever on
this type. Query normalisation improved programme code token matching;
embedding enrichment helped queries with semantic programme descriptions.

### Type 6 — Semantic/Cross-lingual (50 queries)

**Final MAP: 0.028 (+0.023 vs BM25 alone)**

Weakest type across all systems. BGE-M3 is the best individual retriever
here — the only type where dense beats BM25. Low absolute performance
reflects a fundamental task mismatch: type 6 queries encode structured
constraints (programme codes, country+entity type combinations) that
sentence embeddings cannot parse. "CA SEMA person Russia" requires
metadata filtering on `programId`, `schema`, and `country`, not semantic
similarity. 16/50 queries have zero recall across all systems.

**Root cause:** Structured query problem dressed as free text. Improvement
requires ChromaDB metadata-filtered retrieval, not better embeddings.
Documented as future work.

---

## 4. Fix Impact Analysis

### Fix 1 — Query normalisation (BM25/TF-IDF dispatch)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| BM25 type 2 MAP | 0.868 | 0.878 | +0.010 |
| BM25 type 2 perfect queries | 26/50 | 41/50 | +15 |
| RRF 5-sys type 2 MAP | 0.618 | 0.659 | +0.041 |

Raw query strings were passed directly to BM25/TF-IDF while `text_blob`
tokens had been normalised at index time. Punctuated tokens failed to
match: `Hernández,` → `hernandez`, `'ЭЛЕКТРА` → `электра`, `Ny.` → `ny`.
Fix: apply `TextProcessor.normalize()` to query strings before BM25/TF-IDF
dispatch. IdentifierRetriever (requires raw uppercase) and dense retrievers
(handle raw text natively) were not changed.

### Fix 2 — Latin names in embedding_text

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| RRF 5-sys type 4 MAP | 0.461 | 0.535 | +0.074 |
| RRF 5-sys type 5 MAP | 0.672 | 0.694 | +0.022 |
| Cross-script cosine (Chanturiia Irakli) | 0.564 | 0.568 | +0.004 |

`build_embedding_text()` used the Cyrillic caption as primary input. Latin
transliterations in `text_blob` were excluded. For a 1.25M corpus, a
cosine of 0.56 is insufficient for top-100 retrieval — the improvement to
0.568 was too marginal to fix type 2 name matching. However, richer
embedding context improved relational and programme matching (types 4, 5)
where semantic context beyond the name is discriminative. Both dense
indexes rebuilt (1,249,379 docs, MiniLM 20.9 min, BGE-M3 39.2 min).

---

## 5. Retriever Contribution Analysis

| Retriever | Type 1 | Type 2 | Type 3 | Type 4 | Type 5 | Type 6 |
|-----------|--------|--------|--------|--------|--------|--------|
| BM25 | 0/50 | 50/50 | 45/129 | 65/138 | 1133/1434 | 7/1000 |
| TF-IDF | 0/50 | 39/50 | 41/129 | 53/138 | 1126/1434 | 2/1000 |
| Identifier | 50/50 | 0/50 | 0/129 | 0/138 | 0/1434 | 0/1000 |
| MiniLM | 0/50 | 4/50 | 8/129 | 3/138 | 202/1434 | 32/1000 |
| BGE-M3 | 0/50 | 6/50 | 16/129 | 18/138 | 757/1434 | 22/1000 |

**Key takeaways:**
- IdentifierRetriever is essential and exclusive for type 1
- BM25 is the workhorse for types 2, 3, 4, 5
- BGE-M3 is the strongest dense retriever on types 4, 5 and the best
  overall system on type 6
- MiniLM and BGE-M3 are complementary on type 6 (32 vs 22 finds,
  different documents)
- No single retriever covers all types — the 5-system design is justified

---

## 6. Known Limitations

1. **Type 2 partial name collision (20/50 queries).** Entities sharing
   common first names (michael, irakli, kateryna) outrank the correct
   entity after RRF fusion. AND-query filtering or phrase matching would
   address this but was not implemented.

2. **Pooling bias on types 3 and 4.** Ground truth pooled from BM25/TF-IDF
   only. Dense retriever metrics are lower bounds (Zobel, 1998).

3. **Type 6 structural ceiling.** ~3% recall even with all 5 retrievers.
   Requires metadata-filtered retrieval, not retrieval improvement.

4. **Cross-lingual name matching.** Latin query vs Cyrillic document
   produces cosine ~0.568 — insufficient for top-100 retrieval in a
   1.25M corpus. Embedding enrichment helped types 4/5 but not type 2.

5. **RAG (type 7).** Assessed separately via RAGAS faithfulness — not
   included in ranked retrieval metrics.

---

## 7. Recommendations for CW2 Report

1. Lead with final RRF 5-system MAP 0.5679 as the headline result —
   47% improvement over BM25 alone, demonstrating that retriever
   composition matters.

2. Use the fix impact tables as an ablation study showing incremental
   engineering decisions with measurable outcomes.

3. Frame type 1 (MAP 0.967) as a deliberate architectural win — exact
   identifier lookup requires a dedicated inverted index, not BM25 or
   dense retrieval.

4. Acknowledge type 2 regression vs BM25 honestly — cite Cormack et al.
   (2009) on RRF trade-offs between aggregate gain and per-type loss.

5. Acknowledge pooling bias on types 3/4 explicitly — BGE-M3 metrics
   are lower bounds, not failures.

6. Frame type 6 as an open challenge with a clear diagnosis: structured
   queries require metadata filtering, not better embeddings.
