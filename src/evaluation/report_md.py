"""
Markdown report for Types 1–6 evaluation (BM25 vs RRF BGE-M3, optional dense).

Structured like `results/evaluation_analysis.md`: overall table, per-type MAP
comparison, then full TYPE_METRICS per type (e.g. precision@1 for types 1–2).
No per-query rows.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .utils import (
    ALL_TABLE_METRICS,
    OVERALL_METRICS,
    TYPE_METRICS,
    build_bm25_rrf_comparison_table,
)

# Short labels (aligned with coursework narrative / evaluation_analysis.md)
TYPE_LABELS: dict[int, str] = {
    1: "Identifier",
    2: "Name search",
    3: "Thematic",
    4: "Relational",
    5: "Programme",
    6: "Semantic",
}

_METRIC_DISPLAY: dict[str, str] = {
    "map": "MAP",
    "mrr": "MRR",
    "precision@1": "P@1",
    "ndcg@10": "nDCG@10",
    "recall@10": "Recall@10",
    "recall@20": "Recall@20",
}

TYPE_EXPLANATIONS: dict[int, str] = {
    1: (
        "Identifier queries are a deliberate architectural case: exact identifier "
        "lookup is best handled by a dedicated identifier index, not by lexical or "
        "dense semantic matching. BM25 and dense retrievers are expected to add "
        "little here, while fusion only helps if the identifier signal is already present."
    ),
    2: (
        "Name-search queries are dominated by lexical evidence and benefit from query "
        "normalisation. A common failure mode is partial-name collision, where several "
        "entities share one token and fusion amplifies that shared evidence."
    ),
    3: (
        "Thematic queries remain strongly lexical in this setup, so BM25 often stays "
        "competitive. Interpretation should be cautious because pooling for these queries "
        "was built from lexical systems, which means dense-retriever scores can be lower bounds."
    ),
    4: (
        "Relational queries benefit most when embeddings include richer context around "
        "names, ownership, and programme terms. This is where dense retrieval is most likely "
        "to complement BM25 rather than simply duplicate lexical matches."
    ),
    5: (
        "Programme-membership queries usually have strong lexical anchors, so BM25 does a "
        "lot of the work. Dense retrieval still helps when the programme relationship is "
        "described semantically instead of through a single distinctive keyword."
    ),
    6: (
        "Semantic and cross-lingual queries remain the hardest group. Many of these are "
        "not pure semantic-search problems: they bundle structured constraints such as "
        "programme, country, and entity type that embeddings alone cannot enforce."
    ),
}

NEXT_STEPS: list[str] = [
    "Keep reporting overall MAP and Recall@10 as the headline retrieval summary, then use the per-type tables to explain where fusion helps and where it hurts.",
    "For type 2, reduce partial-name collisions with stricter lexical constraints such as AND-style matching, phrase matching, or reranking focused on exact full-name agreement.",
    "For types 3 and 4, keep the pooling-bias caveat explicit because dense metrics may underestimate true usefulness when judgments come mainly from lexical pools.",
    "For type 6, treat metadata-aware retrieval as the main next step: filtering or hybrid retrieval over fields such as programme, schema, and country is more promising than relying on embeddings alone.",
    "Continue treating type 7 separately, since answer generation and RAGAS evaluation are not directly comparable to ranked retrieval metrics for types 1–6.",
]


def _fmt_metric(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        f = float(v)
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return str(v)


def _metric_title(m: str) -> str:
    return _METRIC_DISPLAY.get(m, m)


def _safe_delta(a: Any, b: Any) -> float:
    if pd.isna(a) or pd.isna(b):
        return float("nan")
    return float(a) - float(b)


def write_types_1_6_evaluation_report(
    *,
    path: str | Path,
    sample_size: str,
    bm25_by_type: pd.DataFrame,
    rrf_by_type: pd.DataFrame,
    bm25_overall: Mapping[str, float],
    rrf_overall: Mapping[str, float],
    dense_by_type: pd.DataFrame | None = None,
    run_date: str | None = None,
    systems_evaluated: list[str] | None = None,
) -> None:
    """
    Write a stable Markdown report: DATE, SAMPLE, overall MAP / Recall@10,
    per-type MAP comparison, then full per-type metrics (TYPE_METRICS).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = run_date or date.today().isoformat()

    compare = build_bm25_rrf_comparison_table(bm25_by_type, rrf_by_type)

    lines: list[str] = [
        "# Types 1–6 Evaluation Report",
        "",
        f"**DATE:** {generated}",
        f"**SAMPLE:** {sample_size}",
    ]
    if systems_evaluated:
        lines.append(f"**Systems evaluated:** {', '.join(systems_evaluated)}")
    lines.extend(
        [
            "",
            "Metric policy: `TYPE_METRICS` / `OVERALL_METRICS` in `src/evaluation/utils.py`.",
            "",
            "---",
            "",
            "## 1. Overall pipeline performance",
            "",
            "| System | MAP | Recall@10 |",
            "|--------|-----|-----------|",
        ]
    )
    b_map = bm25_overall.get("map", float("nan"))
    b_r10 = bm25_overall.get("recall@10", float("nan"))
    r_map = rrf_overall.get("map", float("nan"))
    r_r10 = rrf_overall.get("recall@10", float("nan"))
    lines.append(f"| BM25 | {_fmt_metric(b_map)} | {_fmt_metric(b_r10)} |")
    lines.append(f"| RRF (BGE-M3) | {_fmt_metric(r_map)} | {_fmt_metric(r_r10)} |")
    lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 2. Per query type — MAP (BM25 vs RRF)",
            "",
            "| Type | Description | n | BM25 MAP | RRF MAP | RRF − BM25 |",
            "|------|-------------|---|----------|---------|------------|",
        ]
    )
    for qtype in sorted(TYPE_METRICS.keys()):
        row = compare.loc[compare["query_type"] == qtype]
        label = TYPE_LABELS.get(qtype, "")
        nq = ""
        bm_map = rrf_map = d_map = float("nan")
        if not row.empty:
            r0 = row.iloc[0]
            if "n_queries" in r0.index and pd.notna(r0["n_queries"]):
                nq = str(int(r0["n_queries"]))
            bm_map = r0.get("map_bm25", float("nan"))
            rrf_map = r0.get("map_rrf", float("nan"))
            d_map = r0.get("map_RRF_minus_BM25", float("nan"))
        lines.append(
            f"| {qtype} | {label} | {nq} | {_fmt_metric(bm_map)} | {_fmt_metric(rrf_map)} | {_fmt_metric(d_map)} |"
        )
    lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 3. Per query type — full metrics (relevant per type)",
            "",
            "Types 1–2: P@1, MRR, MAP. Types 3–4: nDCG@10, MAP, Recall@10. Types 5–6: Recall@10, Recall@20, MAP.",
            "",
        ]
    )

    for qtype in sorted(TYPE_METRICS.keys()):
        metrics = TYPE_METRICS[qtype]
        row = compare.loc[compare["query_type"] == qtype]
        nq = ""
        if not row.empty and "n_queries" in row.columns:
            nq = row["n_queries"].iloc[0]
            if pd.notna(nq):
                nq = f" ({int(nq)} queries)"
        title = TYPE_LABELS.get(qtype, f"type {qtype}")
        lines.append(f"### Type {qtype} — {title}{nq}")
        lines.append("")
        sub = "| Metric | BM25 | RRF | RRF − BM25 |"
        sep = "|--------|------|-----|------------|"
        lines.extend([sub, sep])
        if row.empty:
            lines.append("| — | — | — | — |")
        else:
            r0 = row.iloc[0]
            for m in metrics:
                cb, cr, cd = f"{m}_bm25", f"{m}_rrf", f"{m}_RRF_minus_BM25"
                b = r0.get(cb, float("nan"))
                rr = r0.get(cr, float("nan"))
                d = r0.get(cd, float("nan"))
                if not (cb in r0.index or cr in r0.index):
                    continue
                disp = _metric_title(m)
                lines.append(
                    f"| `{m}` ({disp}) | {_fmt_metric(b)} | {_fmt_metric(rr)} | {_fmt_metric(d)} |"
                )
        lines.append("")

    lines.extend(["---", "", "## 4. Per query type analysis", ""])

    for qtype in sorted(TYPE_METRICS.keys()):
        title = TYPE_LABELS.get(qtype, f"type {qtype}")
        row = compare.loc[compare["query_type"] == qtype]
        nq = ""
        map_line = ""
        if not row.empty:
            r0 = row.iloc[0]
            if "n_queries" in r0.index and pd.notna(r0["n_queries"]):
                nq = f" ({int(r0['n_queries'])} queries)"
            bm_map = r0.get("map_bm25", float("nan"))
            rrf_map = r0.get("map_rrf", float("nan"))
            map_delta = _safe_delta(rrf_map, bm_map)
            map_line = (
                f"**Current MAP:** BM25 {_fmt_metric(bm_map)}, "
                f"RRF {_fmt_metric(rrf_map)}, "
                f"delta {_fmt_metric(map_delta)}."
            )
        lines.append(f"### Type {qtype} — {title}{nq}")
        lines.append("")
        if map_line:
            lines.append(map_line)
            lines.append("")
        lines.append(TYPE_EXPLANATIONS[qtype])
        lines.append("")

    if dense_by_type is not None and not dense_by_type.empty:
        lines.extend(["---", "", "## 5. Dense BGE-M3 (by type)", ""])
        dcols = ["query_type", "n_queries"] + [
            c for c in ALL_TABLE_METRICS if c in dense_by_type.columns
        ]
        dsub = dense_by_type[[c for c in dcols if c in dense_by_type.columns]]
        dhdr = "| " + " | ".join(str(c) for c in dsub.columns) + " |"
        dsep = "|" + "|".join(["---"] * len(dsub.columns)) + "|"
        lines.extend([dhdr, dsep])
        for _, dr in dsub.iterrows():
            cells: list[str] = []
            for c in dsub.columns:
                v = dr[c]
                if c in ("query_type", "n_queries") and pd.notna(v):
                    try:
                        cells.append(str(int(float(v))))
                    except (TypeError, ValueError):
                        cells.append(_fmt_metric(v))
                else:
                    cells.append(_fmt_metric(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    next_steps_heading = "## 6. Next steps" if dense_by_type is not None and not dense_by_type.empty else "## 5. Next steps"
    lines.extend(["---", "", next_steps_heading, ""])
    for idx, item in enumerate(NEXT_STEPS, start=1):
        lines.append(f"{idx}. {item}")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
