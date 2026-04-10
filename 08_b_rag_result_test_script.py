"""
08 — RAG Result Test Script

Standalone script converted from notebooks/08_rag_from_pooling_3_0.ipynb.
Reads top-K fused results from 08_rag_top_k_extraction.py, sends each query
to gpt-oss:20b via Ollama, runs structural audit, and exports results.

Usage
-----
    python scripts/08_rag_result_test_script.py

    # Custom RAG input file
    python scripts/08_rag_result_test_script.py --input results/08_(RAG)_top_10_fused_with_full_data_20260410_083228.xlsx

Prerequisites
-------------
    1. Ollama running: ollama serve
    2. Model pulled: ollama pull gpt-oss:20b
    3. RAG input file: python scripts/08_rag_top_k_extraction.py

Outputs (in data/rag_output/)
------------------------
    rag_eval_{timestamp}.csv           — per-query evaluation scores
    rag_notes_{timestamp}.json         — all generated notes + metadata
    notes/{query_id}.txt               — individual triage notes
    {input_stem}_RAG_enriched.xlsx     — enriched Excel (rag_enriched + rag_eval_summary)
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ── Locate project root ────────────────────────────────────────────────────

def find_root() -> Path:
    marker = "data/raw_data"
    for key in ("IR_PROJECT_ROOT", "VSCODE_WORKSPACE_FOLDER"):
        v = os.environ.get(key)
        if v and (Path(v) / marker).exists():
            return Path(v)
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / marker).exists():
            return p
    raise FileNotFoundError("Cannot find project root.")


ROOT = find_root()
sys.path.insert(0, str(ROOT))


# ======================================================================
# CONFIGURATION
# ======================================================================

LLM_MODEL = "gpt-oss:20b"
LLM_TEMPERATURE = 0.0
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 600
MAX_RECORDS_PER_QUERY = 10


# ======================================================================
# LLM Client
# ======================================================================

def llm_generate(system_prompt, user_prompt,
                 temperature=LLM_TEMPERATURE, model=LLM_MODEL):
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "stream": False,
                "options": {"temperature": temperature},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.\n"
            f"  1. Install Ollama: https://ollama.com\n"
            f"  2. Pull model: ollama pull {model}\n"
            f"  3. Make sure Ollama is running"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama error: {e}\n  Run: ollama pull {model}")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama timed out after {OLLAMA_TIMEOUT}s.")


# ======================================================================
# Record formatter
# ======================================================================

def format_pooling_row_for_llm(row, rank):
    lines = []

    source = row.get("meta_datasets", "")
    if pd.isna(source) or not source:
        source = "Unknown Source"
    else:
        source = str(source).split(",")[0].strip()

    score = row.get("rrf_score", "")
    if pd.isna(score):
        score = "N/A"

    lines.append(f"[RECORD {rank} — SOURCE: {source} | RRF Score: {score}]")

    caption = row.get("caption", "N/A")
    if pd.notna(caption):
        lines.append(f"Name: {caption}")

    schema = row.get("schema", "")
    if pd.notna(schema) and schema:
        lines.append(f"Entity Type: {schema}")

    country = row.get("meta_country", "")
    if pd.notna(country) and country:
        lines.append(f"Country: {country}")

    program = row.get("meta_programId", "")
    if pd.notna(program) and program:
        lines.append(f"Sanctions Programs: {str(program)[:300]}")

    datasets = row.get("meta_datasets", "")
    if pd.notna(datasets) and datasets:
        lines.append(f"Listed In: {str(datasets)[:300]}")

    id_fields = {
        "id_callSign": "Call Sign", "id_idNumber": "ID Number",
        "id_imoNumber": "IMO Number", "id_innCode": "INN Code",
        "id_kppCode": "KPP Code", "id_mmsi": "MMSI",
        "id_ogrnCode": "OGRN Code", "id_registrationNumber": "Registration Number",
        "id_uniqueEntityId": "Unique Entity ID", "id_taxNumber": "Tax Number",
        "id_email": "Email", "id_phone": "Phone",
        "id_passportNumber": "Passport Number", "id_leiCode": "LEI Code",
        "id_npiCode": "NPI Code",
    }
    for col, label in id_fields.items():
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            lines.append(f"{label}: {str(val)[:200]}")

    text_preview = row.get("embedding_text", row.get("text_blob", ""))
    if pd.notna(text_preview) and str(text_preview).strip():
        lines.append(f"Text Details: {str(text_preview)[:800]}")

    for date_col, label in [("first_seen", "First Seen"), ("last_seen", "Last Seen")]:
        val = row.get(date_col, "")
        if pd.notna(val) and str(val).strip():
            lines.append(f"{label}: {str(val)[:30]}")

    return "\n".join(lines)


# ======================================================================
# Prompts
# ======================================================================

SYSTEM_PROMPT = """You are a sanctions intelligence analyst. You are reviewing search results
retrieved for an analytical query. Your task is to assess each result
against the query criteria and produce a structured intelligence brief.

RULES:

1. Only use information present in the search results below.
   Do not add facts from your own knowledge, even if you are certain
   they are true.

2. For each record, explicitly state which query criteria it satisfies
   and which it does not, citing the specific fields from the record.

3. When a record lacks information needed to assess a criterion,
   state "NOT DETERMINABLE FROM RECORD" — do not assume.

4. Use [Record N] tags to attribute every factual claim.
"""

USER_PROMPT_TEMPLATE = """Analyze the following search results against the query criteria.

QUERY:
Search terms: {query_name}
Query type: {query_type}
Analyst notes: {query_notes}

SEARCH RESULTS:
{formatted_records}

---

REQUIRED OUTPUT:

### CRITERIA EXTRACTION

First, break down the query into its individual criteria:
- Criterion 1: [extracted from query]
- Criterion 2: [extracted from query]
- Criterion 3: [extracted from query]

### PER-RECORD ASSESSMENT

For EACH record:

**Record [N]: [Entity name] | [Entity type] | [Country]**

Criteria met:
- Criterion 1: [MET / NOT MET / NOT DETERMINABLE] — [evidence from record]
- Criterion 2: [MET / NOT MET / NOT DETERMINABLE] — [evidence from record]
- Criterion 3: [MET / NOT MET / NOT DETERMINABLE] — [evidence from record]

Key facts from record:
- [fact 1] [Record N]
- [fact 2] [Record N]

Relevance: [RELEVANT / PARTIALLY RELEVANT / NOT RELEVANT]

### CROSS-RECORD PATTERNS

Identify any connections, patterns, or groupings across the relevant records:
- Common sanctions programs
- Shared jurisdictions or countries
- Related entities mentioned across multiple records
- Common designation reasons

If no patterns are identifiable from the records, state:
"No cross-record patterns identifiable from the provided data."

### SUMMARY

Total records assessed: [N]
Relevant: [N]
Partially relevant: [N]
Not relevant: [N]

Brief (2-3 sentences): Summarize what the relevant records collectively
reveal about the query topic, using only facts from the records.

### GAPS AND LIMITATIONS

What information is missing that would improve this analysis?
What aspects of the query could not be fully addressed by these records?
"""


# ======================================================================
# RAG pipeline
# ======================================================================

def run_rag_for_query(query_group_df):
    first_row = query_group_df.iloc[0]
    query_id = str(first_row.get("query_id", ""))
    query_text = str(first_row.get("query_text", ""))
    query_type = str(first_row.get("query_type", ""))
    query_notes = str(first_row.get("query_notes", ""))

    records_df = query_group_df.head(MAX_RECORDS_PER_QUERY)

    formatted_blocks = []
    for rank, (_, row) in enumerate(records_df.iterrows(), 1):
        formatted_blocks.append(format_pooling_row_for_llm(row, rank))

    formatted_records = "\n\n".join(formatted_blocks)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        query_name=query_text,
        query_type=query_type,
        query_notes=query_notes if query_notes != "nan" else "None",
        formatted_records=formatted_records,
    )

    t0 = time.time()
    generated_note = llm_generate(SYSTEM_PROMPT, user_prompt)
    latency = time.time() - t0

    return {
        "query_id": query_id,
        "query_text": query_text,
        "query_type": query_type,
        "query_notes": query_notes,
        "n_records": len(records_df),
        "formatted_records": formatted_records,
        "user_prompt": user_prompt,
        "generated_note": generated_note,
        "latency_seconds": latency,
    }


# ======================================================================
# Structural audit
# ======================================================================

def structural_audit(generated_output, formatted_records):
    issues = []

    banned = [
        "is known to", "is believed to", "reportedly",
        "it is well known", "based on available information",
    ]
    found_banned = [p for p in banned if p in generated_output.lower()]
    if found_banned:
        issues.append({"type": "BANNED_PHRASE", "severity": "CRITICAL",
                       "phrases": found_banned})

    cited = set(re.findall(r'\[Record (\d+)\]', generated_output, re.IGNORECASE))
    n_input = len(re.findall(r'\[RECORD \d+', formatted_records))
    phantom = [r for r in cited if int(r) > n_input]
    if phantom:
        issues.append({"type": "PHANTOM_SOURCE", "severity": "CRITICAL",
                       "phantom_records": phantom, "max_valid": n_input})

    tp_sections = re.findall(
        r'CLASSIFICATION:\s*TRUE POSITIVE(.*?)(?=CLASSIFICATION:|###|$)',
        generated_output, re.DOTALL | re.IGNORECASE
    )
    weak_tp = False
    for section in tp_sections:
        matches = re.findall(r'—\s*match', section.lower())
        if len(matches) < 2:
            weak_tp = True
            issues.append({"type": "WEAK_TRUE_POSITIVE", "severity": "CRITICAL",
                           "detail": "TRUE POSITIVE with < 2 field matches"})

    assoc = re.findall(r'[^.]*is associated with[^.]*\.', generated_output, re.IGNORECASE)
    for match in assoc:
        if not re.search(r'\[Record \d+\]', match, re.IGNORECASE):
            issues.append({"type": "UNGROUNDED_ASSOCIATION", "severity": "WARNING",
                           "sentence": match.strip()[:150]})

    scores = {
        "no_banned_phrases": 1 if not found_banned else 0,
        "no_phantom_sources": 1 if not phantom else 0,
        "no_weak_true_positive": 0 if weak_tp else 1,
    }
    scores["structural_score"] = round(sum(scores.values()) / 3, 4)

    return {
        "pass": len([i for i in issues if i.get("severity") == "CRITICAL"]) == 0,
        "issues": issues,
        "issue_count": len(issues),
        "scores": scores,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="08 — RAG Result Test Script: run RAG triage on fused results via Ollama."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to 08_(RAG)_*.xlsx. Auto-detects most recent if not provided."
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Query suffix to select, e.g. '003' for Q1_003, Q2_003, etc. "
             "If not provided, runs ALL queries in the input file."
    )
    args = parser.parse_args()

    results_dir = ROOT / "results"
    output_dir = ROOT / "data/rag_output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Resolve input file ────────────────────────────────────────────
    if args.input:
        rag_input_path = Path(args.input)
    else:
        candidates = sorted(results_dir.glob("08_(RAG)_*.xlsx"))
        if not candidates:
            print("ERROR: No 08_(RAG)_*.xlsx found in results/.")
            print("  Run: python scripts/08_rag_top_k_extraction.py")
            return 1
        rag_input_path = candidates[-1]

    if not rag_input_path.exists():
        print(f"ERROR: File not found: {rag_input_path}")
        return 1

    print("=" * 60)
    print("  08 — RAG Result Test Script")
    print("=" * 60)
    query_suffix = args.query
    print(f"  Input       : {rag_input_path.name}")
    print(f"  LLM         : {LLM_MODEL}")
    print(f"  Ollama      : {OLLAMA_BASE_URL}")
    print(f"  Max records : {MAX_RECORDS_PER_QUERY}")
    print(f"  Query       : {f'_{query_suffix} only' if query_suffix else 'ALL queries'}")
    print(f"  Timestamp   : {timestamp}")
    print()

    # Query tag for output filenames
    qtag = f"_Q{query_suffix}" if query_suffix else "_ALL"

    # ── Connectivity test ─────────────────────────────────────────────
    print("Testing Ollama connection...")
    _test = llm_generate("You are a test.", "Say OK and nothing else.")
    print(f"  Response: '{_test.strip()[:50]}'")
    print(f"  {LLM_MODEL} connected and ready.\n")

    # ── Load data ─────────────────────────────────────────────────────
    df = pd.read_excel(rag_input_path, sheet_name="All_Combined")
    df["query_type"] = df["query_id"].str.extract(r"Q(\d+)_")[0].apply(lambda x: f"Type {x}")
    df["query_notes"] = ""

    # Filter to specific query suffix if provided
    if query_suffix:
        target_qids = [f"Q{t}_{query_suffix}" for t in range(1, 7)]
        df = df[df["query_id"].isin(target_qids)].copy()
        if df.empty:
            print(f"ERROR: No queries matching _{query_suffix} found in input file.")
            return 1

    print(f"Loaded {len(df):,} rows, {df['query_id'].nunique()} queries\n")

    # ── Process all queries ───────────────────────────────────────────
    query_ids = df["query_id"].unique()
    print(f"Processing {len(query_ids)} queries...")
    print("=" * 80)

    all_results = []
    all_evals = []

    for i, qid in enumerate(query_ids, 1):
        query_df = df[df["query_id"] == qid]
        query_text = str(query_df.iloc[0].get("query_text", qid))
        print(f"\n[{i}/{len(query_ids)}] {qid}: {query_text[:60]}...")

        result = run_rag_for_query(query_df)
        all_results.append(result)

        audit = structural_audit(result["generated_note"], result["formatted_records"])

        eval_row = {
            "query_id": result["query_id"],
            "query_text": result["query_text"][:60],
            "n_records": result["n_records"],
            "latency_seconds": round(result["latency_seconds"], 1),
            "note_length": len(result["generated_note"]),
            "structural_pass": audit["pass"],
            "structural_issues": audit["issue_count"],
            **audit["scores"],
        }
        all_evals.append(eval_row)

        status = "PASS" if audit["pass"] else "FAIL"
        print(f"  Records: {result['n_records']} | "
              f"Time: {result['latency_seconds']:.1f}s | "
              f"Structural: {status} | "
              f"Note: {len(result['generated_note']):,} chars")

    print(f"\n{'=' * 80}")
    print(f"Complete: {len(all_results)} queries processed.\n")

    # ── Summary ───────────────────────────────────────────────────────
    df_eval = pd.DataFrame(all_evals)
    print("AGGREGATE METRICS:")
    print(f"  Queries processed      : {len(df_eval)}")
    print(f"  Structural pass rate   : {df_eval['structural_pass'].mean():.1%}")
    print(f"  Mean structural score  : {df_eval['structural_score'].mean():.4f}")
    print(f"  Mean latency           : {df_eval['latency_seconds'].mean():.1f}s")
    print(f"  Mean note length       : {df_eval['note_length'].mean():,.0f} chars\n")

    # ── Export ────────────────────────────────────────────────────────
    print("Exporting results...")

    # 1. Evaluation CSV
    eval_path = output_dir / f"rag_eval{qtag}_{timestamp}.csv"
    df_eval.to_csv(eval_path, index=False)
    print(f"  Evaluation scores : {eval_path}")

    # 2. All notes JSON
    notes_path = output_dir / f"rag_notes{qtag}_{timestamp}.json"
    notes_export = []
    for result, ev in zip(all_results, all_evals):
        notes_export.append({
            "query_id": result["query_id"],
            "query_text": result["query_text"],
            "query_type": result["query_type"],
            "n_records": result["n_records"],
            "generated_note": result["generated_note"],
            "latency_seconds": result["latency_seconds"],
            "evaluation": ev,
        })
    with open(notes_path, "w", encoding="utf-8") as f:
        json.dump(notes_export, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Triage notes      : {notes_path}")

    # 3. Individual note text files
    notes_dir = output_dir / "notes"
    notes_dir.mkdir(exist_ok=True)
    for result in all_results:
        note_file = notes_dir / f"{result['query_id']}.txt"
        with open(note_file, "w", encoding="utf-8") as f:
            f.write(f"QUERY: {result['query_text']}\n")
            f.write(f"TYPE: {result['query_type']}\n")
            f.write(f"RECORDS: {result['n_records']}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(result["generated_note"])
    print(f"  Individual notes  : {notes_dir}/ ({len(all_results)} files)")

    # ── Enriched Excel ────────────────────────────────────────────────
    print("\nBuilding enriched Excel...")

    record_notes = {}
    record_class = {}

    for result in all_results:
        qid = result["query_id"]
        query_df = df[df["query_id"] == qid].head(MAX_RECORDS_PER_QUERY)

        sections = re.split(r'### Result \d+:', result["generated_note"])
        sections = [s.strip() for s in sections[1:] if s.strip()]

        if not sections:
            sections = re.split(r'\*\*Result \d+', result["generated_note"])
            sections = [s.strip() for s in sections[1:] if s.strip()]

        if not sections:
            sections = re.split(r'Result \d+:', result["generated_note"])
            sections = [s.strip() for s in sections[1:] if s.strip()]

        for idx, (_, row) in enumerate(query_df.iterrows()):
            did = str(row.get("doc_id", ""))
            key = (qid, did)

            if idx < len(sections):
                note = sections[idx]
                note_lines = note.split("\n", 1)
                record_notes[key] = note_lines[1].strip() if len(note_lines) > 1 else note.strip()
                cls_match = re.search(
                    r"(TRUE POSITIVE|AMBIGUOUS|LIKELY FALSE POSITIVE)",
                    note, re.IGNORECASE
                )
                record_class[key] = cls_match.group(0).upper() if cls_match else ""
            elif idx == 0 and not sections:
                record_notes[key] = result["generated_note"]
                cls_match = re.search(
                    r"(TRUE POSITIVE|AMBIGUOUS|LIKELY FALSE POSITIVE)",
                    result["generated_note"], re.IGNORECASE
                )
                record_class[key] = cls_match.group(0).upper() if cls_match else ""
            else:
                record_notes[key] = ""
                record_class[key] = ""

    df_enriched = df.copy()
    df_enriched["rag_note"] = df_enriched.apply(
        lambda r: record_notes.get((str(r["query_id"]), str(r.get("doc_id", ""))), ""), axis=1
    )
    df_enriched["rag_classification"] = df_enriched.apply(
        lambda r: record_class.get((str(r["query_id"]), str(r.get("doc_id", ""))), ""), axis=1
    )
    df_enriched["rag_model"] = LLM_MODEL
    df_enriched["rag_latency_sec"] = df_enriched["query_id"].map(
        {r["query_id"]: round(r["latency_seconds"], 2) for r in all_results}
    )
    df_enriched["rag_structural_pass"] = df_enriched["query_id"].map(
        {ev["query_id"]: ev.get("structural_pass", "") for ev in all_evals}
    )
    df_enriched["rag_structural_score"] = df_enriched["query_id"].map(
        {ev["query_id"]: ev.get("structural_score", "") for ev in all_evals}
    )
    df_enriched["rag_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    input_stem = rag_input_path.stem
    out_path = output_dir / f"{input_stem}{qtag}_RAG_enriched.xlsx"

    original_cols = [c for c in df.columns if c in df_enriched.columns]
    rag_cols = ["rag_classification", "rag_note", "rag_model",
                "rag_latency_sec", "rag_structural_pass", "rag_structural_score",
                "rag_timestamp"]
    all_cols = original_cols + [c for c in rag_cols if c in df_enriched.columns]
    df_export = df_enriched[all_cols]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name="rag_enriched")
        ws = writer.sheets["rag_enriched"]
        col_widths = {
            "query_id": 12, "query_text": 50, "doc_id": 28, "caption": 30,
            "rag_classification": 22, "rag_note": 80, "rag_model": 22,
            "rag_latency_sec": 14, "rag_structural_pass": 18,
            "rag_structural_score": 18, "rag_timestamp": 20,
            "embedding_text": 60, "text_blob": 60,
            "meta_country": 12, "schema": 12, "meta_programId": 30,
        }
        for col_idx, col_name in enumerate(all_cols, 1):
            width = col_widths.get(col_name, 15)
            letter = ws.cell(row=1, column=col_idx).column_letter
            ws.column_dimensions[letter].width = width
        ws.freeze_panes = "A2"

        df_summary = pd.DataFrame([{
            "query_id": ev["query_id"],
            "query_text": ev.get("query_text", "")[:60],
            "n_records": ev.get("n_records", 0),
            "structural_pass": ev.get("structural_pass", ""),
            "structural_score": ev.get("structural_score", ""),
            "latency_seconds": ev.get("latency_seconds", ""),
            "note_length": ev.get("note_length", ""),
        } for ev in all_evals])
        df_summary.to_excel(writer, index=False, sheet_name="rag_eval_summary")
        ws2 = writer.sheets["rag_eval_summary"]
        ws2.freeze_panes = "A2"
        for col_idx in range(1, len(df_summary.columns) + 1):
            ws2.column_dimensions[ws2.cell(row=1, column=col_idx).column_letter].width = 18

    print(f"\n{'=' * 60}")
    print(f"  ENRICHED EXCEL EXPORTED")
    print(f"{'=' * 60}")
    print(f"  File    : {out_path}")
    print(f"  Rows    : {len(df_export):,}")
    print(f"  Columns : {len(all_cols)}")
    print(f"  Sheets  : 'rag_enriched' + 'rag_eval_summary'")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
