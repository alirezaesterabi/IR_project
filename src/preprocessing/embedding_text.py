"""
Natural-language text for dense retrieval (sentence-transformers).

Built from the processed document dict emitted by build_document(): caption,
schema, metadata (country → full name, program/dataset labels), identifiers,
and optional text_blob tail when the structured parts are sparse.

Lexical indexes (BM25 / TF-IDF) continue to use text_blob and tokens; this
string is stored separately as embedding_text on each JSONL line.
"""

from __future__ import annotations

from typing import Any

import pycountry

_COUNTRY_CACHE: dict[str, str] = {}

_PROGRAM_LABELS = {
    "US-OFAC": "U.S. OFAC sanctions",
    "US-NARCO": "U.S. Foreign Narcotics Kingpin Designation Act",
    "US-GLOMAG": "U.S. Global Magnitsky sanctions",
    "US-BIS": "U.S. Bureau of Industry and Security",
    "US-BIS-DPL": "U.S. BIS Denied Persons List",
    "US-BIS-EL": "U.S. BIS Entity List",
    "US-SDGT": "U.S. Specially Designated Global Terrorist",
    "US-SDN": "U.S. Specially Designated Nationals",
    "US-SAM": "U.S. SAM Exclusions",
    "US-HHS-OIG": "U.S. HHS Office of Inspector General exclusions",
    "EU-UKR": "EU Ukraine-related sanctions",
    "EU-FSF": "EU Financial Sanctions",
    "UA-SA1644": "Ukraine NSDC sanctions",
    "CA-SEMA": "Canada Special Economic Measures Act",
    "INTERPOL-RN": "Interpol Red Notice",
    "SECO": "Swiss SECO sanctions",
    "AU-SANCTIONS": "Australian sanctions",
    "GB-HMT": "UK HM Treasury sanctions",
}

_DATASET_LABELS = {
    "us_ofac_sdn": "U.S. OFAC SDN List",
    "us_ofac_cons": "U.S. OFAC Consolidated List",
    "us_sam_exclusions": "U.S. SAM Exclusions",
    "us_trade_csl": "U.S. Consolidated Screening List",
    "us_bis_denied": "U.S. BIS Denied Persons",
    "us_bis_entity": "U.S. BIS Entity List",
    "eu_journal_sanctions": "EU Official Journal sanctions",
    "eu_fsf": "EU Financial Sanctions",
    "gb_hmt_sanctions": "UK HM Treasury sanctions",
    "un_sc_sanctions": "UN Security Council sanctions",
    "interpol_red_notices": "Interpol Red Notices",
    "ch_seco_sanctions": "Swiss SECO sanctions",
    "opencorporates": "OpenCorporates registry",
    "ext_us_ofac_press_releases": "OFAC press releases",
    "ua_nsdc_sanctions": "Ukraine NSDC sanctions",
    "ca_sema_sanctions": "Canada SEMA sanctions",
    "au_dfat_sanctions": "Australian DFAT sanctions",
}


def _country_name(code: str) -> str:
    """Convert ISO-3166 alpha-2 code to full country name, with cache."""
    code = code.strip().upper()
    if code in _COUNTRY_CACHE:
        return _COUNTRY_CACHE[code]
    try:
        name = pycountry.countries.get(alpha_2=code).name
    except (AttributeError, LookupError):
        _extra = {
            "XK": "Kosovo",
            "TW": "Taiwan",
            "PS": "Palestine",
            "AN": "Netherlands Antilles",
            "CS": "Serbia and Montenegro",
            "SU": "Soviet Union",
            "YU": "Yugoslavia",
            "XX": "Unknown",
        }
        name = _extra.get(code, code)
    _COUNTRY_CACHE[code] = name
    return name


def _program_label(prog_id: str) -> str:
    """Convert a programId like 'US-NARCO' to a readable label."""
    pid = prog_id.strip().upper()
    if pid in _PROGRAM_LABELS:
        return _PROGRAM_LABELS[pid]
    for prefix in sorted(_PROGRAM_LABELS, key=len, reverse=True):
        if pid.startswith(prefix):
            return _PROGRAM_LABELS[prefix]
    return pid.replace("-", " ").replace("_", " ")


def _dataset_label(ds: str) -> str:
    """Convert dataset slug to readable label."""
    return _DATASET_LABELS.get(ds, ds.replace("_", " ").title())


def build_embedding_text(doc: dict[str, Any]) -> str:
    """
    Build a natural-language string for dense embedding from a processed document.

    Parameters
    ----------
    doc : dict
        Output shape of build_document(): caption, schema, metadata,
        identifiers, text_blob.

    Returns
    -------
    str
        Single string for SentenceTransformer-style encoding.
    """
    parts: list[str] = []

    caption = doc.get("caption") or ""
    schema = doc.get("schema") or ""
    if caption:
        parts.append(f"{caption}.")
    if schema:
        parts.append(f"Type: {schema}.")

    meta = doc.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}

    countries_raw = meta.get("country", [])
    if countries_raw and isinstance(countries_raw, list):
        country_names = [_country_name(c) for c in countries_raw[:5] if isinstance(c, str)]
        if country_names:
            parts.append(f"Country: {', '.join(country_names)}.")

    programs_raw = meta.get("programId", [])
    if programs_raw and isinstance(programs_raw, list):
        seen: set[str] = set()
        prog_labels: list[str] = []
        for p in programs_raw[:4]:
            if not isinstance(p, str):
                continue
            lbl = _program_label(p)
            if lbl not in seen:
                prog_labels.append(lbl)
                seen.add(lbl)
        if prog_labels:
            parts.append(f"Sanctioned under: {'; '.join(prog_labels)}.")

    datasets_raw = meta.get("datasets", [])
    if datasets_raw and isinstance(datasets_raw, list):
        ds_labels: list[str] = []
        seen_ds: set[str] = set()
        for ds in datasets_raw[:6]:
            if not isinstance(ds, str):
                continue
            lbl = _dataset_label(ds)
            if lbl not in seen_ds:
                ds_labels.append(lbl)
                seen_ds.add(lbl)
        if ds_labels:
            parts.append(f"Listed in: {'; '.join(ds_labels[:3])}.")

    idents = doc.get("identifiers", {})
    if isinstance(idents, dict):
        id_parts: list[str] = []
        if idents.get("imoNumber"):
            imo_vals = idents["imoNumber"]
            if isinstance(imo_vals, list):
                id_parts.append(
                    f"IMO number: {', '.join(str(v) for v in imo_vals[:2])}"
                )
        if idents.get("callSign"):
            cs_vals = idents["callSign"]
            if isinstance(cs_vals, list):
                id_parts.append(
                    f"Call sign: {', '.join(str(v) for v in cs_vals[:2])}"
                )
        if idents.get("registrationNumber"):
            reg_vals = idents["registrationNumber"]
            if isinstance(reg_vals, list):
                id_parts.append(
                    f"Registration: {', '.join(str(v) for v in reg_vals[:2])}"
                )
        if idents.get("innCode"):
            inn_vals = idents["innCode"]
            if isinstance(inn_vals, list):
                id_parts.append(
                    f"INN code: {', '.join(str(v) for v in inn_vals[:2])}"
                )
        if id_parts:
            parts.append(" ".join(id_parts) + ".")

    # Sparse = only caption + schema lines (<=2 parts), no extra enrichment
    if len(parts) <= 2:
        blob = doc.get("text_blob") or ""
        if isinstance(blob, str) and blob:
            parts.append(blob[:500])

    return " ".join(parts)
