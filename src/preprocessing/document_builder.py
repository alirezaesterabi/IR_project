"""
Flatten raw OpenSanctions entity records into processed document dicts.

Each output document has:
  doc_id      — original entity ID (NK-...)
  caption     — human-readable name (from record["caption"])
  schema      — entity type (Person, Company, Vessel, ...)
  text_blob   — single searchable string; this is what BM25/TF-IDF indexes
  tokens      — list of tokens derived from text_blob
  identifiers — raw identifier values for exact-match queries (Type 1)
  metadata    — country, programId, datasets, timestamps for filtering

run_pipeline() adds embedding_text — natural-language string for dense retrieval
(build_embedding_text), after build_document().

text_blob assembly order (importance-ranked, mirrors plan):
  1. name values            — normalized, not lemmatized
  2. alias, previousName    — normalized, not lemmatized
  3. position, sector, legalForm, keywords, birthPlace — keyword normalized
  4. notes, description     — lemmatized free text
  5. topics                 — keyword normalized
  6. sanctions[*].authority + reason — lemmatized
  7. addressEntity[*].full  — normalized
"""

from typing import Optional
from .text_processing import TextProcessor


# Fields whose values are lists of strings (flat, no nesting)
_NAME_FIELDS = ("name", "alias", "previousName", "weakAlias", "middleName",
                 "fatherName", "motherName")
_KEYWORD_FIELDS = ("position", "sector", "legalForm", "topics", "keywords",
                    "birthPlace")
_DESC_FIELDS = ("notes", "description", "summary")

# Identifiers for exact-match retrieval — stored raw, never tokenized
_IDENTIFIER_FIELDS = (
    "imoNumber",
    "mmsi",
    "callSign",
    "registrationNumber",
    "idNumber",
    "innCode",
    "ogrnCode",
    "kppCode",
    "npiCode",
    "uniqueEntityId",
    "taxNumber",
    "passportNumber",
    "leiCode",
    "vatCode",
    "email",
    "phone",
    "cryptoWalletAddress",
)


def build_document(record: dict, processor: Optional[TextProcessor] = None) -> dict:
    """
    Convert one raw entity record into a processed document dict.

    Parameters
    ----------
    record : dict
        A single parsed JSON record from targets.nested.json.
    processor : TextProcessor, optional
        Shared TextProcessor instance. If None, a module-level default is used.
        Always pass a pre-initialised instance in batch processing to avoid
        reloading spaCy for every document.

    Returns
    -------
    dict with keys:
        doc_id, caption, schema, text_blob, tokens, identifiers, metadata
    """
    if processor is None:
        from .text_processing import get_default_processor
        processor = get_default_processor()

    props = record.get("properties", {})

    # ------------------------------------------------------------------
    # 1–2. Name fields: normalize only
    # ------------------------------------------------------------------
    name_values: list[str] = []
    for field in _NAME_FIELDS:
        for val in props.get(field, []):
            if isinstance(val, str) and val.strip():
                name_values.append(val)

    name_text = processor.build_name_text(name_values)

    # ------------------------------------------------------------------
    # 3. Keyword fields: normalize only (short terms, no NLP needed)
    # ------------------------------------------------------------------
    keyword_values: list[str] = []
    for field in _KEYWORD_FIELDS:
        for val in props.get(field, []):
            if isinstance(val, str) and val.strip():
                keyword_values.append(val)

    keyword_text = processor.build_keyword_text(keyword_values)

    # ------------------------------------------------------------------
    # 4. Free-text fields: normalize + lemmatize
    # ------------------------------------------------------------------
    desc_values: list[str] = []
    for field in _DESC_FIELDS:
        for val in props.get(field, []):
            if isinstance(val, str) and val.strip():
                desc_values.append(val)

    desc_text = processor.build_desc_text(desc_values)

    # ------------------------------------------------------------------
    # 6. Nested sanctions: flatten authority + reason (lemmatized)
    # ------------------------------------------------------------------
    sanctions_raw = props.get("sanctions", [])
    sanctions_objects = [s for s in sanctions_raw if isinstance(s, dict)]
    sanctions_text = processor.build_sanctions_text(sanctions_objects)

    # ------------------------------------------------------------------
    # 7. Nested addressEntity: flatten full address (normalized)
    # ------------------------------------------------------------------
    address_entities = props.get("addressEntity", [])
    address_objects = [a for a in address_entities if isinstance(a, dict)]
    address_text = processor.build_address_text(address_objects)

    # ------------------------------------------------------------------
    # Assemble text_blob (order: names → keywords → desc → sanctions → address)
    # ------------------------------------------------------------------
    parts = [p for p in (name_text, keyword_text, desc_text,
                         sanctions_text, address_text) if p]
    text_blob = " ".join(parts)
    tokens = text_blob.split()

    # ------------------------------------------------------------------
    # Identifiers: raw values, never modified
    # ------------------------------------------------------------------
    identifiers: dict[str, list[str]] = {}
    for field in _IDENTIFIER_FIELDS:
        vals = [str(v) for v in props.get(field, []) if v]
        if vals:
            identifiers[field] = vals

    # ------------------------------------------------------------------
    # Metadata: for filtering and faceted search
    # ------------------------------------------------------------------
    # programId is intentionally metadata-only: it is used for structured
    # filtering (exact match on "US-GLOMAG", "EU-Syria", etc.) and must not
    # be normalised into text_blob where it would become "us glomag".
    # Collect programId from both the entity's top-level properties AND
    # from nested sanction sub-objects (which carry their own programId).
    all_program_ids: list[str] = list(props.get("programId", []))
    for s in sanctions_objects:
        for pid in s.get("properties", {}).get("programId", []):
            if pid and pid not in all_program_ids:
                all_program_ids.append(pid)

    metadata: dict = {
        "country":   props.get("country", []),
        "programId": all_program_ids,
        "datasets":  record.get("datasets", []),
    }

    return {
        "doc_id":      record.get("id", ""),
        "caption":     record.get("caption", ""),
        "schema":      record.get("schema", ""),
        "text_blob":   text_blob,
        "tokens":      tokens,
        "identifiers": identifiers,
        "metadata":    metadata,
        "first_seen":  record.get("first_seen", ""),
        "last_seen":   record.get("last_seen", ""),
    }


if __name__ == "__main__":
    # Smoke test with a hand-crafted mini record
    from .text_processing import TextProcessor

    tp = TextProcessor()
    sample = {
        "id": "NK-TEST001",
        "caption": "Viktor Petrov",
        "schema": "Person",
        "datasets": ["ofac_sdn"],
        "first_seen": "2023-01-01T00:00:00",
        "last_seen": "2026-01-01T00:00:00",
        "properties": {
            "name": ["Viktor Petrov", "Виктор Петров"],
            "alias": ["V. Petrov"],
            "notes": ["Russian oligarch evading OFAC sanctions."],
            "topics": ["sanction", "crime"],
            "country": ["ru"],
            "programId": ["OFAC-SDN"],
            "sanctions": [
                {
                    "schema": "Sanction",
                    "properties": {
                        "authority": ["OFAC"],
                        "reason": ["financing terrorism and arms dealing"],
                    },
                }
            ],
            "addressEntity": [
                {
                    "schema": "Address",
                    "properties": {"full": ["Moscow, Russia"], "city": ["Moscow"]},
                }
            ],
        },
    }

    doc = build_document(sample, tp)
    print("=== build_document() smoke test ===\n")
    for k, v in doc.items():
        if k == "tokens":
            print(f"  {k} ({len(v)} tokens): {v[:10]}...")
        else:
            print(f"  {k}: {v}")
