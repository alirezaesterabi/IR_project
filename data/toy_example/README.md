# Toy Example Dataset — `documents.jsonl`

## What is this file?

`documents.jsonl` is a **line-delimited JSON** file containing **100 pre-processed
OpenSanctions entity records** derived from `data/raw_data/sample_targets.json`.

It is the **shared "running example"** used across all learning notebooks in
`learning/modules/`. By loading this one file you get the same corpus everywhere,
so concepts built in one notebook carry directly into the next.

## How to generate / regenerate

Run every cell in:

```
learning/modules/01_text_processing_indexing_retrieval/01_preprocessing.ipynb
```

The final cells of that notebook flatten `sample_targets.json` through the full
preprocessing pipeline and write `documents.jsonl` here.

## Record schema

Each line is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | Original OpenSanctions entity ID (e.g. `NK-...`) |
| `caption` | `str` | Human-readable primary name |
| `schema` | `str` | Entity type: `Person`, `Company`, `Vessel`, `LegalEntity`, … |
| `text_blob` | `str` | All preprocessed tokens joined by spaces (ready for full-text search) |
| `tokens` | `list[str]` | Preprocessed token list (name tokens + lemmatised description + topics) |
| `identifiers` | `dict` | Raw vessel/registry codes, e.g. `{"imoNumber": ["IMO9553359"]}` |
| `metadata.country` | `list[str]` | ISO-2 country codes |
| `metadata.topics` | `list[str]` | Sanction topic keywords (e.g. `["sanction", "crime"]`) |
| `metadata.datasets` | `list[str]` | Source dataset IDs |

## How to load in any notebook

```python
from pathlib import Path
import json

# Adjust the relative path as needed
DOCS_PATH = Path("data/toy_example/documents.jsonl")
docs = [json.loads(line) for line in DOCS_PATH.open(encoding="utf-8")]

print(f"Loaded {len(docs)} documents")
print(docs[0].keys())
```

## Preprocessing decisions

| Field type | Tokenised? | Normalised? | Stop words removed? | Lemmatised? |
|------------|-----------|-------------|--------------------|-----------:|
| `name`, `alias` | ✓ | ✓ | ✗ | ✗ |
| `notes`, `description` | ✓ | ✓ | ✓ | ✓ |
| `topics`, `sector` | ✓ | ✓ | ✗ | ✗ |
| `imoNumber`, `mmsi`, … | stored raw | ✗ | ✗ | ✗ |

See `src/preprocessing/document_builder.py` for the production-scale version of
the same logic.
