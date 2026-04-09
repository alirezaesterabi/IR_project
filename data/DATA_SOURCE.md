# OpenSanctions Data Source

## Dataset Information

**Dataset**: OpenSanctions Default

**Version**: March 21, 2026 snapshot

**Official URL**: [https://www.opensanctions.org/datasets/default/](https://www.opensanctions.org/datasets/default/)

## Download Instructions

### Primary Dataset

Download the latest version of `targets.nested.json` from:

[https://www.opensanctions.org/datasets/default/](https://www.opensanctions.org/datasets/default/)

Click on the download link for **targets.nested.json** (approximately 3.68 GB).

### File Placement

After downloading, place the file in:

```text
data/raw_data/targets.nested.json
```

For local reruns, the repository also supports:

- `data/raw_data/sample_10k.json` for quick reruns
- `data/raw_data/sample_100k.json` for larger local validation

## Dataset Overview

- **Total entities**: ~1.2 million sanctioned targets (Mar 22nd 2026)
- **Entity types**: People, Companies, Vessels, Crypto wallets, Legal entities
- **Coverage**: 268 countries
- **Update frequency**: Daily (processes every 6 hours)
- **Format**: Line-delimited JSON (one entity per line)

## Key Fields

Each entity record contains:

- `id`: Unique entity identifier
- `schema`: Entity type (Person, Company, Vessel, etc.)
- `caption`: Primary display name
- `properties`: Nested object with fields like:
  - `name`: Array of name variations
  - `alias`: Array of aliases
  - `description`: Textual descriptions
  - `country`: Country codes
  - `imoNumber`: IMO numbers (for vessels)
  - `mmsi`: MMSI identifiers (for vessels)
  - `programId`: Sanction program identifiers
  - And more...

## Files in This Directory

- `raw_data/targets.nested.json` - full dataset, local only
- `raw_data/sample_10k.json` - tracked 10K raw sample for quick reruns
- `raw_data/sample_100k.json` - larger local-only raw sample
- `DATA_SOURCE.md` - this file

## Notes

- The main `targets.nested.json` file is excluded from git version control.
- `sample_100k.json` is intended for local use and is not part of the default tracked dataset story.
- The canonical rerun guide is `docs/rerun_pipeline.md`.
- Always download the latest full dataset from the official source when you need a full-scale rerun.
