# OpenSanctions Data Source

## Dataset Information

**Dataset**: OpenSanctions Default
**Version**: March 21, 2026 snapshot
**Official URL**: https://www.opensanctions.org/datasets/default/

## Download Instructions

### Primary Dataset
Download the latest version of `targets.nested.json` from:
https://www.opensanctions.org/datasets/default/

Click on the download link for **targets.nested.json** (approximately 3.68 GB).

### File Placement
After downloading, place the file in:
```
data/raw_data/targets.nested.json
```

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

- `raw_data/targets.nested.json` - Main dataset (ignored by git due to size)
- `raw_data/sample_targets.json` - Sample of 100 records for testing
- `DATA_SOURCE.md` - This file

## Notes

- The main `targets.nested.json` file is excluded from git version control
- Always download the latest version from the official source
- Sample file is included in git for development/testing purposes
