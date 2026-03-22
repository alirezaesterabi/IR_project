# Data Access Guide

## Overview
This document provides instructions for accessing the OpenSanctions Default dataset used in this project.

## Why Data is Not in Git
The dataset files are excluded from version control because:
- Large file sizes (3.68 GB - 10.3 GB per file)
- GitHub has 100MB file size limit
- No need to version control raw data (it's a snapshot)
- Team members can download once and work locally

## Download Instructions

### Option 1: Google Drive (Recommended for Team)
**Team members**: Download from Google Drive shared folder

🔗 **Google Drive Link**: [TO BE ADDED BY ALIREZA]

Files available:
- `targets.nested.json` (3.68 GB)
- `targets.simple.csv` (431.88 MB)
- `entities.ftm.json` (2.43 GB)
- `statements.csv` (10.3 GB)
- `names.txt` (107.43 MB)

### Option 2: Official Source (Direct Download)
Download directly from OpenSanctions official website:

🔗 **Official URL**: https://www.opensanctions.org/datasets/default/

Click on the download links for each file format.

## Dataset Information

### Available Formats

| Format | File Size | Description | Priority |
|--------|-----------|-------------|----------|
| **targets.nested.json** | 3.68 GB | Nested JSON with full entity relationships | ✅ Essential |
| **targets.simple.csv** | 431.88 MB | Simplified CSV format | ✅ Essential |
| **entities.ftm.json** | 2.43 GB | FollowTheMoney structured format | ⭐ Recommended |
| **statements.csv** | 10.3 GB | Granular statement-based CSV | 🤔 Optional |
| **names.txt** | 107.43 MB | Plain text names only | 💡 Nice to have |
| **senzing.json** | 1.42 GB | Senzing-specific format | ❌ Skip |

### Dataset Scope
- **4.1 million entities** across 268 countries
- **1.2 million primary targets** (sanctioned entities)
- Entity types: People, Companies, Vessels, Cryptocurrency wallets, Legal entities, etc.
- **Update frequency**: Daily (dataset processes every 6 hours)
- **Current version**: March 21, 2026 snapshot

## Local Setup

### 1. Create Directory Structure
```bash
cd /path/to/IR_project
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/evaluation
```

### 2. Place Downloaded Files
After downloading from Google Drive or official source, move files to:
```
data/raw/
├── targets.nested.json       # 3.68 GB
├── targets.simple.csv         # 431.88 MB
├── entities.ftm.json          # 2.43 GB (optional)
├── statements.csv             # 10.3 GB (optional)
└── names.txt                  # 107.43 MB (optional)
```

### 3. Verify Files
```bash
ls -lh data/raw/
```

Expected output:
```
-rw-r--r--  3.68G  targets.nested.json
-rw-r--r--  431.88M  targets.simple.csv
-rw-r--r--  2.43G  entities.ftm.json
...
```

### 4. Verify Data Integrity (Optional)
If SHA256 checksums are provided, verify:
```bash
shasum -a 256 data/raw/targets.nested.json
# Compare with provided checksum
```

## Git Ignore
The following directories are excluded from git (see `.gitignore`):
- `data/` - All data files
- `raw_data/` - Alternative raw data location
- `models/` - Trained models and indices
- `results/` - Evaluation results

**Important**: Never commit large data files to git!

## File Descriptions

### targets.nested.json (Primary Dataset)
- **Format**: Nested JSON
- **Structure**: Full entity records with nested relationships
- **Use case**: Complete entity profiles, ownership chains, sanction events
- **Best for**: Relational queries, RAG summarization

### targets.simple.csv
- **Format**: Flat CSV
- **Structure**: One row per entity, simplified fields
- **Use case**: Quick exploration, initial analysis
- **Best for**: Understanding data structure, testing preprocessing

### entities.ftm.json (FollowTheMoney Format)
- **Format**: FollowTheMoney JSON
- **Structure**: Standardized investigative data model
- **Use case**: Cleaner structure than nested JSON
- **Best for**: Graph relationships, entity resolution
- **Documentation**: https://followthemoney.tech/

### statements.csv (Granular Data)
- **Format**: Statement-based CSV
- **Structure**: One row per statement/attribute
- **Use case**: Most detailed view, ground truth construction
- **Best for**: Validation, understanding entity attributes

### names.txt
- **Format**: Plain text
- **Structure**: One name per line
- **Use case**: Quick name lookups, fuzzy matching tests
- **Best for**: Name variant expansion experiments

## Troubleshooting

### Issue: "File too large to download from Google Drive"
**Solution**:
- Use Google Drive desktop app for large files
- Or download via `gdown` command-line tool:
```bash
pip install gdown
gdown --id GOOGLE_DRIVE_FILE_ID -O data/raw/targets.nested.json
```

### Issue: "Not enough disk space"
**Required space**: ~20 GB for all files + ~10 GB for processed data
**Solution**:
- Download only essential files (targets.nested.json + targets.simple.csv = ~4 GB)
- Skip statements.csv (10 GB) if space is limited

### Issue: "Git is tracking data files"
**Solution**:
```bash
git rm -r --cached data/
git commit -m "Remove data files from git"
```

## API Access (Alternative)
Instead of bulk download, you can use OpenSanctions API:
- Full-text search: https://api.opensanctions.org/search/default
- Entity matching: https://api.opensanctions.org/match/default

**Note**: API has rate limits; bulk download recommended for this project.

## Contact
For data access issues, contact:
- **Team lead**: Alireza Esterabi (ec25791@qmul.ac.uk)
- **Team members**: Kieren Sweetman, Marek Chodkiewicz

## References
- OpenSanctions official site: https://www.opensanctions.org/
- Dataset page: https://www.opensanctions.org/datasets/default/
- FollowTheMoney documentation: https://followthemoney.tech/
