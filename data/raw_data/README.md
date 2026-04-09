# Raw Data Files

This folder contains the raw inputs used by the retrieval pipeline.

## Full OpenSanctions Data

The full dataset file expected by the pipeline is:

- `data/raw_data/targets.nested.json`

Official OpenSanctions dataset page:

- [https://opensanctions.org/datasets/default](https://opensanctions.org/datasets/default)

Recommended "latest" download URL:

- [https://data.opensanctions.org/datasets/latest/default/targets.nested.json](https://data.opensanctions.org/datasets/latest/default/targets.nested.json)

Example dated snapshot URL:

- [https://data.opensanctions.org/datasets/20260407/default/targets.nested.json](https://data.opensanctions.org/datasets/20260407/default/targets.nested.json)

After downloading the full file, place it here:

```text
data/raw_data/targets.nested.json
```

## Local Sample Files

- `sample_10k.json` is the tracked raw sample for quick reruns
- `sample_100k.json` is the larger local-only sample for validation

## Note

The full `targets.nested.json` file is intentionally not tracked in Git because it is very large.
