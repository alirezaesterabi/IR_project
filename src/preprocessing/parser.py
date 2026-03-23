"""
Streaming JSONL parser for targets.nested.json (3.7 GB, ~1.2M entities).

Uses a generator so the full file is never loaded into memory — each record
is yielded one at a time, matching the SPIMI streaming principle from Module 1.
"""

import json
import os
from pathlib import Path
from typing import Generator, Optional

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _find_project_root() -> Path:
    """Find repo root by searching upward for data/raw_data/."""
    marker = Path("data") / "raw_data"
    seeds = [Path.cwd().resolve()]
    for key in ("IR_PROJECT_ROOT", "VSCODE_WORKSPACE_FOLDER", "CURSOR_WORKSPACE_FOLDER"):
        v = os.environ.get(key)
        if v:
            seeds.append(Path(v).expanduser().resolve())
    seen: set[Path] = set()
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        for base in [seed, *seed.parents]:
            if (base / marker).is_dir():
                return base
    raise FileNotFoundError(
        f"Could not find {marker}. Set IR_PROJECT_ROOT to your project path."
    )


def stream_records(
    path: Optional[Path] = None,
    max_records: Optional[int] = None,
    show_progress: bool = True,
) -> Generator[dict, None, None]:
    """
    Yield one parsed JSON record per line from a JSONL file.

    Parameters
    ----------
    path : Path, optional
        Path to the JSONL file. Defaults to data/raw_data/targets.nested.json.
    max_records : int, optional
        Stop after yielding this many records. None = stream entire file.
    show_progress : bool
        Display a tqdm progress bar if tqdm is installed.

    Yields
    ------
    dict
        One parsed entity record per call.

    Notes
    -----
    Bad lines (JSON decode errors) are skipped and counted. A summary is
    printed at the end so data quality issues are visible.
    """
    if path is None:
        root = _find_project_root()
        path = root / "data" / "raw_data" / "targets.nested.json"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    bad_lines = 0
    yielded = 0

    file_size = path.stat().st_size
    with open(path, "r", encoding="utf-8") as f:
        if show_progress and _HAS_TQDM:
            iterator = tqdm(f, desc="Streaming records", unit=" lines",
                            total=None, miniters=10_000)
        else:
            iterator = f

        for line in iterator:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            yield record
            yielded += 1

            if max_records is not None and yielded >= max_records:
                break

    if bad_lines > 0:
        print(f"[parser] Warning: skipped {bad_lines} malformed lines in {path.name}")


def extract_subset(
    n: int = 100_000,
    output_path: Optional[Path] = None,
    source_path: Optional[Path] = None,
    show_progress: bool = True,
) -> Path:
    """
    Extract the first N records from targets.nested.json and write them
    as a smaller JSONL file for development and testing.

    Parameters
    ----------
    n : int
        Number of records to extract. Default 100,000.
    output_path : Path, optional
        Destination file. Defaults to data/raw_data/sample_{n}.json.
    source_path : Path, optional
        Source JSONL file. Defaults to targets.nested.json.

    Returns
    -------
    Path
        Path to the written subset file.
    """
    if source_path is None:
        root = _find_project_root()
        source_path = root / "data" / "raw_data" / "targets.nested.json"

    if output_path is None:
        root = _find_project_root()
        label = f"{n // 1000}k" if n >= 1000 else str(n)
        output_path = root / "data" / "raw_data" / f"sample_{label}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for record in stream_records(source_path, max_records=n,
                                     show_progress=show_progress):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[parser] Wrote {written:,} records to {output_path}")
    return output_path


if __name__ == "__main__":
    # Quick smoke test: stream first 5 records and print their IDs
    print("Streaming first 5 records from targets.nested.json:\n")
    for i, record in enumerate(stream_records(max_records=5, show_progress=False)):
        print(f"  {i+1}. id={record.get('id')}  schema={record.get('schema')}  "
              f"caption={record.get('caption')}")
