"""Structured-file ingestion: CSV, JSON (array of objects), JSONL.

Treats each row/object as one indexable item, not as text to chunk. The
schema is inferred from a sample of rows (type per column, sample values,
unique/null counts) and surfaced to the UI for review before commit.

No third-party deps — stdlib csv + json only. Excel/Parquet would need
openpyxl/pyarrow respectively; not in scope here.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


# ---------- Format detection ----------

def detect_format(filename: str, head_bytes: bytes) -> str | None:
    """Return one of: 'csv', 'json', 'jsonl', or None if unrecognized.

    Extension is the primary signal; content is sniffed only as a tiebreaker
    (helps when a CSV is named .txt, or a JSONL is named .json).
    """
    ext = Path(filename).suffix.lower()
    head = head_bytes[:4096].decode("utf-8", errors="replace").lstrip()
    if ext == ".csv" or ext == ".tsv":
        return "csv"
    if ext == ".jsonl" or ext == ".ndjson":
        return "jsonl"
    if ext == ".json":
        # An array-of-objects JSON. A single object isn't structured for our
        # purposes (no rows), and we don't try to flatten arbitrary trees.
        if head.startswith("["):
            return "json"
        # Could still be JSONL if the file is multiple compact objects per line.
        if head.startswith("{") and "\n{" in head:
            return "jsonl"
        return None
    # No extension match — sniff content.
    if head.startswith("["):
        return "json"
    if head.startswith("{") and "\n{" in head:
        return "jsonl"
    if "," in head.split("\n", 1)[0] and "\n" in head:
        return "csv"
    return None


# ---------- Type inference ----------

# Cheap regexes for type guessing. Anything that fails all of these is `string`.
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$|^-?\d*\.\d+(?:[eE][-+]?\d+)?$|^-?\d+[eE][-+]?\d+$")
_BOOL_VALS = {"true", "false", "yes", "no", "y", "n", "0", "1"}
_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)


def _guess_one(v: Any) -> str:
    """Type guess for a single raw value."""
    if v is None or v == "":
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "json"
    s = str(v).strip()
    if not s:
        return "null"
    if _INT_RE.match(s):
        return "int"
    if _FLOAT_RE.match(s):
        return "float"
    if s.lower() in _BOOL_VALS and s.lower() in ("true", "false"):
        return "bool"
    if _DATE_RE.match(s):
        return "date"
    return "string"


# Promotion rules — combine per-row guesses into a column type.
# int < float (int values fit in a float column);
# anything mixed with string becomes string.
_PROMOTE = {
    ("null", "null"): "null",
    ("int", "int"): "int",
    ("float", "float"): "float",
    ("bool", "bool"): "bool",
    ("date", "date"): "date",
    ("list", "list"): "list",
    ("json", "json"): "json",
    ("string", "string"): "string",
    ("int", "float"): "float",
    ("float", "int"): "float",
}


def _combine(a: str, b: str) -> str:
    if a == "null": return b
    if b == "null": return a
    return _PROMOTE.get((a, b)) or _PROMOTE.get((b, a)) or "string"


# ---------- Inference output ----------

@dataclass
class ColumnInfo:
    name: str
    type: str = "string"           # inferred final type
    null_count: int = 0
    unique_count: int = 0
    sample_values: list = field(default_factory=list)  # up to 5 distinct non-null examples


@dataclass
class StructuredPreview:
    format: str                    # csv|json|jsonl
    row_count: int                 # actual rows parsed (preview is sample, count is full)
    truncated: bool                # True if we stopped at the sample cap
    columns: list[ColumnInfo]
    preview_rows: list[dict]       # first ~50 rows as dicts


# Hard cap on rows read for preview AND inference — keeps a 100k-row CSV
# preview cheap. Commit reads the full file.
_PREVIEW_ROW_CAP = 5_000


def parse_iter(file_bytes: bytes, fmt: str) -> Iterator[dict]:
    """Stream rows as dicts, regardless of format."""
    if fmt == "csv":
        # csv module wants a text stream
        text = file_bytes.decode("utf-8-sig", errors="replace")
        # Detect delimiter: prefer ',' but accept '\t' / ';'
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        for row in reader:
            yield {k: (v if v != "" else None) for k, v in row.items() if k is not None}
    elif fmt == "json":
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        if not isinstance(data, list):
            raise ValueError("JSON file must be an array of objects")
        for obj in data:
            if isinstance(obj, dict):
                yield obj
    elif fmt == "jsonl":
        for line in file_bytes.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
    else:
        raise ValueError(f"Unknown format: {fmt}")


def infer_schema(file_bytes: bytes, filename: str) -> StructuredPreview:
    """Read the file (up to the preview cap) and return inferred schema + sample."""
    fmt = detect_format(filename, file_bytes)
    if fmt is None:
        raise ValueError(
            f"Could not detect structured format for {filename}. "
            "Supported: .csv, .tsv, .json (array), .jsonl, .ndjson"
        )

    columns: dict[str, ColumnInfo] = {}
    column_order: list[str] = []
    types_seen: dict[str, str] = {}     # name -> currently-promoted type
    unique_values: dict[str, set] = {}  # name -> set of values seen (capped)

    preview_rows: list[dict] = []
    row_count = 0
    truncated = False

    for row in parse_iter(file_bytes, fmt):
        if row_count >= _PREVIEW_ROW_CAP:
            truncated = True
            break
        for key, val in row.items():
            if key not in columns:
                columns[key] = ColumnInfo(name=key)
                column_order.append(key)
                types_seen[key] = "null"
                unique_values[key] = set()
            ci = columns[key]
            t = _guess_one(val)
            if t == "null":
                ci.null_count += 1
            else:
                types_seen[key] = _combine(types_seen[key], t)
                # Track up to 1000 distinct values for unique_count + samples
                if len(unique_values[key]) < 1000:
                    try:
                        unique_values[key].add(val if isinstance(val, (str, int, float, bool)) else str(val))
                    except TypeError:
                        pass
        if len(preview_rows) < 50:
            preview_rows.append(row)
        row_count += 1

    for name, ci in columns.items():
        ci.type = types_seen[name] if types_seen[name] != "null" else "string"
        ci.unique_count = len(unique_values[name])
        # Show 5 distinct samples — first by insertion would be nicer but
        # set drops order. Sort for determinism in the UI.
        samples = list(unique_values[name])[:5]
        try:
            samples.sort(key=lambda x: (type(x).__name__, x))
        except TypeError:
            pass
        ci.sample_values = samples

    return StructuredPreview(
        format=fmt,
        row_count=row_count,
        truncated=truncated,
        columns=[columns[n] for n in column_order],
        preview_rows=preview_rows,
    )


def build_row_text(row: dict, content_columns: list[str]) -> str:
    """Concatenate selected columns into the searchable text for a row."""
    parts: list[str] = []
    if not content_columns:
        # Fall back to a "label: value" dump of everything as a last resort
        content_columns = list(row.keys())
    for col in content_columns:
        v = row.get(col)
        if v is None or v == "":
            continue
        if isinstance(v, (list, dict)):
            parts.append(f"{col}: {json.dumps(v, ensure_ascii=False)}")
        else:
            parts.append(f"{col}: {v}")
    return "\n".join(parts)
