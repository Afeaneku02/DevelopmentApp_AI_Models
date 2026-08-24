from __future__ import annotations

import csv
import io
from pathlib import Path

from tools.document_reader.models import Document, ReaderLimits, decode_bytes


def parse_csv(path: Path, limits: ReaderLimits) -> Document:
    warnings: list[str] = []
    rows: list[list[str]] = []
    # decode_bytes handles a leading UTF-8 BOM (and non-UTF-8 fallback) itself.
    text = decode_bytes(path.read_bytes(), warnings, "CSV content")
    handle = io.StringIO(text)
    sample = handle.read(4096)
    handle.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
        warnings.append("Could not detect CSV dialect; using default comma-separated parsing.")
    reader = csv.reader(handle, dialect)
    for index, row in enumerate(reader):
        if index >= limits.max_rows:
            warnings.append(f"CSV rows truncated to {limits.max_rows}.")
            break
        rows.append(row)
    if not rows:
        warnings.append("CSV file is empty.")
    headers = rows[0] if rows else []
    data_rows = rows[1:]
    return Document(
        file_name=path.name,
        file_type="csv",
        metadata={"row_count_preview": len(rows), "column_count": len(headers)},
        tables=[{"name": path.stem, "headers": headers, "rows": data_rows}],
        warnings=warnings,
    )

