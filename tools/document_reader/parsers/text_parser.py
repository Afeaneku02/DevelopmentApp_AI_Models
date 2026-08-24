from __future__ import annotations

import json
from pathlib import Path

from tools.document_reader.models import Document, ReaderLimits, decode_bytes, decode_bytes_with_encoding, truncate_text


def parse_text(path: Path, file_type: str, limits: ReaderLimits) -> Document:
    warnings: list[str] = []
    text, encoding = decode_bytes_with_encoding(path.read_bytes(), warnings, "Text content")
    if not text.strip():
        warnings.append("Document is empty.")
    text = truncate_text(text, limits.max_chars, warnings, "Text content")
    return Document(
        file_name=path.name,
        file_type=file_type,
        metadata={"encoding": encoding},
        sections=[{"type": "text", "text": text}],
        warnings=warnings,
    )


def parse_json(path: Path, limits: ReaderLimits) -> Document:
    warnings: list[str] = []
    raw = decode_bytes(path.read_bytes(), warnings, "JSON content")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return Document(
            file_name=path.name,
            file_type="json",
            sections=[{"type": "raw_text", "text": truncate_text(raw, limits.max_chars, warnings, "JSON raw text")}],
            warnings=warnings + [f"Malformed JSON: {exc}"],
        )
    preview_raw = json.dumps(data, indent=2, ensure_ascii=False)
    preview = truncate_text(preview_raw, limits.max_chars, warnings, "JSON preview")
    section = {"type": "json", "text": preview}
    if len(preview_raw) <= limits.max_chars:
        section["data"] = data
    else:
        section["data_omitted"] = True
    return Document(
        file_name=path.name,
        file_type="json",
        metadata={"root_type": type(data).__name__},
        sections=[section],
        warnings=warnings,
    )
