from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".csv": "csv",
    ".pptx": "pptx",
    ".txt": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
}


@dataclass
class ReaderLimits:
    max_rows: int = 200
    max_chars: int = 20000
    max_sheets: int = 20
    table_preview_rows: int = 25
    max_tables: int = 50
    max_slides: int = 100


@dataclass
class Document:
    file_name: str
    file_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    sheets: list[dict[str, Any]] = field(default_factory=list)
    slides: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "metadata": self.metadata,
            "sections": self.sections,
            "tables": self.tables,
            "sheets": self.sheets,
            "slides": self.slides,
            "warnings": self.warnings,
        }


def detect_file_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or '<none>'}")
    return SUPPORTED_EXTENSIONS[suffix]


def truncate_text(text: str, max_chars: int, warnings: list[str], label: str) -> str:
    if len(text) <= max_chars:
        return text
    warnings.append(f"{label} truncated to {max_chars} characters from {len(text)} characters.")
    return text[:max_chars]


# Encodings tried, in order, after utf-8-sig fails. latin-1 always succeeds
# (it maps every byte 0-255), so it is the guaranteed final fallback and
# content is never silently replaced with U+FFFD without a warning.
_FALLBACK_ENCODINGS = ("cp1252", "latin-1")


def decode_bytes(data: bytes, warnings: list[str], label: str) -> str:
    """Decode file bytes as text without silently corrupting non-UTF-8 content.

    Tries UTF-8 first (via utf-8-sig, so a leading BOM is stripped rather than
    left in the decoded text or -- worse -- causing a plain "utf-8" decode to
    "succeed" with the BOM character still embedded, which then breaks
    json.loads on an otherwise well-formed file). If that fails, tries a short
    list of common fallback encodings and records which one was used, so
    callers always know when the source file was not valid UTF-8 instead of
    getting silent U+FFFD replacement characters with no warning. Use
    `decode_bytes_with_encoding` when the caller also needs to record which
    encoding was actually used.
    """
    text, _ = decode_bytes_with_encoding(data, warnings, label)
    return text


def decode_bytes_with_encoding(data: bytes, warnings: list[str], label: str) -> tuple[str, str]:
    try:
        # utf-8-sig strips a leading BOM if present and otherwise decodes
        # identically to plain utf-8, so it is always the correct first
        # attempt -- never try plain "utf-8" first, or BOM-prefixed input
        # decodes "successfully" with the BOM character still embedded.
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    for encoding in _FALLBACK_ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        warnings.append(f"{label} is not valid UTF-8; decoded using {encoding} instead.")
        return text, encoding
    # Unreachable in practice because latin-1 cannot fail, but keep a fail-safe
    # that still warns rather than returning corrupted text silently.
    warnings.append(f"{label} could not be decoded with any known encoding; some characters were replaced.")
    return data.decode("utf-8", errors="replace"), "utf-8 (replacement)"

