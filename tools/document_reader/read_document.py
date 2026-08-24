from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.document_reader.models import Document, ReaderLimits, detect_file_type
from tools.document_reader.parsers.csv_parser import parse_csv
from tools.document_reader.parsers.docx_parser import parse_docx
from tools.document_reader.parsers.pdf_parser import parse_pdf
from tools.document_reader.parsers.pptx_parser import parse_pptx
from tools.document_reader.parsers.text_parser import parse_json, parse_text
from tools.document_reader.parsers.xlsx_parser import parse_xlsx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


Parser = Callable[[Path, ReaderLimits], Document]


def read_document(path: str | Path, limits: ReaderLimits | None = None) -> Document:
    source = Path(path)
    limits = limits or ReaderLimits()
    if not source.exists():
        return Document(source.name, "unknown", warnings=[f"File does not exist: {source}"])
    if not source.is_file():
        return Document(source.name, "unknown", warnings=[f"Path is not a file: {source}"])
    try:
        file_type = detect_file_type(source)
    except ValueError as exc:
        return Document(source.name, "unknown", warnings=[str(exc)])

    try:
        if file_type in {"txt", "markdown"}:
            return parse_text(source, file_type, limits)
        if file_type == "json":
            return parse_json(source, limits)
        if file_type == "csv":
            return parse_csv(source, limits)
        if file_type == "xlsx":
            return parse_xlsx(source, limits)
        if file_type == "docx":
            return parse_docx(source, limits)
        if file_type == "pptx":
            return parse_pptx(source, limits)
        if file_type == "pdf":
            return parse_pdf(source, limits)
    except PermissionError as exc:
        return Document(source.name, file_type, warnings=[f"File is unreadable: {exc}"])
    except Exception as exc:
        return Document(source.name, file_type, warnings=[f"Unhandled parser error: {exc}"])
    return Document(source.name, file_type, warnings=[f"No parser registered for file type: {file_type}"])


def to_markdown(document: Document) -> str:
    data = document.to_dict()
    lines = [f"# {data['file_name']}", "", f"- Type: `{data['file_type']}`"]
    if data["warnings"]:
        lines.extend(["", "## Warnings", *[f"- {warning}" for warning in data["warnings"]]])
    if data["metadata"]:
        lines.extend(["", "## Metadata", "```json", json.dumps(data["metadata"], indent=2, ensure_ascii=False), "```"])
    if data["sections"]:
        lines.append("\n## Sections")
        for section in data["sections"]:
            title = section.get("type", "section")
            page = f" page {section['page']}" if "page" in section else ""
            lines.extend([f"\n### {title}{page}", section.get("text", "")])
    if data["tables"]:
        lines.append("\n## Tables")
        for table in data["tables"]:
            lines.extend([f"\n### {table.get('name', 'table')}", "```json", json.dumps(table, indent=2, ensure_ascii=False), "```"])
    if data["sheets"]:
        lines.append("\n## Sheets")
        for sheet in data["sheets"]:
            lines.extend([f"\n### {sheet.get('name')}", "```json", json.dumps(sheet, indent=2, ensure_ascii=False), "```"])
    if data["slides"]:
        lines.append("\n## Slides")
        for slide in data["slides"]:
            lines.extend([f"\n### Slide {slide.get('slide_number')}: {slide.get('title', '')}", "```json", json.dumps(slide, indent=2, ensure_ascii=False), "```"])
    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read local documents into structured JSON or Markdown.")
    parser.add_argument("path", help="Path to the document to read.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format.")
    parser.add_argument("--output", help="Optional output path. Defaults to stdout.")
    parser.add_argument("--max-rows", type=int, default=ReaderLimits.max_rows)
    parser.add_argument("--max-chars", type=int, default=ReaderLimits.max_chars)
    parser.add_argument("--max-sheets", type=int, default=ReaderLimits.max_sheets)
    parser.add_argument("--table-preview-rows", type=int, default=ReaderLimits.table_preview_rows)
    parser.add_argument("--max-tables", type=int, default=ReaderLimits.max_tables, help="Max DOCX tables to include.")
    parser.add_argument("--max-slides", type=int, default=ReaderLimits.max_slides, help="Max PPTX slides to include.")
    args = parser.parse_args(argv)

    limits = ReaderLimits(
        max_rows=args.max_rows,
        max_chars=args.max_chars,
        max_sheets=args.max_sheets,
        table_preview_rows=args.table_preview_rows,
        max_tables=args.max_tables,
        max_slides=args.max_slides,
    )
    document = read_document(args.path, limits)
    output = json.dumps(document.to_dict(), indent=2, ensure_ascii=False) if args.format == "json" else to_markdown(document)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)
    # Exit code contract: 0 means a Document was produced, even if `warnings`
    # is non-empty (missing optional dependency, truncation, partial parse,
    # etc.) -- callers must inspect `warnings` for those. 2 means the input
    # itself was invalid (missing path, not a file, or an unsupported
    # extension) and no meaningful Document could be attempted at all.
    return 2 if document.file_type == "unknown" else 0


if __name__ == "__main__":
    raise SystemExit(main())
