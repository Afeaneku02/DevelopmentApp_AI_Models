# Document Reader

Reusable document-ingestion toolkit for Claude Code, Codex, and other local agents.

## Installation

The core TXT, Markdown, JSON, and CSV readers use only the Python standard library and always work. DOCX, XLSX, and PPTX also have a standard-library fallback, but it is intentionally limited (see below). PDF has no standard-library fallback at all: without `docling` or `pymupdf` installed, a PDF is detected but returns no extracted content, only a warning. Treat installing the real parsing libraries as part of setting up this repo, not an optional nice-to-have:

```bash
pip install -r requirements.txt
```

This installs:

- `openpyxl` for Excel workbooks
- `python-docx` for Word documents
- `python-pptx` for PowerPoint decks
- `pymupdf` for PDFs

`docling` (commented out in `requirements.txt`) is genuinely optional: it produces richer PDF-to-Markdown conversion when installed, but `pymupdf` alone is enough for reliable text/page extraction and should be treated as required.

## Usage

```bash
python tools/document_reader/read_document.py path/to/file.xlsx
python tools/document_reader/read_document.py path/to/file.docx --format markdown
python tools/document_reader/read_document.py path/to/file.csv --output extracted.json
```

Limits are available to protect agent context:

```bash
python tools/document_reader/read_document.py workbook.xlsx --max-rows 100 --max-sheets 5 --max-chars 12000
python tools/document_reader/read_document.py deck.pptx --max-slides 20 --table-preview-rows 10
python tools/document_reader/read_document.py report.docx --max-tables 10
```

`--max-rows`, `--max-chars`, `--max-sheets`, `--table-preview-rows`, `--max-tables` (DOCX), and `--max-slides` (PPTX) all default to sane values and are always recorded in `warnings` when they actually truncate something.

## Exit Codes

- `0` - a document object was produced, even if `warnings` is non-empty (missing optional dependency, truncated content, a malformed file that still returned partial output, etc.). Callers must inspect `warnings`, not just the exit code, to know whether extraction was complete.
- `2` - the input itself was invalid: the path does not exist, is not a file, or has an unsupported extension. No meaningful document could be attempted.

## Supported Formats

- PDF: `docling` when installed, otherwise `pymupdf`; warns if neither is installed
- DOCX: `python-docx` when installed, otherwise limited XML extraction (see DOCX fallback note below)
- XLSX: `openpyxl` when installed, otherwise limited XML extraction
- CSV: standard `csv`
- PPTX: `python-pptx` when installed, otherwise limited XML extraction
- TXT, Markdown, JSON: standard library

All text-based parsers (TXT, Markdown, JSON, CSV) decode input as UTF-8 first (a leading UTF-8 BOM is stripped automatically). If a file is not valid UTF-8, they fall back to `cp1252` then `latin-1` (which always succeeds) and record which encoding was actually used in `warnings` and, for TXT/Markdown, in `metadata.encoding`. Non-UTF-8 input is never silently replaced with Unicode replacement characters without a warning.

## Output Schema

All parsers return the same top-level shape:

```json
{
  "file_name": "example.xlsx",
  "file_type": "xlsx",
  "metadata": {},
  "sections": [],
  "tables": [],
  "sheets": [],
  "slides": [],
  "warnings": []
}
```

Warnings are part of the contract. If parsing is incomplete, truncated, missing a dependency, or encounters malformed input, callers must report that limitation.

## Excel Behavior

Excel is preserved as workbook structure, not flattened text. Sheet output includes sheet name, visibility state, used range, headers, preview rows, formulas, merged cells, named ranges when available, and worksheet table ranges when available.

Large workbooks are truncated according to CLI limits, and truncation is explicitly recorded in `warnings`.

## Limitations

- PDF table and heading extraction depends on optional parser quality.
- Scanned/image PDFs may produce little or no text.
- Standard-library DOCX/XLSX/PPTX fallbacks are intentionally limited.
- Very large files are previewed rather than dumped in full.
- Large JSON files include a truncated text preview and set `data_omitted: true` instead of embedding the full parsed object.
- The standard-library DOCX fallback (used only when `python-docx` is not installed) flattens a table nested inside another table's cell into that cell's text rather than representing it as its own nested table, so it is never double-counted as a separate top-level table. `python-docx`, when installed, is unaffected.
- There is currently no way to page/continue past a truncated section - re-run with a higher `--max-chars` (or the matching limit flag) if content is missing; check `warnings` first to know whether truncation actually happened before assuming a section is absent.

## Agent Usage

Claude Code and Codex should use this reader before claiming to understand local binary documents:

```bash
python tools/document_reader/read_document.py local-file.pdf --format json
```

Do not infer structure from raw binary access. Preserve headings, pages, tables, workbook sheets, formulas, slide text, notes, metadata, and warnings where relevant. If extraction fails or is partial, say so explicitly.

Always check `warnings` before trusting the output, regardless of exit code - exit `0` only means a document object was produced, not that extraction was complete (see Exit Codes above). In particular:

- A "no PDF/DOCX/XLSX/PPTX parser is installed" warning means the environment is missing an expected dependency, not that the document has no content - `pip install -r requirements.txt` and re-run before concluding the document is empty.
- A "truncated" warning means content was cut off by a `--max-*` limit, not that the document ends there - re-run with a higher limit for that field if the missing content matters.
