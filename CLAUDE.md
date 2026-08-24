# Repository Instructions

## Document Reading

When analyzing local binary documents such as PDF, DOCX, XLSX, or PPTX, do not assume raw file access means the document has been correctly understood. Use the shared document reader in `tools/document_reader/read_document.py`.

Preserve document structure, tables, workbook sheets, formulas, headings, page numbers, slide text, notes, and metadata where relevant. If parsing is incomplete, truncated, missing an optional dependency, or fails, explicitly report the limitation instead of guessing.

Example:

```bash
python tools/document_reader/read_document.py path/to/file.xlsx --format json
```

Always check the JSON `warnings` array, not just the process exit code. Exit code `0` only means a document object was produced - it does not mean extraction was complete; `2` means the input path/type itself was invalid. If `warnings` mentions a missing optional parser (docling/pymupdf/python-docx/openpyxl/python-pptx), run `pip install -r requirements.txt` and retry before concluding a document has no content. If `warnings` mentions truncation, re-run with a higher `--max-chars`/`--max-rows`/`--max-tables`/`--max-slides` before concluding content is absent.

