from __future__ import annotations

from pathlib import Path

from tools.document_reader.models import Document, ReaderLimits


def parse_pdf(path: Path, limits: ReaderLimits) -> Document:
    warnings: list[str] = []
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except ImportError:
        DocumentConverter = None

    if DocumentConverter is not None:
        try:
            result = DocumentConverter().convert(str(path))
            text = result.document.export_to_markdown()
            return Document(
                file_name=path.name,
                file_type="pdf",
                metadata={"parser": "docling"},
                sections=[{"type": "markdown", "text": text[: limits.max_chars]}],
                warnings=warnings + (["PDF text truncated."] if len(text) > limits.max_chars else []),
            )
        except Exception as exc:
            warnings.append(f"docling PDF parse failed: {exc}")

    try:
        # Import the modern `pymupdf` package name rather than the deprecated
        # `fitz` alias: importing `fitz` directly prints a deprecation banner
        # to stdout, which corrupts the JSON contract for any caller that
        # parses this script's stdout.
        import pymupdf as fitz  # type: ignore
    except ImportError:
        return Document(
            file_name=path.name,
            file_type="pdf",
            metadata={},
            warnings=warnings
            + ["No PDF parser is installed. Install docling or pymupdf to extract PDF content."],
        )

    try:
        pdf = fitz.open(path)
        if pdf.is_encrypted:
            return Document(path.name, "pdf", warnings=warnings + ["PDF is encrypted; content was not extracted."])
        sections = []
        chars = 0
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            chars += len(text)
            if chars > limits.max_chars:
                warnings.append(f"PDF text truncated to {limits.max_chars} characters.")
                text = text[: max(0, len(text) - (chars - limits.max_chars))]
                sections.append({"type": "page", "page": page_index, "text": text})
                break
            sections.append({"type": "page", "page": page_index, "text": text})
        if not any(section.get("text", "").strip() for section in sections):
            warnings.append("PDF parser returned no text; document may be scanned or image-based.")
        return Document(path.name, "pdf", metadata={"parser": "pymupdf", "pages": len(pdf)}, sections=sections, warnings=warnings)
    except Exception as exc:
        return Document(path.name, "pdf", warnings=warnings + [f"Could not parse PDF: {exc}"])

