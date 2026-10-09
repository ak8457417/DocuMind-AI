from __future__ import annotations

import io
import re
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Normalize extracted PDF text without aggressively changing its meaning."""
    if not text:
        return ""

    # Normalize line endings and common invisible characters.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")

    # Remove trailing whitespace and collapse very long blank-line sequences.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_tables(page: fitz.Page) -> list[str]:
    """
    Extract tables when supported by the installed PyMuPDF version.
    Returns each table as readable Markdown-like text.
    """
    table_texts: list[str] = []

    # Page.find_tables() is available in newer PyMuPDF versions.
    if not hasattr(page, "find_tables"):
        return table_texts

    try:
        table_finder = page.find_tables()
        for table_index, table in enumerate(table_finder.tables, start=1):
            rows = table.extract() or []
            normalized_rows = []
            for row in rows:
                cells = [
                    clean_text(str(cell)) if cell is not None else ""
                    for cell in row
                ]
                if any(cells):
                    normalized_rows.append(cells)

            if not normalized_rows:
                continue

            # Pad rows so the Markdown representation remains rectangular.
            column_count = max(len(row) for row in normalized_rows)
            normalized_rows = [
                row + [""] * (column_count - len(row))
                for row in normalized_rows
            ]

            header = normalized_rows[0]
            separator = ["---"] * column_count
            markdown_rows = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(separator) + " |",
            ]
            for row in normalized_rows[1:]:
                markdown_rows.append("| " + " | ".join(row) + " |")

            table_texts.append(
                f"Table {table_index}:\n" + "\n".join(markdown_rows)
            )
    except Exception:
        # Table extraction is best-effort; it should not break normal text extraction.
        return []

    return table_texts


def process_pdf(
    pdf_bytes: bytes,
    filename: str,
    use_ocr: bool = False,
    include_tables: bool = True,
) -> List[Document]:
    """
    Extract text and tables page by page from a digital/text-based PDF.

    Each returned LangChain Document represents a page's text or one table.
    Metadata includes source filename, 1-based page number, content type,
    and extraction method. OCR is intentionally not implemented in this version.
    The use_ocr parameter remains accepted for compatibility with the current app.py.
    """
    del use_ocr  # Explicitly ignored until OCR is added in a later phase.

    if not pdf_bytes:
        raise ValueError(f"{filename} is empty.")

    try:
        pdf = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open '{filename}' as a valid PDF: {exc}") from exc

    documents: List[Document] = []

    try:
        if pdf.needs_pass:
            raise ValueError(f"'{filename}' is password-protected.")

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            page_text = clean_text(page.get_text("text", sort=True))

            if page_text:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                            "content_type": "text",
                            "extraction_method": "pymupdf",
                        },
                    )
                )

            if include_tables:
                for table_index, table_text in enumerate(_extract_tables(page), start=1):
                    documents.append(
                        Document(
                            page_content=table_text,
                            metadata={
                                "source": filename,
                                "page": page_number,
                                "content_type": "table",
                                "table_index": table_index,
                                "extraction_method": "pymupdf",
                            },
                        )
                    )
    finally:
        pdf.close()

    if not documents:
        raise ValueError(
            f"No selectable text or extractable tables were found in '{filename}'. "
            "This may be a scanned/image-only PDF. OCR is currently disabled."
        )

    return documents
