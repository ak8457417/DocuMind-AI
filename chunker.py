from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split extracted PDF documents into searchable chunks.

    Normal page text is split into overlapping chunks.
    Table documents are kept intact where possible so rows and headers stay together.
    LangChain's split_documents() carries metadata (source, page, content_type, etc.)
    into each resulting chunk.
    """
    if not documents:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    text_documents: List[Document] = []
    table_documents: List[Document] = []

    for document in documents:
        content = (document.page_content or "").strip()
        if not content:
            continue

        content_type = document.metadata.get("content_type", "text")
        cleaned_document = Document(
            page_content=content,
            metadata=dict(document.metadata),
        )

        if content_type == "table":
            table_documents.append(cleaned_document)
        else:
            text_documents.append(cleaned_document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
        strip_whitespace=True,
    )

    chunks: List[Document] = []
    if text_documents:
        chunks.extend(splitter.split_documents(text_documents))

    # Keep each table as one document to preserve its header/row relationships.
    # If a table is unusually large, split it as a fallback and keep metadata.
    for table_document in table_documents:
        if len(table_document.page_content) <= chunk_size:
            chunks.append(table_document)
        else:
            chunks.extend(splitter.split_documents([table_document]))

    # Ensure every chunk has useful metadata for citations/debugging.
    for chunk in chunks:
        chunk.metadata.setdefault("source", "Unknown document")
        chunk.metadata.setdefault("page", "?")
        chunk.metadata.setdefault("content_type", "text")

    return chunks
