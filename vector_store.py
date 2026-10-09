import os
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from chunker import split_documents


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the configured Google embedding model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it to your .env file and restart the app."
        )

    model_name = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "models/gemini-embedding-001",
    )
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
    )


def build_vector_store(page_documents: List[Document]) -> Chroma:
    """
    Chunk page-level documents and index them in a new Chroma collection.

    A unique collection name prevents different indexing operations from
    accidentally sharing a collection. Chroma is in-memory for this starter
    because no persistent directory is configured.
    """
    if not page_documents:
        raise ValueError("No documents were provided for indexing.")

    chunks = split_documents(page_documents)
    if not chunks:
        raise ValueError("No searchable chunks were created from the documents.")

    embeddings = get_embeddings()
    collection_name = f"documind_{uuid.uuid4().hex[:12]}"

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )


def retrieve_documents(
    question: str,
    vector_store: Chroma,
    k: int = 4,
    content_type: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve the top relevant chunks for a question.

    Optional content_type can be 'text' or 'table'. The filter is applied
    against the metadata preserved by document_processor.py and chunker.py.
    """
    if not question or not question.strip():
        return []
    if k < 1:
        raise ValueError("k must be at least 1.")

    search_kwargs = {"k": k}
    if content_type:
        search_kwargs["filter"] = {"content_type": content_type}

    return vector_store.similarity_search(
        question.strip(),
        **search_kwargs,
    )
