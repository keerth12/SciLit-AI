"""
vectorstore.py
--------------
Builds and queries a PERSISTENT Chroma vector store.

CONCEPT — persistence:
The previous version's FAISS index lived only in RAM. Close the app,
and you'd have to re-upload and re-embed the PDF from scratch every time
— wasteful and slow for anything beyond a quick demo.

Chroma, given a `persist_directory`, writes vectors to disk as you add
them. Reopen the app later, point it at the same directory, and the
embeddings are already there — no re-processing needed. This is the
difference between a demo and something you'd actually keep using.

CONCEPT — collections:
Chroma organizes vectors into named "collections" (think: separate
folders). We give each uploaded PDF its own collection, named from a
hash of the file content, so uploading two different PDFs doesn't mix
their chunks together, and re-uploading the SAME pdf reuses its existing
collection instead of re-embedding it.
"""

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.exceptions import VectorStoreError
from src.logger import get_logger

logger = get_logger(__name__)

# Loaded once at import time — loading the embedding model is the slow part.
_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)


def _collection_name_for(file_bytes: bytes) -> str:
    """
    Derives a stable, unique collection name from the file's content.
    Using a content hash (not the filename) means re-uploading the exact
    same PDF twice reuses the same collection — no duplicate embedding
    work — while two different files with the same NAME don't collide.
    """
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"pdf_{digest}"


def get_or_build_vectorstore(chunks: list[Document], file_bytes: bytes) -> Chroma:
    """
    Returns a Chroma vector store for these chunks. If a collection for
    this exact file content already exists on disk, Chroma just opens it
    (fast). Otherwise it embeds the chunks and creates it (slower, one time).
    """
    try:
        collection_name = _collection_name_for(file_bytes)

        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=_embeddings,
            persist_directory=settings.vectorstore_dir,
        )

        # If this collection is empty, this is a new PDF — embed and add it.
        if vectorstore._collection.count() == 0:
            logger.info(f"Embedding {len(chunks)} chunks into new collection '{collection_name}'")
            vectorstore.add_documents(chunks)
        else:
            logger.info(f"Reusing existing collection '{collection_name}' (already embedded)")

        return vectorstore

    except Exception as e:
        logger.error(f"Vector store build failed: {e}")
        raise VectorStoreError(f"Could not build/load vector store: {e}") from e