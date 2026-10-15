"""
test_splitter.py
-----------------
A unit test — code that checks OTHER code is correct, automatically.

WHY THIS MATTERS FOR "PRODUCTION LEVEL":
This test needs no API key, no internet connection, and no running
Streamlit app. It runs in under a second and tells you immediately if a
future change to splitter.py broke something. Run it with:
    pytest tests/test_splitter.py -v

As you add more files, you'd add matching test files (test_loader.py,
test_vectorstore.py, etc.) — not to test LangChain itself (it has its
own tests), but to test YOUR logic built around it.
"""

from langchain_core.documents import Document

from src.rag.splitter import split_documents


def test_split_documents_produces_multiple_chunks():
    """A long document should be broken into more than one chunk."""
    long_text = "This is a sentence. " * 200  # ~4000 characters
    doc = Document(page_content=long_text, metadata={"page": 1})

    chunks = split_documents([doc])

    assert len(chunks) > 1


def test_split_documents_preserves_metadata():
    """Chunks must keep the page-number metadata from the source document
    — this is what lets the UI show 'Source: page N' later."""
    doc = Document(page_content="Some content. " * 100, metadata={"page": 5})

    chunks = split_documents([doc])

    assert all(chunk.metadata.get("page") == 5 for chunk in chunks)


def test_split_documents_handles_short_text():
    """A document shorter than chunk_size should still return exactly
    one chunk, not zero and not an error."""
    doc = Document(page_content="Short text.", metadata={"page": 1})

    chunks = split_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."