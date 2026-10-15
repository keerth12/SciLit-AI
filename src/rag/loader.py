"""
loader.py
---------
Turns an uploaded PDF into a list of LangChain Document objects.

CONCEPT — the Document object:
LangChain represents a piece of text as a `Document`, which bundles:
  - page_content: the actual text
  - metadata: a dict of extra info (page number, source filename, etc.)
This metadata TRAVELS WITH THE TEXT through chunking, embedding, and
retrieval. That's what lets us later say "this answer came from page 4"
without any extra bookkeeping — PyPDFLoader fills in metadata={"page": 3}
automatically per page.
"""

import tempfile

from langchain_community.document_loaders import PyPDFLoader

from src.exceptions import PDFProcessingError
from src.logger import get_logger

logger = get_logger(__name__)


def load_pdf(uploaded_file) -> list:
    """
    Takes a Streamlit UploadedFile object and returns a list of
    LangChain Document objects, one per page.

    WHY THE TEMP FILE?
    PyPDFLoader expects a file PATH on disk, but Streamlit gives us an
    in-memory file-like object. We write it to a temporary file so
    PyPDFLoader has something to point at, then it's auto-deleted when
    the `with` block exits.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file.flush()  # ensure bytes are actually written before reading them back

            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()

        if not documents or all(not doc.page_content.strip() for doc in documents):
            raise PDFProcessingError(
                "No extractable text found. This PDF may be a scanned image "
                "without a text layer (would need OCR)."
            )

        logger.info(f"Loaded {len(documents)} pages from '{uploaded_file.name}'")
        return documents

    except PDFProcessingError:
        raise  # re-raise our own error type as-is
    except Exception as e:
        logger.error(f"Failed to load PDF '{uploaded_file.name}': {e}")
        raise PDFProcessingError(f"Could not read PDF: {e}") from e