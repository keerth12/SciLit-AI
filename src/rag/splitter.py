"""
splitter.py
-----------
Splits Document objects into smaller chunks.

CONCEPT — why RecursiveCharacterTextSplitter specifically:
Last version's chunker sliced text at fixed character positions, which
can cut a sentence in half arbitrarily. RecursiveCharacterTextSplitter is
smarter: it tries to split on paragraph breaks first ("\\n\\n"), then
sentence breaks, then words, only falling back to a hard character cut
as a last resort. Net effect: chunks are much more likely to be
semantically coherent (a full sentence/paragraph) rather than truncated
mid-thought.

Each output chunk is STILL a Document object, so page-number metadata
from loader.py carries through automatically.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Tried in this order until one actually splits the text:
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks