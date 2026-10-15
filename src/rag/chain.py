"""
chain.py
--------
Assembles the actual RAG chain using LCEL (LangChain Expression Language)
— the `|` pipe syntax.

CONCEPT — LCEL:
You can read `retriever | format_docs | prompt | llm | parser` almost
like a Unix pipe: the output of each step becomes the input of the next.
This replaces the manual "retrieve chunks, join them into a string,
build a prompt, call the API, extract the text" steps we wrote by hand
in the previous version — LangChain handles the plumbing, you declare
the pipeline.

CONCEPT — Retriever:
A Retriever is LangChain's standard wrapper around "given a query,
return relevant Documents." vectorstore.as_retriever() turns our Chroma
store into one. Anything downstream that expects a Retriever doesn't
care whether it's backed by Chroma, FAISS, Pinecone, etc.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.config import settings
from src.exceptions import LLMGenerationError
from src.llm import get_llm
from src.logger import get_logger

logger = get_logger(__name__)

_PROMPT = ChatPromptTemplate.from_template(
    """You are a research assistant answering questions about an uploaded document.
Answer using ONLY the context below. If the context doesn't contain the
answer, say so plainly — do not use outside knowledge or guess.

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs: list[Document]) -> str:
    """
    Turns retrieved Document objects into one text block for the prompt,
    labeling each with its page number (available because loader.py's
    metadata carried through chunking and embedding untouched).
    """
    return "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in docs
    )


def build_rag_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.retrieval_top_k}
    )
    llm = get_llm()

    def retrieve_and_format(question: str):
        docs = retriever.invoke(question)

        print("\n" + "=" * 60)
        print("RETRIEVED CHUNKS")
        print("=" * 60)

        for i, doc in enumerate(docs, start=1):
            print(f"\n--- Chunk {i} | Page {doc.metadata.get('page', '?')} ---")
            print(doc.page_content[:1500])

        print("=" * 60 + "\n")

        return _format_docs(docs)

    chain = (
        {
            "context": RunnableLambda(retrieve_and_format),
            "question": RunnablePassthrough()
        }
        | _PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def ask_question(chain, question: str) -> str:
    """Thin wrapper that adds error handling around chain.invoke()."""
    try:
        return chain.invoke(question)
    except Exception as e:
        logger.error(f"RAG chain failed: {e}")
        raise LLMGenerationError(f"Failed to generate an answer: {e}") from e