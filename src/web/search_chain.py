"""
search_chain.py
----------------
Web search mode, built the same LCEL way as the PDF chain, but backed by
Tavily instead of a vector store.

CONCEPT — why Tavily over raw DuckDuckGo scraping:
Tavily is a search API BUILT for feeding LLMs: it returns clean,
pre-summarized snippets with source URLs (not raw HTML you'd have to
parse yourself), and it's the tool LangChain's own docs recommend for
this exact use case. It has a free tier (1,000 searches/month at the
time of writing — verify current limits at tavily.com when you sign up).
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_tavily import TavilySearch

from src.config import settings
from src.exceptions import WebSearchError
from src.llm import get_llm
from src.logger import get_logger

logger = get_logger(__name__)

_PROMPT = ChatPromptTemplate.from_template(
    """You are a web research assistant.

Answer the user's question using ONLY the web search results provided
below.

Rules:
1. Give a direct and concise answer.
2. Use the information from the search results, not outside knowledge.
3. Cite claims using [1], [2], [3], etc. matching the search result numbers.
4. Do not invent citation numbers.
5. If multiple sources support a claim, you may cite multiple sources.
6. If the search results do not contain enough information to answer,
   clearly say that the search results do not provide enough information.
7. Do not create URLs or citations that are not present in the search results.

Search results:
{context}

Question:
{question}

Answer:"""
)


def _format_results(results: dict) -> str:
    """Format Tavily results into numbered sources for the LLM."""
    items = results.get("results", [])

    formatted = []

    for i, result in enumerate(items, start=1):
        title = result.get("title", "Untitled source")
        url = result.get("url", "")
        content = result.get("content", "")

        formatted.append(
            f"[{i}] {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    return "\n\n".join(formatted)


def build_web_search_chain():
    """
    Returns a runnable chain for web search mode. Same invoke("question")
    interface as the RAG chain — that symmetry is what let app.py treat
    both modes almost identically.
    """
    search_tool = TavilySearch(tavily_api_key=settings.tavily_api_key, max_results=5)
    llm = get_llm()

    def run_search(question: str) -> dict:
        try:
            return {"context": _format_results(search_tool.invoke(question)), "question": question}
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise WebSearchError(f"Web search failed: {e}") from e

    chain = RunnableLambda(run_search) | _PROMPT | llm | StrOutputParser()
    return chain


def get_raw_results(question: str) -> list[dict]:
    """
    Separately fetches raw results (title/url/snippet) for display in the
    UI's "show sources" panel — kept independent of the answer-generation
    chain so the UI can show sources even if formatting logic changes.
    """
    search_tool = TavilySearch(tavily_api_key=settings.tavily_api_key, max_results=5)
    try:
        response = search_tool.invoke(question)
        return response.get("results", [])
    except Exception as e:
        logger.error(f"Tavily raw fetch failed: {e}")
        raise WebSearchError(f"Web search failed: {e}") from e