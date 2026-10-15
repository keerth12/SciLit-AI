"""
exceptions.py
-------------
Custom exceptions so calling code can distinguish WHAT went wrong without
parsing error message strings.

Example of why this matters: in app.py, we can write
    except PDFProcessingError:
        st.error("Couldn't read that PDF — is it a scanned image?")
    except LLMGenerationError:
        st.error("The AI model didn't respond — check your API key / quota.")
instead of one giant `except Exception` that treats every failure the same.
"""


class SciLitError(Exception):
    """Base class for all custom errors in this app. Lets calling code
    catch 'any error from our own code' with one except clause if needed."""


class PDFProcessingError(SciLitError):
    """Raised when a PDF can't be read or extracted (corrupt file,
    scanned/image-only PDF with no extractable text, etc.)."""


class VectorStoreError(SciLitError):
    """Raised when building or querying the vector store fails."""


class LLMGenerationError(SciLitError):
    """Raised when the LLM call fails (bad API key, rate limit, network)."""


class WebSearchError(SciLitError):
    """Raised when the Tavily web search call fails."""