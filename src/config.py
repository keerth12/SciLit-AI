"""
config.py
---------
Centralized, VALIDATED configuration.

WHY THIS EXISTS:
In a hobby script, you'd sprinkle os.environ.get("SOME_KEY") across every
file that needs it. Two problems with that: (1) if a key is missing, you
only find out when that specific line of code finally runs — could be
deep into a user's session. (2) there's no single place to see everything
your app depends on.

pydantic-settings solves both: it reads your .env file ONCE at startup,
checks that every required field is present and the right type, and
raises a clear error immediately if anything's missing. Every other file
imports the `settings` object from here instead of touching os.environ
directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Required — app will refuse to start without these.
    groq_api_key: str
    tavily_api_key: str

    # Optional — sensible defaults, but overridable via .env if needed.
    groq_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    vectorstore_dir: str = "./chroma_db"
    log_level: str = "INFO"

    # Tells pydantic-settings to read from a .env file, matching field
    # names to env var names case-insensitively (groq_api_key <-> GROQ_API_KEY).
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


# Instantiated once, at import time. If .env is missing required keys,
# this line itself raises a clear ValidationError right when the app
# starts — not buried inside some function call later.
settings = Settings()