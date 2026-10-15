"""
logger.py
---------
Structured logging, used instead of print() everywhere else in the app.

WHY NOT JUST print()?
- print() gives no timestamp, no severity, and no way to turn it off
  without deleting code.
- A logger lets you say "only show me WARNING and above" in production,
  while seeing everything (DEBUG) while developing — same code, one
  config change.
- Logs can be written to a file, so if something breaks while you're
  not watching the terminal, you can look back afterward.
"""

import logging
import sys

from src.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Call this once per file: `logger = get_logger(__name__)`.
    Using __name__ means log lines are tagged with which file they came
    from (e.g. "src.rag.chain"), which is invaluable once the app has
    more than a couple of files.
    """
    logger = logging.getLogger(name)

    # Guard against adding duplicate handlers if get_logger() is called
    # more than once for the same name (Streamlit reruns the script a lot).
    if not logger.handlers:
        logger.setLevel(settings.log_level)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger