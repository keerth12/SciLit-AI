# Research Assistant (RAG + Web Search) — Streamlit App

A chatbot with two modes:
- **📄 PDF Q&A** — upload a PDF, ask questions, get answers grounded in that document (RAG).
- **🌐 Web Search** — ask a question, it searches the live web and synthesizes an answer.

## How it's structured (and why)

```
app.py               <- Streamlit UI + mode switching + session state
rag_core.py           <- PDF -> chunks -> embeddings -> FAISS index -> retrieval
web_search_core.py    <- DuckDuckGo search -> formatted context
llm.py                <- sends (question + context) to Claude, gets an answer back
requirements.txt
.env.example
```

Each file has ONE job. `rag_core.py` doesn't know Streamlit exists.
`llm.py` doesn't know or care whether its context came from a PDF or the
web. This separation is what makes the app easy to extend later — e.g. to
add a third mode (say, searching your own notes), you'd write one new
retrieval file and plug it into the same `answer_from_context()` function.

## The RAG concept, in one paragraph

You can't just paste a whole PDF into a prompt (it may be too long, and
LLMs get less accurate as context grows: "lost in the middle" is a real,
studied effect). So instead: break the document into small chunks, turn
each chunk into a vector (a list of numbers capturing its meaning) using
an embedding model, and store those vectors in an index. When a question
comes in, embed the QUESTION the same way, and find which chunk-vectors
are closest to it (cosine similarity). Those chunks are almost certainly
the ones relevant to the question. Hand only those to the LLM. This is
retrieval-augmented generation: retrieval narrows down what's relevant;
generation turns it into a readable answer.

## Setup

1. **Create a virtual environment** (keeps this project's packages separate
   from your system Python):
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API key:**
   ```bash
   cp .env.example .env
   # then edit .env and paste your real ANTHROPIC_API_KEY
   ```
   Get a key at https://console.anthropic.com/ (Settings -> API Keys).
   Note this is a separate account/key from a Claude.ai subscription —
   API usage is billed per-token, so keep an eye on usage while testing.

4. **Run it locally:**
   ```bash
   streamlit run app.py
   ```
   This opens http://localhost:8501 in your browser.

## Deploying to Streamlit Community Cloud

1. Push this project to a GitHub repo (make sure `.env` is in `.gitignore`
   — never commit real API keys).
2. Go to https://share.streamlit.io/, sign in, click "New app", point it
   at your repo and `app.py`.
3. In the app's settings, open **"Secrets"** and add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-real-key"
   ```
   This is Streamlit Cloud's equivalent of the local `.env` file —
   `os.environ.get("ANTHROPIC_API_KEY")` picks it up automatically once
   set here, no code changes needed.
4. Deploy. First load will be slower since it downloads the embedding
   model (~80MB) — that's normal.

## Things worth experimenting with once it's running

- **Chunk size** (`chunk_size` in `rag_core.py`): smaller chunks give more
  precise retrieval but less surrounding context per chunk. Try 400 vs
  1200 and see how answer quality changes on a technical PDF.
- **top_k** in `search_index()`: how many chunks get sent to the LLM.
  Too few -> might miss the answer. Too many -> dilutes relevance and
  costs more tokens.
- **Swap the embedding model**: try `all-mpnet-base-v2` (bigger, slower,
  more accurate) instead of `all-MiniLM-L6-v2` in `rag_core.py`.
- **Add citations for PDF mode too**: track which page each chunk came
  from (pypdf gives you page numbers) and show "Source: page 4" next to
  each excerpt.

## Known limitations (good to know, not bugs)

- Scanned/image-only PDFs won't extract text (would need OCR, e.g.
  `pytesseract`).
- `duckduckgo-search` is free but can be rate-limited under heavy use;
  for production, swap `web_search_core.py` to a paid search API.
- The FAISS index lives in memory only — it's rebuilt if you refresh the
  page or upload a new PDF. Fine for a single-session assistant; you'd
  need to persist it (e.g. save to disk, or use a real vector DB) if you
  wanted the index to survive restarts.
