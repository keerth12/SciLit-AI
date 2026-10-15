"""
app.py
------
Streamlit UI. Compared to the previous version, this file got SIMPLER —
most of the logic moved into src/, and app.py's job shrank to: collect
input, call a chain, display output, handle errors gracefully. That's
what "separation of concerns" buys you in practice.
"""

import streamlit as st

from src.config import settings
from src.exceptions import SciLitError
from src.logger import get_logger
from src.rag.chain import ask_question, build_rag_chain
from src.rag.loader import load_pdf
from src.rag.splitter import split_documents
from src.rag.vectorstore import get_or_build_vectorstore
from src.web.search_chain import build_web_search_chain, get_raw_results

logger = get_logger(__name__)

st.set_page_config(page_title="SciLit", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------------
# Session state — same purpose as before: survive Streamlit's reruns.
# ---------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "indexed_file_hash" not in st.session_state:
    st.session_state.indexed_file_hash = None

st.sidebar.title("🔎 SciLit")
mode = st.sidebar.radio("Mode", ["📄 PDF Q&A", "🌐 Web Search"])

if st.sidebar.button("Clear conversation"):
    st.session_state.chat_history = []
    st.rerun()

# ---------------------------------------------------------------------
# PDF mode setup
# ---------------------------------------------------------------------
if mode == "📄 PDF Q&A":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hash(file_bytes)

        # Only rebuild the chain if this is a genuinely new file — Chroma
        # itself also avoids re-embedding a file it's already seen (see
        # vectorstore.py), so this guard mainly saves rebuilding the CHAIN
        # object and re-reading the PDF bytes on every rerun.
        if st.session_state.indexed_file_hash != file_hash:
            try:
                with st.spinner("Reading and indexing your PDF..."):
                    uploaded_file.seek(0)
                    documents = load_pdf(uploaded_file)
                    chunks = split_documents(documents)
                    vectorstore = get_or_build_vectorstore(chunks, file_bytes)
                    st.session_state.rag_chain = build_rag_chain(vectorstore)
                    st.session_state.indexed_file_hash = file_hash
                st.sidebar.success(f"Indexed: {uploaded_file.name} ({len(chunks)} chunks)")
            except SciLitError as e:
                st.sidebar.error(f"Couldn't process this PDF: {e}")
                logger.error(f"PDF processing failed: {e}")
    else:
        st.sidebar.info("Upload a PDF to begin.")

# ---------------------------------------------------------------------
# Chat window — shared by both modes
# ---------------------------------------------------------------------
st.title(mode)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question...")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        answer = None
        try:
            if mode == "📄 PDF Q&A":
                if st.session_state.rag_chain is None:
                    answer = "Please upload a PDF first."
                    st.warning(answer)
                else:
                    with st.spinner("Thinking..."):
                        answer = ask_question(st.session_state.rag_chain, user_question)
                    st.markdown(answer)

            else:  # Web Search mode
                with st.spinner("Searching and thinking..."):
                    chain = build_web_search_chain()
                    answer = chain.invoke(user_question)
                st.markdown(answer)

                with st.expander("Show sources"):
                    for i, r in enumerate(get_raw_results(user_question), start=1):
                        st.markdown(f"**[{i}] [{r['title']}]({r['url']})**\n\n{r['content']}")

        except SciLitError as e:
            answer = f"Something went wrong: {e}"
            st.error(answer)
            logger.error(f"Request failed in {mode}: {e}")

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

with st.sidebar.expander("⚙️ Current config"):
    st.caption(f"LLM: {settings.groq_model}")
    st.caption(f"Embedding model: {settings.embedding_model}")
    st.caption(f"Chunk size: {settings.chunk_size} / overlap: {settings.chunk_overlap}")