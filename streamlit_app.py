import os
import base64
from pathlib import Path
from typing import List

import streamlit as st
import ujson
import yaml
import requests

from retrieve import HybridRetriever

# ----------------------------------------
# CONFIG & PATHS
# ----------------------------------------
ROOT = Path(__file__).parent.resolve()
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
CHUNK_DIR = ROOT / CFG["paths"]["chunks"]
PDF_DIR = ROOT / CFG["paths"]["pdf_dir"]

LOGO_PATH = ROOT / "assets" / "sbu_logo.png"


# ----------------------------------------
# Helpers
# ----------------------------------------
def read_chunk_text(meta: dict) -> str:
    """Read the text for a given chunk from the JSONL chunk file."""
    path = CHUNK_DIR / f"{Path(meta['pdf']).stem}.jsonl"
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = ujson.loads(line)
            if row.get("chunk_id") == meta.get("chunk_id"):
                return row.get("text", "")
    return ""


def trim(text: str, limit: int = 900) -> str:
    """Trim long text to a fixed character limit."""
    return text if len(text) <= limit else text[:limit] + "…"


def build_prompt(query: str, contexts: List[str], citations: List[str]) -> str:
    """Build the LLM prompt from query + contexts + citations."""
    ctx_blocks = [f"{c}\n[Citation: {cite}]" for c, cite in zip(contexts, citations)]
    ctx_text = "\n\n---\n\n".join(ctx_blocks[:3])

    return (
        "You are a precise, citation-first assistant.\n"
        "Answer ONLY using facts directly from the context. "
        "If the answer is not present, say that you don't know from the provided documents.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{ctx_text}\n\n"
        "Answer in a short paragraph and include citation tags like [Citation: PDF p.X]."
    )


def generate_answer(query: str, contexts: List[str], citations: List[str]) -> str:
    """
    Call the LLM backend (Ollama/OpenAI-compatible).
    Uses env vars:
      - OLLAMA_BASE (or defaults to chatgpt.microsopht.com/ollama)
      - OLLAMA_API_KEY or OPENAI_API_KEY
      - OLLAMA_MODEL (default: llama3:latest)
    """
    base = (os.getenv("OLLAMA_BASE") or "https://chatgpt.microsopht.com/ollama").rstrip("/")
    api_key = os.getenv("OLLAMA_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("OLLAMA_MODEL", "llama3:latest")

    url = f"{base}/api/chat"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Only use the provided context."},
            {"role": "user", "content": build_prompt(query, contexts, citations)},
        ],
        "stream": False,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code == 401:
            return "(Auth error 401: check OLLAMA_BASE and your API key.)"
        if not r.ok:
            return f"(Gateway error {r.status_code}: {r.text[:250]})"
        data = r.json()
        return data.get("message", {}).get("content") or data.get("response") or "(No text returned)"
    except Exception as e:
        return f"(Gateway request failed: {e})"


@st.cache_resource
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


# ----------------------------------------
# Styling
# ----------------------------------------
def load_css():
    st.markdown(
        """
        <style>
        /* Global background like SBU portal */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #f4f4f4 !important;
        }

        /* Center main content in a white panel */
        .block-container {
            padding-top: 0;
            padding-bottom: 2.5rem;
            max-width: 1000px;
        }

        /* Full-width red banner */
        .sbu-banner-wrapper {
            position: relative;
            left: 50%;
            right: 50%;
            margin-left: -50vw;
            margin-right: -50vw;
            width: 100vw;
            background-color: #990000;
            box-shadow: 0 3px 8px rgba(0,0,0,0.3);
            margin-bottom: 1.5rem;
        }
        .sbu-banner-inner {
            max-width: 1000px;
            margin: 0 auto;
            padding: 0.7rem 1.4rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            color: white;
        }
        .sbu-logo {
            height: 52px;
            width: auto;
            border-radius: 3px;
        }
        .sbu-banner-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .sbu-banner-subtitle {
            font-size: 0.95rem;
            opacity: 0.95;
        }

        /* White content panel */
        .content-panel {
            background: white;
            border-radius: 8px;
            padding: 1.5rem 1.7rem 2rem 1.7rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid #e5e7eb;
        }

        /* Input + button */
        .stTextInput > div > div > input {
            border-radius: 6px;
            border: 1px solid #bfbfbf;
        }
        .stButton > button {
            background-color: #990000 !important;
            color: white !important;
            border-radius: 6px;
            border: 0;
            padding: 0.5rem 1.25rem;
            font-weight: 600;
            transition: 0.2s;
        }
        .stButton > button:hover {
            background-color: #7a0000 !important;
        }

        .answer-box {
            border-left: 4px solid #990000;
            padding-left: 1rem;
            margin-top: 0.5rem;
        }

        .streamlit-expanderHeader {
            font-weight: 600;
        }

        /* Footer */
        .sbu-footer {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 1.5rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def logo_base64() -> str:
    """Return base64 string for logo image, or empty if not found."""
    if not LOGO_PATH.exists():
        return ""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def render_banner():
    """SBU-style full-width banner with logo + title."""
    b64 = logo_base64()
    if b64:
        img_html = f'<img src="data:image/png;base64,{b64}" class="sbu-logo" />'
    else:
        img_html = "📰"

    st.markdown(
        f"""
        <div class="sbu-banner-wrapper">
          <div class="sbu-banner-inner">
            {img_html}
            <div>
              <div class="sbu-banner-title">Statesman RAG Search</div>
              <div class="sbu-banner-subtitle">
                Stony Brook University · Statesman Newspaper Archive Explorer
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------
# MAIN UI
# ----------------------------------------
def main():
    st.set_page_config(
        page_title="Statesman RAG – Stony Brook",
        page_icon="📰",
        layout="wide",
    )
    load_css()
    render_banner()

    # Sidebar – model & stats
    with st.sidebar:
        st.title("Settings")

        base_default = os.getenv("OLLAMA_BASE") or "https://chatgpt.microsopht.com/ollama"
        model_default = os.getenv("OLLAMA_MODEL", "llama3:latest")

        st.caption("Model endpoint (session only):")
        base_url = st.text_input("OLLAMA_BASE", value=base_default)
        model_name = st.text_input("OLLAMA_MODEL", value=model_default)

        os.environ["OLLAMA_BASE"] = base_url
        os.environ["OLLAMA_MODEL"] = model_name

        st.markdown("---")
        try:
            num_pdfs = len(list(PDF_DIR.glob("*.pdf")))
            st.markdown(f"**📄 PDFs loaded:** {num_pdfs}")
        except Exception:
            pass

        st.markdown("---")
        st.caption("Export these variables in your terminal to make them permanent.")

    # Main content panel
    with st.container():
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)

        # Centered search box + button
        center = st.columns([1, 2, 1])
        with center[1]:
            st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)

            query = st.text_input(
                "Query",
                placeholder="Ask a question about the Statesman archive...",
                label_visibility="collapsed",
            )
            ask = st.button("Search", use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        if ask and not query.strip():
            st.warning("Please enter a question.")

        if ask and query.strip():
            retriever = get_retriever()

            with st.spinner("🔍 Retrieving relevant passages…"):
                hits = retriever.search(query)
                if not hits:
                    st.error("No results found. Try rephrasing the question.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                contexts_all = [read_chunk_text(h) for h in hits]
                contexts = [trim(c) for c in contexts_all if c]
                if not contexts:
                    st.error("No readable context. Check ingestion.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                citations = [
                    f"{h['pdf']} p.{h.get('page', '?')} ({h['chunk_id']})"
                    for h in hits
                ]

            with st.spinner("🤖 Generating answer…"):
                answer = generate_answer(query, contexts, citations)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Answer")
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

            with col2:
                st.subheader("Top Sources")
                for i, (h, ctx) in enumerate(zip(hits, contexts_all), start=1):
                    with st.expander(f"Source {i}: {h['pdf']} p.{h.get('page','?')} ({h['chunk_id']})"):
                        st.write(trim(ctx, 1200))

        # Footer
        st.markdown(
            '<div class="sbu-footer">© 2025 · Stony Brook University · Statesman RAG Search</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
