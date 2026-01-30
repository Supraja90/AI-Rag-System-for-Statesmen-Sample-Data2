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
    return text if len(text) <= limit else text[:limit] + "…"


def build_prompt(query: str, contexts: List[str], citations: List[str]) -> str:
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
        /* Page background */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #f2f2f2 !important;
        }

        .block-container {
            padding-top: 0;
            padding-bottom: 2.5rem;
            max-width: 1000px;
        }

        /* First content block = red hero card in the middle */
        .block-container > div:nth-child(1) {
            background: #990000;
            border-radius: 18px;
            padding: 1.8rem 1.8rem 2.0rem 1.8rem;
            margin:  2rem auto;
            max-width: 900px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.3);
            color: white;
        } 

        .hero-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.0rem;
        }

        .hero-logo {
            height: 54px;
            width: auto;
            border-radius: 4px;
            background: white;
            padding: 2px 4px;
        }

        .hero-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }

        .hero-subtitle {
            font-size: 0.95rem;
            opacity: 0.95;
        }

        .hero-search-row {
            margin-top: 0.4rem;
            margin-bottom: 1.2rem;
        }

        /* Search widgets in hero */
        .stTextInput > div > div > input {
            border-radius: 999px;
            border: none;
            padding-left: 1rem;
        }

        .stButton > button {
            background-color: #ffffff !important;
            color: #990000 !important;
            border-radius: 999px;
            border: 0;
            padding: 0.45rem 1.6rem;
            font-weight: 600;
            transition: 0.2s;
        }
        .stButton > button:hover {
            background-color: #f3f4f6 !important;
        }

        /* Answer white card */
        .answer-box {
            background: #ffffff;
            color: #222222 !important;
            padding: 1.2rem 1.4rem;
            border-radius: 12px;
            margin-top: 0.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            font-size: 1.05rem;
            line-height: 1.55rem;
        }

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
    if not LOGO_PATH.exists():
        return ""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ----------------------------------------
# Hero (red card with search + answer)
# ----------------------------------------
def render_hero() -> tuple[str, bool]:
    hero_container = st.container()

    with hero_container:
        # Logo + title
        b64 = logo_base64()
        logo_html = (
            f'<img src="data:image/png;base64,{b64}" class="hero-logo" />'
            if b64
            else "📰"
        )

        st.markdown(
            f"""
            <div class="hero-header">
                {logo_html}
                <div>
                    <div class="hero-title">Statesman RAG Search</div>
                    <div class="hero-subtitle">
                        Search the <em>Statesman</em> student newspaper with retrieval-augmented generation.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Search row
        st.markdown('<div class="hero-search-row">', unsafe_allow_html=True)
        cols = st.columns([4, 1])
        with cols[0]:
            query = st.text_input(
                "",
                placeholder="Ask a question about the Statesman archive...",
                label_visibility="collapsed",
            )
        with cols[1]:
            ask = st.button("Search", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    return query, ask, hero_container


# ----------------------------------------
# MAIN UI
# ----------------------------------------
def main():
    st.set_page_config(page_title="DEBUG", layout="wide")

    st.set_page_config(
        page_title="Statesman RAG – Stony Brook",
        page_icon="📰",
        layout="wide",
    )
    load_css()

    # Sidebar: model settings
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

    # Hero card
    # Warm up retriever ONCE with feedback
    with st.spinner("Loading search index (first time may take a minute)..."):
        retriever = get_retriever()

    # Hero card
    query, ask, hero_container = render_hero()


    # Answer rendered inside same red hero container
    if ask and not query.strip():
        with hero_container:
            st.warning("Please enter a question.")

    if ask and query.strip():
    

        with hero_container:
            with st.spinner("🔍 Retrieving relevant passages…"):
                hits = retriever.search(query)
                if not hits:
                    st.error("No results found. Try rephrasing the question.")
                    st.markdown(
                        '<div class="sbu-footer">© 2025 · Stony Brook University · Statesman RAG Search</div>',
                        unsafe_allow_html=True,
                    )
                    return

                contexts_all = [read_chunk_text(h) for h in hits]
                contexts = [trim(c) for c in contexts_all if c]
                if not contexts:
                    st.error("No readable context. Check ingestion.")
                    st.markdown(
                        '<div class="sbu-footer">© 2025 · Stony Brook University · Statesman RAG Search</div>',
                        unsafe_allow_html=True,
                    )
                    return

                citations = [
                    f"{h['pdf']} p.{h.get('page', '?')} ({h['chunk_id']})"
                    for h in hits
                ]

            with st.spinner("🤖 Generating answer…"):
                answer = generate_answer(query, contexts, citations)

            st.subheader("Answer")
            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

    # Footer (outside hero)
    st.markdown(
        '<div class="sbu-footer">© 2025 · Stony Brook University · Statesman RAG Search</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
