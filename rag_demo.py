import os
import requests
import yaml
import ujson
from pathlib import Path
from typing import List

from retrieve import HybridRetriever

ROOT = Path(__file__).parent.resolve()
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
CHUNK_DIR = ROOT / CFG["paths"]["chunks"]


def read_chunk_text(meta: dict) -> str:
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
    ctx = []
    for c, cite in zip(contexts, citations):
        ctx.append(f"{c}\n[Citation: {cite}]")
    ctx_block = "\n\n---\n\n".join(ctx[:3])

    return (
        "You are a precise, citation-first assistant.\n"
        "Answer ONLY with facts supported by the context. "
        "If the answer is not present, say: 'I don't know from the provided documents.'\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{ctx_block}\n\n"
        "Answer with short paragraphs and include the citation tags like [Citation: PDF p.X]."
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
            {"role": "system", "content": "You answer strictly from the provided context."},
            {"role": "user", "content": build_prompt(query, contexts, citations)}
        ],
        "stream": False
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code == 401:
            return "(Auth error 401: Check OLLAMA_BASE and your API key.)"
        if not r.ok:
            return f"(Gateway error {r.status_code}: {r.text[:300]})"
        data = r.json()

        return data.get("message", {}).get("content") or data.get("response") or "(No text returned)"
    except Exception as e:
        return f"(Gateway request failed: {e})"


def main():
    R = HybridRetriever()

    print("Ask about the Statesmen documents (Ctrl+C to exit).")
    print("Examples: 'Who founded The Statesman?', 'Tell about student health fee raise'")

    while True:
        q = input("\nQ> ").strip()
        if not q:
            continue

        hits = R.search(q)
        if not hits:
            print("No results found. Try another query.")
            continue

        contexts_all = R.read_texts(hits)
        contexts = [trim(c) for c in contexts_all if c]

        if not contexts:
            print("No readable context; check OCR/index.")
            continue

        citations = [
            f"{h['pdf']} p.{h.get('page','?')} ({h['chunk_id']})"
            for h in hits
        ]

        ans = generate_answer(q, contexts, citations)

        print("\n" + "=" * 80 + "\n")
        print(ans)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
