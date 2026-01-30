from pathlib import Path
import ujson
import numpy as np
import faiss
import yaml
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

EMB_DIR = ROOT / CFG["paths"]["embeddings"]
CHUNK_DIR = ROOT / CFG["paths"]["chunks"]



# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
# It reads meta.jsonl file and returns list of dicts.
def _load_meta(meta_path: Path) -> List[dict]:
    out = []
    if not meta_path.exists():
        return out
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(ujson.loads(line))
    return out

#Figures out which .jsonl chunk file to open and retrieves the text.
def _load_chunk_text(meta: dict) -> str:
    path = CHUNK_DIR / f"{Path(meta['pdf']).stem}.jsonl"
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = ujson.loads(line)
            if row.get("chunk_id") == meta.get("chunk_id"):
                return row.get("text", "")
    return ""

#Used to normalize dense and sparse scores before combining them into a hybrid score.

def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float32)
    span = float(arr.max() - arr.min())
    if span <= 1e-9:
        return [0.0 for _ in scores]
    return ((arr - arr.min()) / span).tolist()


# ---------------------------------------------------------------------
# Dense Retriever (FAISS + SentenceTransformer)
# Understands semantic meaning using embeddings.
# ---------------------------------------------------------------------

class TextRetriever:
    def __init__(self):
        idx_path = EMB_DIR / "text" / "index.faiss"
        meta_path = EMB_DIR / "text" / "meta.jsonl"

        self.index = faiss.read_index(idx_path.as_posix())
        self.meta = _load_meta(meta_path)

        self.embedder = SentenceTransformer(CFG["models"]["text_embed"])

    #Prepares the query embedding so FAISS can search it.
    def embed_query(self, query: str) -> np.ndarray:
        v = self.embedder.encode([query], normalize_embeddings=True)
        return v.astype(np.float32)

    #Performs a FAISS search and retrieves the top-k relevant chunks.
    def search(self, query: str, k: int = 5) -> List[dict]:
        v = self.embed_query(query)
        D, I = self.index.search(v, max(k, 1))

        hits = []
        for score, idx in zip(D[0], I[0]):
            m = dict(self.meta[int(idx)])
            m["score_dense"] = float(score)
            hits.append(m)
        return hits


# ---------------------------------------------------------------------
# Sparse Retriever (BM25 with uni/bi/tri-grams)
# Understands keyword matching.
# ---------------------------------------------------------------------
class BM25Retriever:
    def __init__(self):
        self.docs_meta: List[dict] = []
        self.docs_tokens: List[List[str]] = []

        chunk_files = sorted(CHUNK_DIR.glob("*.jsonl"))
        for cf in chunk_files:
            with open(cf, "r", encoding="utf-8") as f:
                for line in f:
                    row = ujson.loads(line)
                    text = row.get("text", "")
                    tokens = self._tokenize(text)

                    self.docs_meta.append({
                        "pdf": row["pdf"],
                        "chunk_id": row["chunk_id"],
                        "page": row.get("page")
                    })
                    self.docs_tokens.append(tokens)

        self.bm25 = BM25Okapi(self.docs_tokens)

    def _tokenize(self, s: str) -> List[str]:
        s = re.sub(r"[^a-zA-Z0-9\s]", " ", s).lower()
        toks = s.split()

        bigrams = [
            toks[i] + "_" + toks[i + 1]
            for i in range(len(toks) - 1)
        ]
        trigrams = [
            toks[i] + "_" + toks[i + 1] + "_" + toks[i + 2]
            for i in range(len(toks) - 2)
        ]

        return toks + bigrams + trigrams
# Performs a BM25 search and retrieves the top-k relevant chunks.
    def search(self, query: str, k: int = 5) -> List[dict]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_idx = np.argsort(scores)[::-1][:k]

        hits = []
        for idx in top_idx:
            m = dict(self.docs_meta[int(idx)])
            m["score_sparse"] = float(scores[int(idx)])
            hits.append(m)

        return hits


# ---------------------------------------------------------------------
# Optional Reranker (CrossEncoder)
# Refines ranking using a cross-encoder (better but slower)
# ---------------------------------------------------------------------

class CrossEncoderReranker:
    def __init__(self):
        self.model_name = CFG["models"]["reranker"]
        self.enabled = bool(CFG["retrieval"]["use_reranker"]) and bool(self.model_name)

        if self.enabled:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: List[dict], top_n: int) -> List[dict]:
        if not self.enabled or not candidates:
            return candidates[:top_n]

        pairs = [(query, _load_chunk_text(c)) for c in candidates]
        scores = self.model.predict(pairs).tolist()

        for c, s in zip(candidates, scores):
            c["score_rerank"] = float(s)

        return sorted(candidates, key=lambda x: x["score_rerank"], reverse=True)[:top_n]


# ---------------------------------------------------------------------
# Hybrid Retriever (Dense + Sparse + Reranker)
# ---------------------------------------------------------------------

class HybridRetriever:
    def __init__(self):
        self.dense = TextRetriever()
        self.sparse = BM25Retriever()

        self.alpha = float(CFG["retrieval"]["hybrid_alpha"])
        self.top_dense = int(CFG["retrieval"]["dense_top_k"])
        self.top_sparse = int(CFG["retrieval"]["sparse_top_k"])
        self.return_k = int(CFG["retrieval"]["return_k"])

        self.reranker = CrossEncoderReranker()
        self.rerank_top_n = int(CFG["retrieval"]["rerank_top_n"])

    def search(self, query: str) -> List[dict]:
        dense_hits = self.dense.search(query, k=self.top_dense)
        sparse_hits = self.sparse.search(query, k=self.top_sparse)

        by_key: Dict[Tuple[str, str], dict] = {}

        # Dense
        for h in dense_hits:
            key = (h["pdf"], h["chunk_id"])
            by_key.setdefault(
                key,
                {"pdf": h["pdf"], "chunk_id": h["chunk_id"], "page": h.get("page"),
                 "score_dense": 0.0, "score_sparse": 0.0}
            )
            by_key[key]["score_dense"] = h["score_dense"]

        # Sparse
        for h in sparse_hits:
            key = (h["pdf"], h["chunk_id"])
            by_key.setdefault(
                key,
                {"pdf": h["pdf"], "chunk_id": h["chunk_id"], "page": h.get("page"),
                 "score_dense": 0.0, "score_sparse": 0.0}
            )
            by_key[key]["score_sparse"] = h["score_sparse"]

        items = list(by_key.values())

        dn = _normalize([x["score_dense"] for x in items])
        sn = _normalize([x["score_sparse"] for x in items])

        for i, it in enumerate(items):
            it["score_hybrid"] = self.alpha * dn[i] + (1.0 - self.alpha) * sn[i]

        items.sort(key=lambda x: x["score_hybrid"], reverse=True)

        if self.reranker.enabled:
            return self.reranker.rerank(query, items, self.rerank_top_n)

        return items[:self.return_k]

    def read_texts(self, metas: List[dict]) -> List[str]:
        return [_load_chunk_text(m) for m in metas]
