import os
import re
import io
import ujson
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import fitz                           # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text

from sentence_transformers import SentenceTransformer
import faiss
import yaml
import nltk

Image.MAX_IMAGE_PIXELS = None
# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

PDF_DIR   = ROOT / CFG["paths"]["pdf_dir"]
STORAGE   = ROOT / CFG["paths"]["storage"]
CHUNK_DIR = ROOT / CFG["paths"]["chunks"]
IMG_DIR   = ROOT / CFG["paths"]["images"]
EMB_DIR   = ROOT / CFG["paths"]["embeddings"]
LOG_DIR   = ROOT / CFG["paths"]["logs"]

for d in [CHUNK_DIR, IMG_DIR, EMB_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TEXT_EMB_MODEL = CFG["models"]["text_embed"]
CLIP_MODEL     = CFG["models"]["clip_image"]

CHUNK_CHARS    = int(CFG["retrieval"]["chunk_chars"])
CHUNK_OVERLAP  = int(CFG["retrieval"]["chunk_overlap"])
USE_SENT_SPLIT = bool(CFG["retrieval"]["use_sentence_split"])

OCR_ONLY_NEEDED     = bool(CFG["runtime"]["ocr_only_if_needed"])
OCR_TEXT_THRESHOLD  = int(CFG["runtime"]["ocr_text_threshold"])


# -------------------------------------------------------
# NLTK Setup
# -------------------------------------------------------
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass


_ensure_nltk()


def safe_sent_tokenize(text: str) -> List[str]:
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        return re.split(r'(?<=[.!?])\s+', text.strip())


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_text_pymupdf(pdf_path: Path) -> Tuple[str, List[int]]:
    text_parts, counts = [], []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            t = page.get_text("text")
            text_parts.append(t)
            counts.append(len(t.encode("utf-8")))
    return clean_text("\n".join(text_parts)), counts


def extract_text_pdfminer(pdf_path: Path) -> str:
    try:
        return clean_text(pdfminer_extract_text(str(pdf_path)))
    except Exception:
        return ""


def extract_pdf_text(pdf_path: Path) -> Tuple[str, List[int]]:
    try:
        t, counts = extract_text_pymupdf(pdf_path)
        if len(t) > 50:
            return t, counts
    except Exception:
        pass
    return extract_text_pdfminer(pdf_path), []


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ocr_with_easyocr(image_bytes: bytes) -> str:
    try:
        import easyocr, torch
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        results = reader.readtext(arr, detail=0)
        return "\n".join(results).strip()
    except Exception as e:
        print("OCR error:", e)
        return ""


def render_page_image(doc, page_idx: int, out_path: Path) -> None:
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=200)
    pix.save(out_path.as_posix())


def extract_images(pdf_path: Path, outdir: Path) -> List[Path]:
    saved = []
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc):
            page_png = outdir / f"{pdf_path.stem}_p{page_idx+1:03d}.png"
            if not page_png.exists():
                render_page_image(doc, page_idx, page_png)
            saved.append(page_png)

            for i, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix2 = fitz.Pixmap(doc, xref)

                if pix2.n >= 5:
                    pix2 = fitz.Pixmap(fitz.csRGB, pix2)

                img_path = outdir / f"{pdf_path.stem}_p{page_idx+1:03d}_fig{i+1}.png"
                if not img_path.exists():
                    pix2.save(img_path.as_posix())
                saved.append(img_path)

    return saved


def sentence_chunks(text: str, max_chars=1200, overlap=200) -> List[str]:
    if not USE_SENT_SPLIT:
        return sliding_chunks(text, max_chars, overlap)

    sents = safe_sent_tokenize(text)
    chunks, cur = [], ""

    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = s if not cur else (cur + " " + s)
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap and len(cur) > overlap else ""
            cur = (tail + " " + s).strip() if tail else s

    if cur.strip():
        chunks.append(cur.strip())

    return chunks


def sliding_chunks(s: str, max_chars: int, overlap: int) -> List[str]:
    s = s.strip()
    chunks, i, n = [], 0, len(s)

    while i < n:
        j = min(i + max_chars, n)
        chunks.append(s[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
        if i >= n:
            break

    return chunks


# -------------------------------------------------------
# Layout-aware extraction
# -------------------------------------------------------
def _page_blocks(page):
    BLK_TEXT = 4
    blocks = page.get_text("blocks") or []

    out = []
    for b in blocks:
        if len(b) < 5:
            continue
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[BLK_TEXT]
        txt = clean_text(txt)
        if txt:
            out.append((x0, y0, x1, y1, txt))

    out.sort(key=lambda t: (round(t[1], 1), round(t[0], 1)))
    return out


def _group_blocks_into_columns(blocks, page_width, split_x=None):
    if not blocks:
        return {}

    if split_x is None:
        split_x = page_width * 0.5

    cols = {0: [], 1: []}

    for b in blocks:
        x0 = b[0]
        col = 0 if x0 < split_x else 1
        cols[col].append(b)

    for c in cols:
        cols[c].sort(key=lambda t: (t[1], t[0]))

    return cols


def _merge_adjacent_blocks(blocks, max_gap_px=25):
    if not blocks:
        return []

    merged = []
    cur_x0, cur_y0, cur_x1, cur_y1, cur_txt = list(blocks[0])

    for (x0, y0, x1, y1, txt) in blocks[1:]:
        gap = y0 - cur_y1

        if gap <= max_gap_px:
            cur_txt = (cur_txt + " " + txt).strip()
            cur_y1 = max(cur_y1, y1)
            cur_x0 = min(cur_x0, x0)
            cur_x1 = max(cur_x1, x1)
        else:
            merged.append((cur_x0, cur_y0, cur_x1, cur_y1, cur_txt))
            cur_x0, cur_y0, cur_x1, cur_y1, cur_txt = x0, y0, x1, y1, txt

    merged.append((cur_x0, cur_y0, cur_x1, cur_y1, cur_txt))
    return merged


def _split_by_title_patterns(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    articles, cur = [], ""

    for ln in lines:
        is_title_case = bool(re.match(r'^[A-Z][A-Za-z]+(\s+[A-Z][A-Za-z]+){2,}$', ln))
        is_all_caps = ln.isupper() and len(ln.split()) >= 3

        if is_title_case or is_all_caps:
            if cur.strip():
                articles.append(cur.strip())
                cur = ""
        cur += (" " + ln)

    if cur.strip():
        articles.append(cur.strip())

    return [a for a in articles if len(a) >= 60]


def extract_article_texts(pdf_path: Path) -> List[Dict[str, Any]]:
    articles = []
    with fitz.open(pdf_path) as doc:
        for pno, page in enumerate(doc, start=1):
            blocks = _page_blocks(page)
            native_bytes = sum(len(b[4].encode("utf-8")) for b in blocks)

            # OCR fallback
            if native_bytes < OCR_TEXT_THRESHOLD and OCR_ONLY_NEEDED:
                outdir = IMG_DIR / pdf_path.stem
                outdir.mkdir(parents=True, exist_ok=True)

                page_png = outdir / f"{pdf_path.stem}_p{pno:03d}.png"
                if not page_png.exists():
                    render_page_image(doc, pno - 1, page_png)

                ocr_dir = outdir / "_ocr"
                ocr_dir.mkdir(parents=True, exist_ok=True)

                h = sha256_bytes(page_png.read_bytes())
                cache_path = ocr_dir / f"{h}.txt"

                if cache_path.exists():
                    ocr_text = cache_path.read_text(encoding="utf-8")
                else:
                    ocr_text = ocr_with_easyocr(page_png.read_bytes())
                    cache_path.write_text(ocr_text, encoding="utf-8")

                ocr_text = clean_text(ocr_text)
                if len(ocr_text) >= 60:
                    articles.append({"page": pno, "text": ocr_text, "bbox": None})
                continue

            if not blocks:
                continue

            cols = _group_blocks_into_columns(blocks, page.rect.width)

            segments = []
            for col_id in sorted(cols.keys()):
                segments += _merge_adjacent_blocks(cols[col_id])

            for (x0, y0, x1, y1, txt) in segments:
                for sa in _split_by_title_patterns(txt):
                    articles.append({"page": pno, "text": sa, "bbox": (x0, y0, x1, y1)})

    return articles


# -------------------------------------------------------
# Embedding Builders
# -------------------------------------------------------
def build_text_embeddings(texts: List[str], model_name=TEXT_EMB_MODEL) -> np.ndarray:
    st = SentenceTransformer(model_name)
    embs = st.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.asarray(embs, dtype=np.float32)


def build_image_embeddings(image_paths: List[Path], model_name=CLIP_MODEL) -> np.ndarray:
    from transformers import CLIPProcessor, CLIPModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    embs = []

    for p in tqdm(image_paths, desc="CLIP image embeddings"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue

        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            feats = model.get_image_features(**inputs)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().numpy()[0].astype(np.float32))

    return np.vstack(embs) if embs else np.zeros((0, 512), dtype=np.float32)


def write_faiss_index(
    vectors: np.ndarray,
    index_path: Path,
    index_type: str = "flat",
    metric: str = "ip",
    hnsw_m: int = 32,
    ivf_nlist: int = 4096
):
    if vectors.size == 0:
        return None

    d = vectors.shape[1]
    met = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2

    if index_type == "flat":
        index = faiss.IndexFlatIP(d) if met == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)

    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m)

    elif index_type == "ivf":
        quant = faiss.IndexFlatIP(d) if met == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quant, d, ivf_nlist, met)
        index.train(vectors)

    else:
        index = faiss.IndexFlatIP(d)

    index.add(vectors)
    faiss.write_index(index, index_path.as_posix())

    return index


# -------------------------------------------------------
# Main Ingest Pipeline
# -------------------------------------------------------
def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}. Put your Statesman PDFs there.")
        return

    all_chunk_texts, all_chunk_meta = [], []
    all_image_paths, all_image_meta = [], []

    for pdf in tqdm(pdfs, desc="Ingest PDFs"):
        # Images for OCR fallback & image retrieval
        out_img_dir = IMG_DIR / pdf.stem
        out_img_dir.mkdir(parents=True, exist_ok=True)

        img_paths = extract_images(pdf, out_img_dir)

        # Layout-aware article extraction
        articles = extract_article_texts(pdf)

        # Chunk each article
        chunks = []
        for ai, art in enumerate(articles):
            art_text = art["text"]
            page_num = art["page"]

            for ci, ch in enumerate(sentence_chunks(art_text, CHUNK_CHARS, CHUNK_OVERLAP)):
                chunks.append({
                    "chunk_id": f"p{page_num:03d}_a{ai:02d}_c{ci:02d}",
                    "page": page_num,
                    "text": ch
                })

        # Save chunk file
        chunk_path = CHUNK_DIR / f"{pdf.stem}.jsonl"
        with open(chunk_path, "w", encoding="utf-8") as fw:
            for ch in chunks:
                item = {
                    "pdf": pdf.name,
                    "chunk_id": ch["chunk_id"],
                    "page": ch["page"],
                    "text": ch["text"]
                }
                fw.write(ujson.dumps(item, ensure_ascii=False) + "\n")

        # Collect for embeddings
        for ch in chunks:
            all_chunk_texts.append(ch["text"])
            all_chunk_meta.append({
                "pdf": pdf.name,
                "chunk_id": ch["chunk_id"],
                "page": ch["page"]
            })

        # Collect images for CLIP
        for p in img_paths:
            all_image_paths.append(p)
            all_image_meta.append({
                "pdf": pdf.name,
                "path": str(p)
            })

    # Build TEXT embeddings
    print("Building TEXT embeddings …")
    text_vecs = (
        SentenceTransformer(TEXT_EMB_MODEL)
        .encode(all_chunk_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        .astype(np.float32)
        if all_chunk_texts else np.zeros((0, 384), np.float32)
    )

    # Build IMAGE embeddings
    print("Building IMAGE embeddings …")
    image_vecs = build_image_embeddings(all_image_paths) if all_image_paths else np.zeros((0, 512), np.float32)

    # Write FAISS + metadata
    (EMB_DIR / "text").mkdir(parents=True, exist_ok=True)
    (EMB_DIR / "image").mkdir(parents=True, exist_ok=True)

    write_faiss_index(
        text_vecs,
        EMB_DIR / "text" / "index.faiss",
        index_type=CFG["index"]["type"],
        metric=CFG["index"]["metric"],
        hnsw_m=int(CFG["index"]["hnsw_m"]),
        ivf_nlist=int(CFG["index"]["ivf_nlist"]),
    )

    with open(EMB_DIR / "text" / "meta.jsonl", "w", encoding="utf-8") as fw:
        for m in all_chunk_meta:
            fw.write(ujson.dumps(m, ensure_ascii=False) + "\n")

    if image_vecs.size:
        write_faiss_index(
            image_vecs,
            EMB_DIR / "image" / "index.faiss",
            index_type="flat",
            metric="ip"
        )
        with open(EMB_DIR / "image" / "meta.jsonl", "w", encoding="utf-8") as fw:
            for m in all_image_meta:
                fw.write(ujson.dumps(m, ensure_ascii=False) + "\n")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
