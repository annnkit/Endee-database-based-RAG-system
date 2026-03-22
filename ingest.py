"""
ingest.py - Load documents → chunk → embed → upsert into Endee vector database
"""

import os
import uuid
import json
import argparse
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    import fitz          # PyMuPDF – PDF support
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ── Config ──────────────────────────────────────────────────────────────────
ENDEE_URL   = os.getenv("ENDEE_URL", "http://localhost:8080")
COLLECTION  = "research_papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE  = 400          # tokens (approx words)
CHUNK_OVERLAP = 50

embedder = SentenceTransformer(EMBED_MODEL)


# ── Text helpers ──────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 30]   # drop tiny fragments


def extract_text_from_pdf(path: Path) -> list[tuple[str, int]]:
    """Return list of (text, page_number) tuples."""
    if not PDF_SUPPORT:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")
    doc    = fitz.open(str(path))
    pages  = []
    for i, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if text:
            pages.append((text, i))
    return pages


def extract_text_from_txt(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(text, 1)]


# ── Endee helpers ─────────────────────────────────────────────────────────
def create_collection_if_needed(dim: int = 384):
    """Create the Endee collection (idempotent)."""
    payload = {
        "name": COLLECTION,
        "dimension": dim,
        "distance": "cosine",
    }
    resp = requests.post(f"{ENDEE_URL}/collections", json=payload, timeout=10)
    if resp.status_code not in (200, 201, 409):   # 409 = already exists
        resp.raise_for_status()
    print(f"✓ Collection '{COLLECTION}' ready (dim={dim})")


def upsert_vectors(vectors: list[dict]):
    """Bulk-upsert a list of {id, vector, metadata} dicts into Endee."""
    payload = {"collection": COLLECTION, "vectors": vectors}
    resp = requests.post(f"{ENDEE_URL}/upsert", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Main ingestion logic ──────────────────────────────────────────────────
def ingest_file(path: Path, batch_size: int = 64) -> int:
    path   = Path(path)
    suffix = path.suffix.lower()
    title  = path.stem

    if suffix == ".pdf":
        pages = extract_text_from_pdf(path)
    elif suffix in (".txt", ".md"):
        pages = extract_text_from_txt(path)
    else:
        print(f"  Skipping unsupported format: {path.name}")
        return 0

    records = []
    for page_text, page_num in pages:
        for chunk in chunk_text(page_text):
            vec = embedder.encode(chunk).tolist()
            records.append({
                "id":       str(uuid.uuid4()),
                "vector":   vec,
                "metadata": {
                    "text":   chunk,
                    "title":  title,
                    "page":   page_num,
                    "source": str(path),
                },
            })

    # Upsert in batches
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        upsert_vectors(batch)
        total += len(batch)
        print(f"  Upserted {total}/{len(records)} chunks …", end="\r")

    print(f"\n✓ {path.name}: {total} chunks ingested")
    return total


def ingest_directory(data_dir: str):
    data_dir = Path(data_dir)
    files    = list(data_dir.glob("**/*.pdf")) + \
               list(data_dir.glob("**/*.txt")) + \
               list(data_dir.glob("**/*.md"))

    if not files:
        print(f"No supported files found in {data_dir}")
        return

    # Create collection using first embedding dimension
    sample_vec = embedder.encode("test")
    create_collection_if_needed(dim=len(sample_vec))

    grand_total = 0
    for f in files:
        print(f"\nProcessing: {f.name}")
        grand_total += ingest_file(f)

    print(f"\n{'='*50}")
    print(f"Ingestion complete. Total chunks: {grand_total}")


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Endee")
    parser.add_argument("--data-dir", default="data", help="Directory with documents")
    args = parser.parse_args()
    ingest_directory(args.data_dir)
