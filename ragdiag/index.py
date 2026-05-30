"""
Build and persist a FAISS index over the chunk corpus.

Embedding model: BAAI/bge-small-en-v1.5 (local, ~130MB, strong for retrieval).
Index file: data/index/chunks.faiss + data/index/meta.json
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks/ai_act_en.json")
INDEX_DIR = Path("data/index")
FAISS_PATH = INDEX_DIR / "chunks.faiss"
META_PATH = INDEX_DIR / "meta.json"

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def _chunk_to_embed_text(chunk: dict) -> str:
    """Produce the string we embed — title + text gives retrieval a boost."""
    parts = []
    if chunk.get("title"):
        parts.append(chunk["title"])
    parts.append(chunk["text"])
    return " ".join(parts)[:2048]  # stay within model token budget


def build(model_name: str = DEFAULT_MODEL, force: bool = False) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if FAISS_PATH.exists() and META_PATH.exists() and not force:
        print(f"[index] already built → {FAISS_PATH}  (use --force to rebuild)")
        return

    chunks = json.loads(CHUNKS_PATH.read_text())
    print(f"[index] loading model {model_name} …")
    model = SentenceTransformer(model_name)

    texts = [_chunk_to_embed_text(c) for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    print(f"[index] embedding {len(texts)} chunks …")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    )
    elapsed = time.time() - t0
    print(f"[index] embedded in {elapsed:.1f}s  shape={embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine (embeddings normalized)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(FAISS_PATH))
    META_PATH.write_text(
        json.dumps(
            {"chunk_ids": chunk_ids, "model": model_name, "dim": dim, "n": len(chunks)},
            indent=2,
        )
    )
    print(f"[index] saved {index.ntotal} vectors → {FAISS_PATH}")


def load() -> tuple[faiss.Index, list[str], SentenceTransformer]:
    if not FAISS_PATH.exists():
        raise FileNotFoundError("Index not found — run 'python -m ragdiag index' first.")
    meta = json.loads(META_PATH.read_text())
    index = faiss.read_index(str(FAISS_PATH))
    model = SentenceTransformer(meta["model"])
    return index, meta["chunk_ids"], model


def search(
    query: str,
    index: faiss.Index,
    chunk_ids: list[str],
    model: SentenceTransformer,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return [(chunk_id, score), ...] sorted by descending score."""
    vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(vec, k)
    return [(chunk_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i >= 0]
