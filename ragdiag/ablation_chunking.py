"""
Chunking ablation: structural vs fixed-size 512-token windows.

The eval set was generated against structural chunks, so gold_chunk_ids are
structural IDs. For fixed chunks, we map each structural gold ID to the set of
fixed-chunk IDs that overlap it — any of those in top-k counts as a hit.

This is conservative: we only call it a hit if a window that actually contains
the answer text is retrieved.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunk_fixed import build as build_fixed, FIXED_CHUNKS_PATH
from .index import DEFAULT_MODEL, BATCH_SIZE, _chunk_to_embed_text
from .metrics import aggregate, failure_decomposition

EVAL_PATH = Path("data/eval/eval_set.json")
CHUNKS_PATH = Path("data/chunks/ai_act_en.json")
RESULTS_DIR = Path("results")

FIXED_INDEX_DIR = Path("data/index_fixed")
FIXED_FAISS = FIXED_INDEX_DIR / "chunks_fixed.faiss"
FIXED_META = FIXED_INDEX_DIR / "meta.json"


def _build_fixed_index(fixed_chunks: list[dict], model_name: str = DEFAULT_MODEL) -> tuple:
    FIXED_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if FIXED_FAISS.exists() and FIXED_META.exists():
        print("[ablation] fixed index cached")
        model = SentenceTransformer(model_name)
        index = faiss.read_index(str(FIXED_FAISS))
        meta = json.loads(FIXED_META.read_text())
        return index, meta["chunk_ids"], model

    print(f"[ablation] embedding {len(fixed_chunks)} fixed chunks …")
    model = SentenceTransformer(model_name)
    texts = [_chunk_to_embed_text(c) for c in fixed_chunks]
    chunk_ids = [c["chunk_id"] for c in fixed_chunks]

    t0 = time.time()
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )
    print(f"[ablation] embedded in {time.time()-t0:.1f}s  shape={embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(FIXED_FAISS))
    FIXED_META.write_text(json.dumps({"chunk_ids": chunk_ids, "model": model_name, "dim": dim}))
    return index, chunk_ids, model


def _gold_fixed_ids(structural_id: str, fixed_chunks: list[dict]) -> set[str]:
    """Return all fixed-chunk IDs derived from a given structural chunk."""
    return {c["chunk_id"] for c in fixed_chunks if c["_source_chunk_id"] == structural_id}


def run(k: int = 10, tag: str = "fixed512") -> tuple:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build fixed chunks
    fixed_chunks = build_fixed()
    fixed_by_id = {c["chunk_id"]: c for c in fixed_chunks}

    # Build index
    index, chunk_ids, model = _build_fixed_index(fixed_chunks)

    pairs = json.loads(EVAL_PATH.read_text())
    verified = [p for p in pairs if p.get("verified") == "pass"]
    eval_pairs = verified if verified else pairs
    print(f"[ablation] {len(eval_pairs)} eval pairs  corpus={len(fixed_chunks)} fixed chunks")

    results = []
    latencies = []

    for i, pair in enumerate(eval_pairs, 1):
        t0 = time.perf_counter()
        vec = model.encode([pair["question"]], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(vec, k)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        retrieved_ids = [chunk_ids[idx] for idx in indices[0] if idx >= 0]

        # Map structural gold IDs → set of valid fixed-chunk IDs
        gold_fixed = set()
        for src_id in pair["gold_chunk_ids"]:
            gold_fixed |= _gold_fixed_ids(src_id, fixed_chunks)

        gold_rank = next(
            (r for r, cid in enumerate(retrieved_ids, 1) if cid in gold_fixed), None
        )

        results.append({
            "id": pair["id"],
            "chunk_id": pair["chunk_id"],
            "chunk_type": pair["chunk_type"],
            "question": pair["question"],
            "gold": list(gold_fixed),           # expanded gold set
            "gold_structural": pair["gold_chunk_ids"],
            "retrieved": retrieved_ids,
            "gold_rank": gold_rank,
            "latency_ms": round(lat, 2),
            "n_gold_windows": len(gold_fixed),
        })

        if i % 20 == 0 or i == len(eval_pairs):
            p50 = sorted(latencies)[len(latencies) // 2]
            print(f"  [{i}/{len(eval_pairs)}] p50_lat={p50:.0f}ms")

    agg = aggregate(results)
    decomp = failure_decomposition(results, k=5)
    p_lat = sorted(latencies)
    n = len(p_lat)

    # Extra: avg gold windows per query (shows how much the corpus expanded)
    avg_gold_windows = sum(r["n_gold_windows"] for r in results) / len(results)

    summary = {
        "tag": tag,
        "n_eval": len(results),
        "model": DEFAULT_MODEL,
        "corpus_size": len(fixed_chunks),
        "window_tokens": 512,
        "overlap_tokens": 128,
        "avg_gold_windows": round(avg_gold_windows, 2),
        "metrics": agg,
        "failure_decomposition_k5": decomp,
        "latency_ms": {
            "p50": round(p_lat[n // 2], 2),
            "p95": round(p_lat[int(n * 0.95)], 2),
            "mean": round(sum(latencies) / n, 2),
        },
    }

    out_path = RESULTS_DIR / f"retrieval_{tag}.json"
    out_path.write_text(json.dumps({"summary": summary, "per_query": results}, indent=2))
    print(f"\n[ablation] saved → {out_path}")
    return results, summary
