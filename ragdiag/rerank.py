"""
Cross-encoder reranker ablation.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (~80MB, fast on CPU)
Strategy: retrieve top-K with the bi-encoder, rerank top-K with cross-encoder, return top-k.

This is the single knob being varied: reranker on vs off.
All other config (embedding model, chunking, eval set) stays identical to baseline.
"""

import json
import time
from pathlib import Path

from sentence_transformers import CrossEncoder

from .index import load, search
from .metrics import aggregate, failure_decomposition

EVAL_PATH = Path("data/eval/eval_set.json")
CHUNKS_PATH = Path("data/chunks/ai_act_en.json")
RESULTS_DIR = Path("results")

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVE_K = 20   # retrieve wider set for reranker to work with
FINAL_K = 10      # return top-10 after reranking (matches baseline)


def run_reranked(retrieve_k: int = RETRIEVE_K, final_k: int = FINAL_K, tag: str = "reranker") -> tuple:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(EVAL_PATH.read_text())
    verified = [p for p in pairs if p.get("verified") == "pass"]
    eval_pairs = verified if verified else pairs
    print(f"[rerank] {len(eval_pairs)} eval pairs")

    print(f"[rerank] loading bi-encoder index …")
    index, chunk_ids, bi_model = load()

    print(f"[rerank] loading cross-encoder {RERANKER_MODEL} …")
    cross_encoder = CrossEncoder(RERANKER_MODEL, max_length=512)

    chunks_by_id = {}
    for c in json.loads(CHUNKS_PATH.read_text()):
        chunks_by_id[c["chunk_id"]] = c

    results = []
    latencies_bi = []
    latencies_ce = []

    for i, pair in enumerate(eval_pairs, 1):
        query = pair["question"]

        # Stage 1: bi-encoder retrieval
        t0 = time.perf_counter()
        hits = search(query, index, chunk_ids, bi_model, k=retrieve_k)
        lat_bi = (time.perf_counter() - t0) * 1000
        latencies_bi.append(lat_bi)

        candidate_ids = [cid for cid, _ in hits]

        # Stage 2: cross-encoder reranking
        t1 = time.perf_counter()
        ce_inputs = [
            (query, chunks_by_id[cid]["text"][:512])
            for cid in candidate_ids
            if cid in chunks_by_id
        ]
        ce_scores = cross_encoder.predict(ce_inputs)
        lat_ce = (time.perf_counter() - t1) * 1000
        latencies_ce.append(lat_ce)

        # Sort by cross-encoder score descending
        ranked = sorted(zip(candidate_ids, ce_scores), key=lambda x: x[1], reverse=True)
        retrieved_ids = [cid for cid, _ in ranked[:final_k]]

        gold = pair["gold_chunk_ids"]
        gold_set = set(gold)
        gold_rank = next((r for r, cid in enumerate(retrieved_ids, 1) if cid in gold_set), None)

        results.append({
            "id": pair["id"],
            "chunk_id": pair["chunk_id"],
            "chunk_type": pair["chunk_type"],
            "question": pair["question"],
            "gold": gold,
            "retrieved": retrieved_ids,
            "gold_rank": gold_rank,
            "latency_ms_biencoder": round(lat_bi, 2),
            "latency_ms_crossencoder": round(lat_ce, 2),
            "latency_ms_total": round(lat_bi + lat_ce, 2),
            "ce_scores": [round(float(s), 4) for _, s in ranked[:final_k]],
        })

        if i % 20 == 0 or i == len(eval_pairs):
            p50_total = sorted([r["latency_ms_total"] for r in results])[len(results) // 2]
            print(f"  [{i}/{len(eval_pairs)}] p50_total={p50_total:.0f}ms")

    agg = aggregate(results)
    decomp = failure_decomposition(results, k=5)

    total_lats = [r["latency_ms_total"] for r in results]
    p_total = sorted(total_lats)
    n = len(p_total)

    summary = {
        "tag": tag,
        "n_eval": len(results),
        "bi_encoder": "BAAI/bge-small-en-v1.5",
        "cross_encoder": RERANKER_MODEL,
        "retrieve_k": retrieve_k,
        "final_k": final_k,
        "metrics": agg,
        "failure_decomposition_k5": decomp,
        "latency_ms": {
            "p50_total": round(p_total[n // 2], 2),
            "p95_total": round(p_total[int(n * 0.95)], 2),
            "p50_biencoder": round(sorted(latencies_bi)[n // 2], 2),
            "p50_crossencoder": round(sorted(latencies_ce)[n // 2], 2),
        },
    }

    out_path = RESULTS_DIR / f"retrieval_{tag}.json"
    out_path.write_text(json.dumps({"summary": summary, "per_query": results}, indent=2))
    print(f"\n[rerank] saved → {out_path}")
    return results, summary
