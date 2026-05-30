"""
Run retrieval over the eval set and produce per-query result records.

Output: results/retrieval_results.json
"""

import json
import time
from pathlib import Path

from .index import load, search
from .metrics import aggregate, failure_decomposition

EVAL_PATH = Path("data/eval/eval_set.json")
CHUNKS_PATH = Path("data/chunks/ai_act_en.json")
RESULTS_DIR = Path("results")
TOP_K = 10  # retrieve up to 10; metrics computed at 1,3,5,10


def run(k: int = TOP_K, tag: str = "baseline") -> list[dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(EVAL_PATH.read_text())
    # Only use verified-pass pairs if any have been verified; else use all
    verified = [p for p in pairs if p.get("verified") == "pass"]
    eval_pairs = verified if verified else pairs
    print(f"[retrieve] {len(eval_pairs)} eval pairs  (verified={len(verified)})")

    print("[retrieve] loading index …")
    index, chunk_ids, model = load()

    chunks_by_id = {}
    for c in json.loads(CHUNKS_PATH.read_text()):
        chunks_by_id[c["chunk_id"]] = c

    results = []
    latencies = []

    for i, pair in enumerate(eval_pairs, 1):
        t0 = time.perf_counter()
        hits = search(pair["question"], index, chunk_ids, model, k=k)
        lat = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(lat)

        retrieved_ids = [cid for cid, _ in hits]
        gold = pair["gold_chunk_ids"]
        gold_set = set(gold)

        gold_rank = None
        for rank, cid in enumerate(retrieved_ids, 1):
            if cid in gold_set:
                gold_rank = rank
                break

        results.append(
            {
                "id": pair["id"],
                "chunk_id": pair["chunk_id"],
                "chunk_type": pair["chunk_type"],
                "question": pair["question"],
                "gold": gold,
                "retrieved": retrieved_ids,
                "gold_rank": gold_rank,
                "latency_ms": round(lat, 2),
                "scores": [round(s, 4) for _, s in hits],
            }
        )

        if i % 20 == 0 or i == len(eval_pairs):
            print(f"  [{i}/{len(eval_pairs)}] p50_lat={sorted(latencies)[len(latencies)//2]:.0f}ms")

    # Aggregate metrics
    agg = aggregate(results)
    decomp = failure_decomposition(results, k=5)

    p_latencies = sorted(latencies)
    n = len(p_latencies)

    summary = {
        "tag": tag,
        "n_eval": len(results),
        "model": "BAAI/bge-small-en-v1.5",
        "top_k_retrieved": k,
        "metrics": agg,
        "failure_decomposition_k5": decomp,
        "latency_ms": {
            "p50": round(p_latencies[n // 2], 2),
            "p95": round(p_latencies[int(n * 0.95)], 2),
            "mean": round(sum(latencies) / n, 2),
        },
    }

    out_path = RESULTS_DIR / f"retrieval_{tag}.json"
    out_path.write_text(json.dumps({"summary": summary, "per_query": results}, indent=2))
    print(f"\n[retrieve] saved → {out_path}")
    return results, summary
