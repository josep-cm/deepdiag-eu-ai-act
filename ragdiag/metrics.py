"""
Retrieval metrics implemented from scratch.

All functions take:
  retrieved: list[str]   — chunk_ids in ranked order (best first)
  gold:      set[str]    — ground-truth chunk_ids (usually a singleton)

Metrics:
  recall@k        — fraction of gold chunks found in top-k
  precision@k     — fraction of top-k that are gold
  hit_rate@k      — 1 if any gold chunk in top-k, else 0  (= recall@k for single gold)
  mrr             — reciprocal rank of first gold hit (0 if not found in retrieved)
  average_precision — area under precision-recall curve (for multi-gold; = 1/rank for single)
"""

from __future__ import annotations


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in gold)
    return hits / len(gold)


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in gold)
    return hits / k


def hit_rate_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    return 1.0 if any(cid in gold for cid in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: list[str], gold: set[str]) -> float:
    """Mean average precision for a single query."""
    if not gold:
        return 0.0
    hits = 0
    ap = 0.0
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            hits += 1
            ap += hits / rank
    return ap / len(gold)


def aggregate(results: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """
    Aggregate per-query metric dicts into mean values.

    Each result dict must have:
      retrieved: list[str]
      gold:      list[str]
    """
    n = len(results)
    if n == 0:
        return {}

    totals: dict[str, float] = {f"recall@{k}": 0.0 for k in ks}
    totals.update({f"precision@{k}": 0.0 for k in ks})
    totals.update({f"hit_rate@{k}": 0.0 for k in ks})
    totals["mrr"] = 0.0
    totals["map"] = 0.0

    for r in results:
        ret = r["retrieved"]
        gold = set(r["gold"])
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(ret, gold, k)
            totals[f"precision@{k}"] += precision_at_k(ret, gold, k)
            totals[f"hit_rate@{k}"] += hit_rate_at_k(ret, gold, k)
        totals["mrr"] += reciprocal_rank(ret, gold)
        totals["map"] += average_precision(ret, gold)

    return {k: v / n for k, v in totals.items()}


def failure_decomposition(results: list[dict], k: int = 5) -> dict:
    """
    Classify each eval item as retrieval-bound or generation-bound.

    retrieval_bound: gold chunk NOT in top-k  → retrieval is the bottleneck
    generation_bound: gold chunk IS in top-k  → generation is the bottleneck
      (generation_bound only meaningful after we have answer correctness scores)

    Returns aggregate counts and rates.
    """
    retrieval_bound = 0
    gold_in_topk = 0

    for r in results:
        gold = set(r["gold"])
        if any(cid in gold for cid in r["retrieved"][:k]):
            gold_in_topk += 1
        else:
            retrieval_bound += 1

    n = len(results)
    return {
        "k": k,
        "total": n,
        "gold_in_top_k": gold_in_topk,
        "retrieval_bound": retrieval_bound,
        "gold_in_top_k_rate": gold_in_topk / n if n else 0.0,
        "retrieval_bound_rate": retrieval_bound / n if n else 0.0,
    }
