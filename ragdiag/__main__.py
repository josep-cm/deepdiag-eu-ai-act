"""Entry point: python -m ragdiag <command> [options]"""

import argparse
import json
import sys
from pathlib import Path

from .download import download
from .parse import parse

CHUNKS_PATH = Path("data/chunks/ai_act_en.json")
EVAL_PATH = Path("data/eval/eval_set.json")


def cmd_ingest(args):
    html_path = download(lang=args.lang, force=args.force)
    chunks = parse(html_path, lang=args.lang.upper())
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    print(f"[ingest] wrote {len(chunks)} chunks → {CHUNKS_PATH}")


def cmd_show(args):
    if not CHUNKS_PATH.exists():
        sys.exit("Run 'python -m ragdiag ingest' first.")
    chunks = json.loads(CHUNKS_PATH.read_text())
    by_type = {}
    for c in chunks:
        by_type.setdefault(c["type"], []).append(c)
    for chunk_type, n in [("recital", 5), ("article", 5), ("annex", 5)]:
        lst = by_type.get(chunk_type, [])
        print(f"\n{'='*70}")
        print(f"  {chunk_type.upper()}S — showing {min(n, len(lst))} of {len(lst)}")
        print(f"{'='*70}")
        for c in lst[:n]:
            print(f"\n[{c['chunk_id']}]  title={c['title']!r}")
            print(f"  cross_refs: {c['cross_references'][:5]}")
            print(f"  text ({len(c['text'])} chars): {c['text'][:250]!r}")


def cmd_evalgen(args):
    if not CHUNKS_PATH.exists():
        sys.exit("Run 'python -m ragdiag ingest' first.")
    from .evalgen import generate
    generate(resume=not args.fresh)


def cmd_verify(args):
    """Interactive hand-check tool for eval set quality."""
    if not EVAL_PATH.exists():
        sys.exit("Run 'python -m ragdiag evalgen' first.")

    pairs = json.loads(EVAL_PATH.read_text())
    chunks = {}
    if CHUNKS_PATH.exists():
        for c in json.loads(CHUNKS_PATH.read_text()):
            chunks[c["chunk_id"]] = c

    # Filter to unverified or show a stratified sample
    unverified = [p for p in pairs if p.get("verified") is None]
    if args.sample:
        import random
        random.seed(0)
        # Stratified: equal share from each type
        by_type: dict[str, list] = {}
        for p in unverified:
            by_type.setdefault(p["chunk_type"], []).append(p)
        pool = []
        per_type = max(1, args.sample // len(by_type))
        for lst in by_type.values():
            pool.extend(random.sample(lst, min(per_type, len(lst))))
        unverified = pool[:args.sample]

    print(f"\n=== VERIFY MODE: {len(unverified)} pairs to check ===")
    print("Commands: [y]es (answerable+correct)  [n]o (reject)  [s]kip  [q]uit\n")

    passed = rejected = skipped = 0

    for p in unverified:
        chunk = chunks.get(p["chunk_id"], {})
        print(f"\n{'─'*70}")
        print(f"ID: {p['id']}  chunk: {p['chunk_id']}")
        if chunk:
            print(f"CHUNK TEXT (first 400 chars):\n{chunk.get('text','')[:400]}")
        print(f"\nQUESTION: {p['question']}")
        print(f"ANSWER:   {p['answer']}")
        print()

        while True:
            cmd = input("  verdict [y/n/s/q]: ").strip().lower()
            if cmd in ("y", "n", "s", "q"):
                break

        if cmd == "q":
            break
        elif cmd == "y":
            p["verified"] = "pass"
            passed += 1
        elif cmd == "n":
            p["verified"] = "reject"
            rejected += 1
        else:
            skipped += 1

    # Save back
    EVAL_PATH.write_text(json.dumps(pairs, ensure_ascii=False, indent=2))

    total_checked = passed + rejected
    pass_rate = passed / total_checked if total_checked else 0
    print(f"\n=== VERIFICATION SUMMARY ===")
    print(f"  Checked:  {total_checked}")
    print(f"  Passed:   {passed}  ({100*pass_rate:.0f}%)")
    print(f"  Rejected: {rejected}")
    print(f"  Skipped:  {skipped}")
    print(f"  Pass rate: {100*pass_rate:.1f}%")


def cmd_generate(args):
    if not Path("data/eval/eval_set.json").exists():
        sys.exit("Run evalgen first.")
    from .generate import run
    run(k=args.k, tag=args.tag, resume=not args.fresh)


def cmd_gen_metrics(args):
    gen_path = Path(f"results/generation_{args.tag}.json")
    if not gen_path.exists():
        sys.exit(f"Run 'python -m ragdiag generate --tag {args.tag}' first.")
    from .gen_metrics import score_all
    summary = score_all(gen_path, tag=args.tag, ragas_sample=args.ragas_sample)
    m = summary["generation_metrics"]
    d = summary["failure_decomposition"]
    ra = summary.get("ragas_agreement", {})
    print(f"\n{'='*60}")
    print(f"  GENERATION METRICS  [{args.tag}]  judge={summary['judge_model']}")
    print(f"{'='*60}")
    print(f"  faithfulness:     {m['faithfulness_mean']:.3f}")
    print(f"  context_recall:   {m['context_recall_mean']:.3f}")
    print(f"  answer_relevance: {m['answer_relevance_mean']:.3f}")
    print(f"\n  FULL FAILURE DECOMPOSITION (n={d['total']})")
    print(f"  retrieval-bound:   {d['retrieval_bound']:3d}  ({100*d['retrieval_bound_rate']:.1f}%)")
    print(f"  generation-bound:  {d['generation_bound']:3d}  ({100*d['generation_bound_rate']:.1f}%)")
    print(f"  both OK:           {d['both_ok']:3d}  ({100*d['both_ok_rate']:.1f}%)")
    if ra:
        print(f"\n  RAGAS AGREEMENT (n={ra['n_sample']})")
        print(f"  faithfulness    pearson_r={ra['faithfulness_pearson_r']}  MAE={ra['faithfulness_mae']}")
        print(f"  context_recall  pearson_r={ra['context_recall_pearson_r']}  MAE={ra['context_recall_mae']}")


def cmd_ablation_chunking(args):
    from .ablation_chunking import run
    results, summary = run(tag=args.tag)
    m = summary["metrics"]
    d = summary["failure_decomposition_k5"]
    lat = summary["latency_ms"]
    print(f"\n{'='*60}")
    print(f"  FIXED-SIZE CHUNKING RESULTS  [{args.tag}]")
    print(f"{'='*60}")
    print(f"  corpus: {summary['corpus_size']} fixed chunks  (avg gold windows/query={summary['avg_gold_windows']})")
    print(f"  recall@1={m['recall@1']:.3f}  recall@3={m['recall@3']:.3f}  recall@5={m['recall@5']:.3f}  recall@10={m['recall@10']:.3f}")
    print(f"  hit_rate@5={m['hit_rate@5']:.3f}  mrr={m['mrr']:.3f}  map={m['map']:.3f}")
    print(f"  latency  p50={lat['p50']:.0f}ms  p95={lat['p95']:.0f}ms")
    print(f"\n  FAILURE DECOMPOSITION (k=5)")
    print(f"  gold in top-5:   {d['gold_in_top_k']:3d} / {d['total']}  ({100*d['gold_in_top_k_rate']:.1f}%)")
    print(f"  retrieval-bound: {d['retrieval_bound']:3d} / {d['total']}  ({100*d['retrieval_bound_rate']:.1f}%)")


def cmd_rerank(args):
    if not Path("data/eval/eval_set.json").exists():
        sys.exit("Run 'python -m ragdiag evalgen' first.")
    from .rerank import run_reranked
    results, summary = run_reranked(tag=args.tag)
    m = summary["metrics"]
    d = summary["failure_decomposition_k5"]
    lat = summary["latency_ms"]
    print(f"\n{'='*60}")
    print(f"  RERANKER RESULTS  [{args.tag}]")
    print(f"{'='*60}")
    print(f"  recall@1={m['recall@1']:.3f}  recall@3={m['recall@3']:.3f}  recall@5={m['recall@5']:.3f}  recall@10={m['recall@10']:.3f}")
    print(f"  hit_rate@5={m['hit_rate@5']:.3f}  mrr={m['mrr']:.3f}  map={m['map']:.3f}")
    print(f"  latency  p50={lat['p50_total']:.0f}ms  p95={lat['p95_total']:.0f}ms")
    print(f"           (bi={lat['p50_biencoder']:.0f}ms + ce={lat['p50_crossencoder']:.0f}ms)")
    print(f"\n  FAILURE DECOMPOSITION (k=5)")
    print(f"  gold in top-5:   {d['gold_in_top_k']:3d} / {d['total']}  ({100*d['gold_in_top_k_rate']:.1f}%)")
    print(f"  retrieval-bound: {d['retrieval_bound']:3d} / {d['total']}  ({100*d['retrieval_bound_rate']:.1f}%)")


def cmd_compare(args):
    """Print side-by-side comparison table of two result files."""
    import glob
    files = sorted(glob.glob("results/retrieval_*.json"))
    if not files:
        sys.exit("No result files found in results/.")

    runs = {}
    for f in files:
        data = json.loads(Path(f).read_text())
        s = data["summary"]
        runs[s["tag"]] = s

    # Header
    tags = list(runs.keys())
    print(f"\n{'Metric':<22}" + "".join(f"{t:>16}" for t in tags))
    print("─" * (22 + 16 * len(tags)))

    metrics_to_show = [
        ("recall@1", "recall@1"),
        ("recall@3", "recall@3"),
        ("recall@5", "recall@5"),
        ("recall@10", "recall@10"),
        ("hit_rate@5", "hit_rate@5"),
        ("mrr", "mrr"),
        ("map", "map"),
    ]
    for label, key in metrics_to_show:
        row = f"{label:<22}"
        for t in tags:
            v = runs[t]["metrics"].get(key, 0)
            row += f"{v:>16.3f}"
        print(row)

    print()
    # Failure decomp
    print(f"{'retrieval-bound%':<22}" + "".join(
        f"{100*runs[t]['failure_decomposition_k5']['retrieval_bound_rate']:>15.1f}%" for t in tags
    ))
    print(f"{'gold-in-top5%':<22}" + "".join(
        f"{100*runs[t]['failure_decomposition_k5']['gold_in_top_k_rate']:>15.1f}%" for t in tags
    ))

    print()
    # Latency — use p50/p95 regardless of key name
    def _p50(s):
        lat = s.get("latency_ms", {})
        return lat.get("p50_total") or lat.get("p50") or 0

    def _p95(s):
        lat = s.get("latency_ms", {})
        return lat.get("p95_total") or lat.get("p95") or 0

    print(f"{'latency p50 (ms)':<22}" + "".join(f"{_p50(runs[t]):>16.0f}" for t in tags))
    print(f"{'latency p95 (ms)':<22}" + "".join(f"{_p95(runs[t]):>16.0f}" for t in tags))


def cmd_index(args):
    from .index import build
    build(model_name=args.model, force=args.force)


def cmd_retrieve(args):
    if not Path("data/eval/eval_set.json").exists():
        sys.exit("Run 'python -m ragdiag evalgen' first.")
    from .retrieve import run
    results, summary = run(k=args.k, tag=args.tag)
    m = summary["metrics"]
    d = summary["failure_decomposition_k5"]
    lat = summary["latency_ms"]
    print(f"\n{'='*60}")
    print(f"  RETRIEVAL RESULTS  [{args.tag}]")
    print(f"{'='*60}")
    print(f"  recall@1={m['recall@1']:.3f}  recall@3={m['recall@3']:.3f}  recall@5={m['recall@5']:.3f}  recall@10={m['recall@10']:.3f}")
    print(f"  hit_rate@5={m['hit_rate@5']:.3f}  mrr={m['mrr']:.3f}  map={m['map']:.3f}")
    print(f"  latency  p50={lat['p50']:.0f}ms  p95={lat['p95']:.0f}ms")
    print(f"\n  FAILURE DECOMPOSITION (k=5)")
    print(f"  gold in top-5:      {d['gold_in_top_k']:3d} / {d['total']}  ({100*d['gold_in_top_k_rate']:.1f}%)")
    print(f"  retrieval-bound:    {d['retrieval_bound']:3d} / {d['total']}  ({100*d['retrieval_bound_rate']:.1f}%)")


def cmd_evalstats(args):
    """Print eval set statistics."""
    if not EVAL_PATH.exists():
        sys.exit("Run 'python -m ragdiag evalgen' first.")
    pairs = json.loads(EVAL_PATH.read_text())
    by_type: dict[str, list] = {}
    for p in pairs:
        by_type.setdefault(p["chunk_type"], []).append(p)
    verified = [p for p in pairs if p.get("verified") == "pass"]
    rejected = [p for p in pairs if p.get("verified") == "reject"]
    print(f"\nEval set: {len(pairs)} total pairs")
    for t, lst in sorted(by_type.items()):
        print(f"  {t}: {len(lst)}")
    print(f"Verified pass: {len(verified)}")
    print(f"Verified reject: {len(rejected)}")
    print(f"Unverified: {len(pairs) - len(verified) - len(rejected)}")
    if verified or rejected:
        total = len(verified) + len(rejected)
        print(f"Pass rate: {100*len(verified)/total:.1f}%")


def main():
    parser = argparse.ArgumentParser(prog="ragdiag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Download + parse the AI Act")
    p_ingest.add_argument("--lang", default="EN")
    p_ingest.add_argument("--force", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_show = sub.add_parser("show", help="Show sample chunks")
    p_show.set_defaults(func=cmd_show)

    p_evalgen = sub.add_parser("evalgen", help="Generate Q/A eval pairs")
    p_evalgen.add_argument("--fresh", action="store_true", help="Ignore existing pairs")
    p_evalgen.set_defaults(func=cmd_evalgen)

    p_verify = sub.add_parser("verify", help="Hand-check eval pairs")
    p_verify.add_argument("--sample", type=int, default=None,
                          help="Number of pairs to verify (stratified sample)")
    p_verify.set_defaults(func=cmd_verify)

    p_gen = sub.add_parser("generate", help="Generate answers for eval set")
    p_gen.add_argument("--k", type=int, default=5)
    p_gen.add_argument("--tag", default="baseline")
    p_gen.add_argument("--fresh", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    p_gm = sub.add_parser("gen-metrics", help="Score generated answers")
    p_gm.add_argument("--tag", default="baseline")
    p_gm.add_argument("--ragas-sample", type=int, default=20)
    p_gm.set_defaults(func=cmd_gen_metrics)

    p_chunk = sub.add_parser("ablation-chunking", help="Fixed-size vs structural chunking ablation")
    p_chunk.add_argument("--tag", default="fixed512")
    p_chunk.set_defaults(func=cmd_ablation_chunking)

    p_rerank = sub.add_parser("rerank", help="Run retrieval + reranker ablation")
    p_rerank.add_argument("--tag", default="reranker")
    p_rerank.set_defaults(func=cmd_rerank)

    p_compare = sub.add_parser("compare", help="Side-by-side metric table for all runs")
    p_compare.set_defaults(func=cmd_compare)

    p_index = sub.add_parser("index", help="Build FAISS index")
    p_index.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p_index.add_argument("--force", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_retrieve = sub.add_parser("retrieve", help="Run retrieval eval")
    p_retrieve.add_argument("--k", type=int, default=10)
    p_retrieve.add_argument("--tag", default="baseline")
    p_retrieve.set_defaults(func=cmd_retrieve)

    p_stats = sub.add_parser("evalstats", help="Print eval set stats")
    p_stats.set_defaults(func=cmd_evalstats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
