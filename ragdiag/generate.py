"""
Generation pipeline: retrieve top-k chunks → generate answer via Ollama.

Produces: results/generation_baseline.json
Each record carries:
  question, gold_answer, generated_answer, retrieved_chunks, gold_in_topk
"""

import json
import re
import time
from pathlib import Path

import requests

from .index import load, search

EVAL_PATH = Path("data/eval/eval_set.json")
CHUNKS_PATH = Path("data/chunks/ai_act_en.json")
RESULTS_DIR = Path("results")
PROMPTS_DIR = Path("prompts")

OLLAMA_URL = "http://localhost:11434/api/generate"
GENERATOR_MODEL = "qwen2.5:3b"
TOP_K = 5
CONTEXT_CHAR_LIMIT = 600   # per chunk, to stay within context window


def _ollama(prompt: str, model: str = GENERATOR_MODEL, max_tokens: int = 300) -> str | None:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.RequestException:
            if attempt == 3:
                return None
            time.sleep(2)
    return None


def run(k: int = TOP_K, tag: str = "baseline", resume: bool = True) -> list[dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"generation_{tag}.json"

    pairs = json.loads(EVAL_PATH.read_text())
    verified = [p for p in pairs if p.get("verified") == "pass"]
    eval_pairs = verified if verified else pairs

    chunks_by_id = {c["chunk_id"]: c for c in json.loads(CHUNKS_PATH.read_text())}

    # Resume support
    done_ids: set[str] = set()
    existing: list[dict] = []
    if resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        done_ids = {r["id"] for r in existing}
        print(f"[generate] resuming — {len(existing)} done")

    print(f"[generate] loading index …")
    index, chunk_ids, model = load()

    gen_prompt_tpl = (PROMPTS_DIR / "generate_answer.txt").read_text()
    results = list(existing)

    todo = [p for p in eval_pairs if p["id"] not in done_ids]
    print(f"[generate] {len(todo)} pairs to process  (model={GENERATOR_MODEL}, k={k})")

    for i, pair in enumerate(todo, 1):
        hits = search(pair["question"], index, chunk_ids, model, k=k)
        retrieved_ids = [cid for cid, _ in hits]
        gold_set = set(pair["gold_chunk_ids"])
        gold_in_topk = any(cid in gold_set for cid in retrieved_ids)

        # Build context string
        context_parts = []
        for cid in retrieved_ids:
            c = chunks_by_id.get(cid, {})
            label = f"[{c.get('type','').upper()} {c.get('number','')}]"
            if c.get("title"):
                label += f" {c['title']}"
            text = c.get("text", "")[:CONTEXT_CHAR_LIMIT]
            context_parts.append(f"{label}\n{text}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = gen_prompt_tpl.format(context=context, question=pair["question"])
        generated = _ollama(prompt)

        record = {
            "id": pair["id"],
            "chunk_id": pair["chunk_id"],
            "chunk_type": pair["chunk_type"],
            "question": pair["question"],
            "gold_answer": pair["answer"],
            "generated_answer": generated or "",
            "retrieved_chunk_ids": retrieved_ids,
            "gold_chunk_ids": pair["gold_chunk_ids"],
            "gold_in_topk": gold_in_topk,
            "context": context,
        }
        results.append(record)

        if generated:
            print(f"  [{i:3d}/{len(todo)}] {pair['id']}  gold_in_top{k}={gold_in_topk}  gen={generated[:60]!r}")
        else:
            print(f"  [{i:3d}/{len(todo)}] {pair['id']}  GENERATION FAILED")

        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print(f"[generate] done → {out_path}")
    return results
