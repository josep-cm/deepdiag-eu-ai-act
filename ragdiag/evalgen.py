"""
Generate ~100 Q/A pairs from the chunk corpus via a local Ollama model.

Stratified sampling:
  - articles:  60 samples  (richest source of specific obligations)
  - recitals:  30 samples
  - annexes:   10 samples

Each pair carries the gold chunk_id so retrieval evaluation knows ground truth.
Output: data/eval/eval_set.json
"""

import json
import random
import re
import time
from pathlib import Path
from typing import Optional

import requests

CHUNKS_PATH = Path("data/chunks/ai_act_en.json")
EVAL_PATH = Path("data/eval/eval_set.json")
PROMPT_PATH = Path("prompts/qa_generation.txt")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

STRATA = {"article": 60, "recital": 30, "annex": 10}
MIN_CHUNK_CHARS = 300
# Article 3 is the definitions article — too long and maps to many chunks
SKIP_CHUNK_IDS = {"article-3"}
RANDOM_SEED = 42


def _call_ollama(system: str, user: str, retries: int = 3) -> Optional[str]:
    payload = {
        "model": MODEL,
        "prompt": f"{system}\n\n---\n\nCHUNK TEXT:\n{user}",
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 300},
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.RequestException as exc:
            if attempt == retries:
                print(f"    [ollama] failed after {retries} attempts: {exc}")
                return None
            time.sleep(2)
    return None


def _parse_json_response(raw: str) -> Optional[dict]:
    """Extract the first JSON object from the model response."""
    raw = raw.strip()
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to extract JSON block
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate chunk text to stay within model context budget."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def generate(resume: bool = True) -> list[dict]:
    chunks = json.loads(CHUNKS_PATH.read_text())
    system_prompt = PROMPT_PATH.read_text().strip()

    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing pairs if resuming
    existing: list[dict] = []
    done_ids: set[str] = set()
    if resume and EVAL_PATH.exists():
        existing = json.loads(EVAL_PATH.read_text())
        done_ids = {p["chunk_id"] for p in existing}
        print(f"[evalgen] resuming — {len(existing)} pairs already done")

    # Stratified sample
    by_type: dict[str, list[dict]] = {}
    for c in chunks:
        if len(c["text"]) < MIN_CHUNK_CHARS:
            continue
        if c["chunk_id"] in SKIP_CHUNK_IDS:
            continue
        by_type.setdefault(c["type"], []).append(c)

    rng = random.Random(RANDOM_SEED)
    selected: list[dict] = []
    for chunk_type, n in STRATA.items():
        pool = by_type.get(chunk_type, [])
        # Exclude already-done chunks
        pool = [c for c in pool if c["chunk_id"] not in done_ids]
        k = min(n, len(pool))
        selected.extend(rng.sample(pool, k))

    print(f"[evalgen] {len(selected)} chunks to process  (model={MODEL})")

    results = list(existing)
    failed = 0

    for i, chunk in enumerate(selected, 1):
        cid = chunk["chunk_id"]
        truncated = _truncate(chunk["text"])
        context = f"[{chunk['type'].upper()} {chunk['number']}]"
        if chunk.get("title"):
            context += f" — {chunk['title']}"
        context += f"\n\n{truncated}"

        print(f"  [{i:3d}/{len(selected)}] {cid} … ", end="", flush=True)
        raw = _call_ollama(system_prompt, context)

        if raw is None:
            print("SKIP (no response)")
            failed += 1
            continue

        parsed = _parse_json_response(raw)
        if not parsed or "question" not in parsed or "answer" not in parsed:
            print(f"SKIP (bad JSON): {raw[:80]!r}")
            failed += 1
            continue

        q = parsed["question"].strip()
        a = parsed["answer"].strip()

        if not q or not a or len(q) < 15:
            print("SKIP (empty/short)")
            failed += 1
            continue

        pair = {
            "id": f"qa-{len(results) + 1:04d}",
            "chunk_id": cid,
            "chunk_type": chunk["type"],
            "chunk_number": chunk["number"],
            "chunk_title": chunk.get("title"),
            "question": q,
            "answer": a,
            "gold_chunk_ids": [cid],
            "verified": None,  # filled in during hand-check
        }
        results.append(pair)
        print(f"OK  Q={q[:60]!r}")

        # Save incrementally so we can resume after interruptions
        EVAL_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print(
        f"\n[evalgen] done — {len(results)} pairs saved, {failed} failed → {EVAL_PATH}"
    )
    return results
