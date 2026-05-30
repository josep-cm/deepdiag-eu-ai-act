"""
Generation metrics — implemented from scratch, then cross-checked against RAGAS.

FROM-SCRATCH IMPLEMENTATIONS
─────────────────────────────
faithfulness:
  Decompose the generated answer into atomic claims via LLM.
  For each claim ask the LLM: "Is this claim supported by the context? yes/no"
  Score = supported_claims / total_claims
  (Mirrors the RAGAS faithfulness algorithm.)

context_recall:
  For each sentence in the gold answer ask the LLM:
  "Can this sentence be attributed to the context? yes/no"
  Score = attributed_sentences / total_sentences
  (Mirrors RAGAS context_recall.)

answer_relevance (simple):
  Ask LLM to rate 0-1 whether the generated answer addresses the question.
  Stored for reference; RAGAS uses a different embedding-based method.

RAGAS CROSS-CHECK
──────────────────
We run RAGAS faithfulness + context_recall on the same data using Ollama
as the backing LLM (via langchain), then compute Pearson r and mean absolute
error between our scores and RAGAS scores.

All judge calls use temperature=0 and log raw outputs for reproducibility.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests

RESULTS_DIR = Path("results")
PROMPTS_DIR = Path("prompts")
OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = os.environ.get("RAGDIAG_JUDGE_MODEL", "qwen2.5:3b")
# Set RAGDIAG_JUDGE_MODEL=claude-3-5-haiku-20241022 to use an API judge
# Set ANTHROPIC_API_KEY env var for API judge access


def _ollama_judge(prompt: str) -> str:
    """Call local Ollama at temperature=0 for reproducible judge scoring."""
    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 400},
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.RequestException:
            if attempt == 3:
                return ""
            time.sleep(2)
    return ""


def _api_judge(prompt: str) -> str:
    """Call Anthropic API judge — used when RAGDIAG_JUDGE_MODEL is an API model."""
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return ""
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _judge(prompt: str) -> str:
    api_models = {"claude", "gpt", "gemini"}
    if any(m in JUDGE_MODEL for m in api_models):
        return _api_judge(prompt)
    return _ollama_judge(prompt)


def _parse_yes_no(raw: str) -> Optional[bool]:
    raw = raw.lower().strip()
    if raw.startswith("yes"):
        return True
    if raw.startswith("no"):
        return False
    if "yes" in raw[:20]:
        return True
    if "no" in raw[:20]:
        return False
    return None


def _parse_json_score(raw: str) -> Optional[float]:
    try:
        m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        if m:
            return min(1.0, max(0.0, float(m.group(1))))
    except Exception:
        pass
    return None


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ── FROM-SCRATCH: FAITHFULNESS ────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """Break the following answer into a list of atomic factual claims.
Each claim must be a single, self-contained statement.
Return ONLY a JSON array of strings, e.g. ["claim1", "claim2"].

Answer: {answer}"""

_CLAIM_CHECK_PROMPT = """Context:
{context}

Claim: {claim}

Is this claim directly supported by the context above?
Respond with ONLY "yes" or "no"."""


def faithfulness_scratch(answer: str, context: str, log: list) -> float:
    if not answer.strip():
        return 0.0

    # Step 1: decompose into claims
    raw_claims = _judge(_DECOMPOSE_PROMPT.format(answer=answer))
    log.append({"step": "decompose", "raw": raw_claims})

    try:
        m = re.search(r"\[.*?\]", raw_claims, re.DOTALL)
        claims = json.loads(m.group(0)) if m else []
    except Exception:
        claims = []

    if not claims:
        # Fallback: treat sentences as claims
        claims = _split_sentences(answer)

    if not claims:
        return 0.0

    # Step 2: check each claim against context
    supported = 0
    claim_log = []
    for claim in claims:
        raw = _judge(_CLAIM_CHECK_PROMPT.format(context=context[:1500], claim=claim))
        verdict = _parse_yes_no(raw)
        if verdict is True:
            supported += 1
        claim_log.append({"claim": claim, "raw": raw, "verdict": verdict})

    log.append({"step": "claim_checks", "claims": claim_log})
    return supported / len(claims)


# ── FROM-SCRATCH: CONTEXT RECALL ──────────────────────────────────────────────

_SENTENCE_ATTR_PROMPT = """Context:
{context}

Statement: {sentence}

Can this statement be directly attributed to the context above?
Respond with ONLY "yes" or "no"."""


def context_recall_scratch(gold_answer: str, context: str, log: list) -> float:
    sentences = _split_sentences(gold_answer)
    if not sentences:
        return 0.0

    attributed = 0
    sentence_log = []
    for sent in sentences:
        raw = _judge(_SENTENCE_ATTR_PROMPT.format(context=context[:1500], sentence=sent))
        verdict = _parse_yes_no(raw)
        if verdict is True:
            attributed += 1
        sentence_log.append({"sentence": sent, "raw": raw, "verdict": verdict})

    log.append({"step": "context_recall", "sentences": sentence_log})
    return attributed / len(sentences)


# ── FROM-SCRATCH: ANSWER RELEVANCE ────────────────────────────────────────────

_RELEVANCE_PROMPT = """Question: {question}

Answer: {answer}

Does this answer directly address the question?
Score 0.0 (completely off-topic) to 1.0 (fully answers the question).
Respond with ONLY valid JSON: {{"score": <float>}}"""


def answer_relevance_scratch(question: str, answer: str, log: list) -> float:
    raw = _judge(_RELEVANCE_PROMPT.format(question=question, answer=answer))
    log.append({"step": "answer_relevance", "raw": raw})
    score = _parse_json_score(raw)
    return score if score is not None else 0.5


# ── RAGAS CROSS-CHECK ─────────────────────────────────────────────────────────

def _run_ragas(records: list[dict]) -> list[dict]:
    """Run RAGAS faithfulness + context_recall via Ollama LangChain backend."""
    try:
        from datasets import Dataset
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
        from ragas import evaluate
        from ragas.metrics import context_recall as ragas_cr
        from ragas.metrics import faithfulness as ragas_faith
    except ImportError as e:
        print(f"[ragas] import error: {e} — skipping RAGAS cross-check")
        return []

    print(f"[ragas] running on {len(records)} records …")

    llm = ChatOllama(model=JUDGE_MODEL, temperature=0)
    embeddings = OllamaEmbeddings(model=JUDGE_MODEL)

    ragas_faith.llm = llm
    ragas_cr.llm = llm
    ragas_faith.embeddings = embeddings
    ragas_cr.embeddings = embeddings

    data = {
        "question":  [r["question"]          for r in records],
        "answer":    [r["generated_answer"]   for r in records],
        "contexts":  [r["retrieved_chunk_ids"][:5] for r in records],  # placeholder
        "ground_truth": [r["gold_answer"]     for r in records],
    }
    # RAGAS needs actual context text, not IDs
    chunks_by_id = {}
    cp = Path("data/chunks/ai_act_en.json")
    if cp.exists():
        for c in json.loads(cp.read_text()):
            chunks_by_id[c["chunk_id"]] = c["text"][:800]

    data["contexts"] = [
        [chunks_by_id.get(cid, "") for cid in r["retrieved_chunk_ids"][:5]]
        for r in records
    ]

    ds = Dataset.from_dict(data)
    try:
        result = evaluate(ds, metrics=[ragas_faith, ragas_cr])
        df = result.to_pandas()
        out = []
        for _, row in df.iterrows():
            out.append({
                "ragas_faithfulness": float(row.get("faithfulness", 0)),
                "ragas_context_recall": float(row.get("context_recall", 0)),
            })
        return out
    except Exception as e:
        print(f"[ragas] evaluation error: {e}")
        return []


# ── MAIN SCORER ───────────────────────────────────────────────────────────────

def score_all(gen_path: Path, tag: str = "baseline", ragas_sample: int = 20) -> dict:
    records = json.loads(gen_path.read_text())
    print(f"[gen_metrics] scoring {len(records)} records  judge={JUDGE_MODEL}")

    scored = []
    for i, r in enumerate(records, 1):
        log = []
        faith = faithfulness_scratch(r["generated_answer"], r["context"], log)
        cr    = context_recall_scratch(r["gold_answer"],     r["context"], log)
        ar    = answer_relevance_scratch(r["question"],  r["generated_answer"], log)

        scored.append({
            **r,
            "faithfulness":     round(faith, 3),
            "context_recall":   round(cr,    3),
            "answer_relevance": round(ar,    3),
            "judge_log":        log,
        })

        if i % 10 == 0 or i == len(records):
            avg_f  = sum(x["faithfulness"]   for x in scored) / len(scored)
            avg_cr = sum(x["context_recall"] for x in scored) / len(scored)
            print(f"  [{i:3d}/{len(records)}]  faithfulness={avg_f:.3f}  context_recall={avg_cr:.3f}")

    # ── RAGAS cross-check on a sample ─────────────────────────────────────────
    sample = scored[:ragas_sample]
    ragas_scores = _run_ragas(sample)

    agreement = {}
    if ragas_scores and len(ragas_scores) == len(sample):
        import math
        ours_f  = [s["faithfulness"]   for s in sample]
        ragas_f = [s["ragas_faithfulness"]  for s in ragas_scores]
        ours_cr  = [s["context_recall"]  for s in sample]
        ragas_cr = [s["ragas_context_recall"] for s in ragas_scores]

        def pearson(a, b):
            n = len(a)
            ma, mb = sum(a)/n, sum(b)/n
            num = sum((x-ma)*(y-mb) for x,y in zip(a,b))
            den = math.sqrt(sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b))
            return num/den if den else 0.0

        def mae(a, b):
            return sum(abs(x-y) for x,y in zip(a,b)) / len(a)

        agreement = {
            "n_sample": len(sample),
            "faithfulness_pearson_r":   round(pearson(ours_f, ragas_f),  3),
            "faithfulness_mae":         round(mae(ours_f, ragas_f),       3),
            "context_recall_pearson_r": round(pearson(ours_cr, ragas_cr), 3),
            "context_recall_mae":       round(mae(ours_cr, ragas_cr),     3),
        }
        # Attach RAGAS scores back to sample records
        for s, rs in zip(sample, ragas_scores):
            s.update(rs)
        print(f"\n[ragas] agreement: faithfulness r={agreement['faithfulness_pearson_r']}  "
              f"context_recall r={agreement['context_recall_pearson_r']}")
    else:
        print("[ragas] skipped or failed — no agreement computed")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(scored)
    agg = {
        "faithfulness_mean":     round(sum(s["faithfulness"]     for s in scored) / n, 3),
        "context_recall_mean":   round(sum(s["context_recall"]   for s in scored) / n, 3),
        "answer_relevance_mean": round(sum(s["answer_relevance"] for s in scored) / n, 3),
    }

    # ── Full failure decomposition ─────────────────────────────────────────────
    retrieval_bound    = sum(1 for s in scored if not s["gold_in_topk"])
    generation_bound   = sum(1 for s in scored if s["gold_in_topk"] and s["faithfulness"] < 0.5)
    both_ok            = sum(1 for s in scored if s["gold_in_topk"] and s["faithfulness"] >= 0.5)

    decomp = {
        "total": n,
        "retrieval_bound":  retrieval_bound,
        "generation_bound": generation_bound,
        "both_ok":          both_ok,
        "retrieval_bound_rate":  round(retrieval_bound  / n, 3),
        "generation_bound_rate": round(generation_bound / n, 3),
        "both_ok_rate":          round(both_ok          / n, 3),
    }

    summary = {
        "tag": tag,
        "judge_model": JUDGE_MODEL,
        "n": n,
        "generation_metrics": agg,
        "failure_decomposition": decomp,
        "ragas_agreement": agreement,
    }

    out_path = RESULTS_DIR / f"gen_metrics_{tag}.json"
    out_path.write_text(json.dumps({"summary": summary, "per_item": scored}, indent=2, ensure_ascii=False))
    print(f"\n[gen_metrics] saved → {out_path}")
    return summary
