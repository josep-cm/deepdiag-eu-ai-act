"""
RAG Diagnostics — Interactive UI
Run with: .venv/bin/streamlit run app.py
"""

import json
import re
import time
from pathlib import Path

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
GENERATOR_MODEL = "qwen2.5:3b"
TOP_K = 5
CONTEXT_CHAR_LIMIT = 600

CHUNKS_PATH = Path("data/chunks/ai_act_en.json")
EVAL_PATH   = Path("data/eval/eval_set.json")
PROMPTS_DIR = Path("prompts")

TYPE_COLOR = {
    "article": "#2563eb",   # blue
    "recital": "#7c3aed",   # purple
    "annex":   "#059669",   # green
    "fixed":   "#d97706",   # amber
}
TYPE_EMOJI = {"article": "📋", "recital": "📖", "annex": "📎"}

# ── Resource loading (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading index and models…")
def load_resources():
    from ragdiag.index import load
    index, chunk_ids, model = load()
    chunks_by_id = {c["chunk_id"]: c
                    for c in json.loads(CHUNKS_PATH.read_text())}
    eval_by_question = {}
    if EVAL_PATH.exists():
        for p in json.loads(EVAL_PATH.read_text()):
            eval_by_question[p["question"].strip().lower()] = p
    return index, chunk_ids, model, chunks_by_id, eval_by_question


def search(query, index, chunk_ids, model, k=TOP_K):
    from ragdiag.index import search as _search
    return _search(query, index, chunk_ids, model, k=k)


def generate_answer(question, context):
    tpl = (PROMPTS_DIR / "generate_answer.txt").read_text()
    prompt = tpl.format(context=context, question=question)
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": GENERATOR_MODEL, "prompt": prompt,
                  "stream": False, "options": {"temperature": 0.0, "num_predict": 300}},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"⚠️ Generation failed: {e}"


def score_faithfulness(answer, context):
    """Quick single-call faithfulness estimate (0-1)."""
    prompt = f"""Context:
{context[:1200]}

Answer: {answer}

Is every claim in the answer directly supported by the context?
Reply with a score from 0.0 (nothing supported) to 1.0 (fully supported) as JSON: {{"score": <float>}}"""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": GENERATOR_MODEL, "prompt": prompt,
                  "stream": False, "options": {"temperature": 0.0, "num_predict": 60}},
            timeout=30,
        )
        raw = r.json().get("response", "")
        m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        return float(m.group(1)) if m else None
    except Exception:
        return None


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepDiag: RAG on the EU AI Act",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ DeepDiag: RAG on the EU AI Act")
st.caption("Ask anything about the EU AI Act (Regulation EU 2024/1689)")

# Load resources
with st.spinner("Loading index…"):
    index, chunk_ids, model, chunks_by_id, eval_by_question = load_resources()

st.success(f"Ready — {len(chunks_by_id)} chunks indexed", icon="✅")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks retrieved (k)", 1, 10, TOP_K)
    show_faithfulness = st.toggle("Score faithfulness", value=True)
    show_chunk_text   = st.toggle("Show full chunk text", value=False)

    st.divider()
    st.header("📚 Example questions")
    examples = [
        "What AI practices are prohibited under the EU AI Act?",
        "What is the maximum fine for providers of general-purpose AI models?",
        "What threshold determines a model has significant generality?",
        "What must providers indicate on their high-risk AI systems?",
        "What are the compliance deadlines for high-risk AI systems?",
        "Who counts as a deployer under this regulation?",
        "What high-risk AI systems are listed in Annex III?",
        "What does AI literacy mean?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question_input"] = ex

    st.divider()
    st.caption("Model: qwen2.5:3b (local)")
    st.caption("Embeddings: BGE-small-en-v1.5")

# ── Main input ────────────────────────────────────────────────────────────────

question = st.text_input(
    "Your question",
    placeholder="e.g. What AI practices are prohibited?",
    key="question_input",
)

run = st.button("Ask", type="primary", disabled=not question)

if run and question:
    q = question.strip()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    hits = search(q, index, chunk_ids, model, k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    retrieved_ids = [cid for cid, _ in hits]
    retrieved_scores = [s for _, s in hits]

    # Build context
    context_parts = []
    for cid, score in hits:
        c = chunks_by_id.get(cid, {})
        label = f"[{c.get('type','').upper()} {c.get('number','')}]"
        if c.get("title"):
            label += f" {c['title']}"
        context_parts.append(f"{label}\n{c.get('text','')[:CONTEXT_CHAR_LIMIT]}")
    context = "\n\n---\n\n".join(context_parts)

    # ── Generation ────────────────────────────────────────────────────────────
    with st.spinner("Generating answer…"):
        t1 = time.perf_counter()
        answer = generate_answer(q, context)
        gen_ms = (time.perf_counter() - t1) * 1000

    # ── Faithfulness ──────────────────────────────────────────────────────────
    faith_score = None
    if show_faithfulness and answer and "does not answer" not in answer.lower():
        with st.spinner("Scoring faithfulness…"):
            faith_score = score_faithfulness(answer, context)

    # ── Check if this is an eval question ─────────────────────────────────────
    eval_match = eval_by_question.get(q.lower())
    gold_in_topk = False
    if eval_match:
        gold_ids = set(eval_match.get("gold_chunk_ids", []))
        gold_in_topk = any(cid in gold_ids for cid in retrieved_ids)

    # ── Layout ────────────────────────────────────────────────────────────────
    st.divider()

    col_ans, col_ret = st.columns([3, 2], gap="large")

    with col_ans:
        st.subheader("Answer")

        # Answer box
        if "does not answer" in answer.lower() or "not contain" in answer.lower():
            st.warning(answer, icon="⚠️")
        else:
            st.info(answer, icon="💬")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieval", f"{retrieval_ms:.0f} ms")
        m2.metric("Generation", f"{gen_ms:.0f} ms")
        if faith_score is not None:
            color = "normal" if faith_score >= 0.7 else ("off" if faith_score < 0.4 else "normal")
            m3.metric("Faithfulness", f"{faith_score:.2f}", delta=None)
        if eval_match:
            m4.metric(
                "Gold in top-k",
                "✅ Yes" if gold_in_topk else "❌ No",
                delta=None,
            )

        # If eval match: show gold answer for comparison
        if eval_match:
            with st.expander("📊 Eval set comparison"):
                st.caption("This question is in the evaluation set.")
                st.markdown(f"**Gold answer:** {eval_match['answer']}")
                st.markdown(f"**Gold chunk:** `{eval_match['gold_chunk_ids'][0]}`")
                if gold_in_topk:
                    rank = next(
                        (i+1 for i, cid in enumerate(retrieved_ids)
                         if cid in set(eval_match["gold_chunk_ids"])),
                        None
                    )
                    st.success(f"Gold chunk found at rank {rank} ✓")
                else:
                    st.error("Gold chunk NOT in top-k — retrieval-bound failure")

    with col_ret:
        st.subheader(f"Retrieved chunks (k={top_k})")

        for i, (cid, score) in enumerate(hits, 1):
            c = chunks_by_id.get(cid, {})
            ctype = c.get("type", "unknown")
            color = TYPE_COLOR.get(ctype, "#6b7280")
            emoji = TYPE_EMOJI.get(ctype, "📄")

            # Check if this is the gold chunk
            is_gold = eval_match and cid in set(eval_match.get("gold_chunk_ids", []))
            gold_tag = " 🏆 gold" if is_gold else ""

            with st.expander(
                f"**#{i}** {emoji} `{cid}`{gold_tag}  — score {score:.3f}",
                expanded=(i == 1),
            ):
                if c.get("title"):
                    st.markdown(f"**{c['title']}**")

                # Type badge
                st.markdown(
                    f'<span style="background:{color};color:white;'
                    f'padding:2px 8px;border-radius:4px;font-size:0.75rem">'
                    f'{ctype.upper()}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(f"chunk_id: `{cid}`")

                if c.get("cross_references"):
                    st.caption("Cross-refs: " + ", ".join(c["cross_references"][:4]))

                text = c.get("text", "")
                if show_chunk_text:
                    st.text_area("Full text", text, height=150, key=f"chunk_{i}_{cid}",
                                 label_visibility="collapsed")
                else:
                    st.caption(text[:300] + ("…" if len(text) > 300 else ""))

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Source: EU AI Act, Regulation (EU) 2024/1689 — "
    "[EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689)"
)
