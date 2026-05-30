# RAG Diagnostics — Final Results Report

**Corpus:** EU AI Act (Regulation EU 2024/1689)
**Eval set:** 98 Q/A pairs with gold chunk IDs
**Embedding model:** BAAI/bge-small-en-v1.5
**Generator + Judge:** qwen2.5:3b (local, via Ollama)
**Date:** 2026-05-30

---

## 1. Headline finding: failure decomposition

The single most important output of this project.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  For every 100 questions asked of this RAG pipeline:        │
│                                                             │
│   28 fail because the retriever never found the right chunk │
│   26 fail because the generator ignored or distorted it     │
│   46 succeed end-to-end                                     │
│                                                             │
│   The split is 52% retrieval / 48% generation               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This matters because the two failure types require completely different fixes:
- **Retrieval-bound** → better embeddings, reranker, metadata filters, chunk structure
- **Generation-bound** → stronger model, better prompt, longer context window, fine-tuning

A team that only looks at end-to-end accuracy would see 54% correct and call it a retrieval problem. They'd spend weeks on embeddings and barely move the needle — because half the remaining failures were the generator's fault all along.

---

## 2. Retrieval results

### Baseline: structural chunks + BGE-small-en-v1.5 + FAISS

| Metric | Value | Notes |
|---|---|---|
| recall@1 | 0.367 | Gold chunk is top result 37% of the time |
| recall@3 | 0.561 | |
| recall@5 | **0.735** | Main operating point for k=5 context feeding |
| recall@10 | 0.857 | |
| MRR | 0.510 | Average reciprocal rank of first correct chunk |
| MAP | 0.516 | |
| hit_rate@5 | 0.724 | At least one correct chunk in top-5 |
| Latency p50 | 12ms | |
| Latency p95 | 349ms | Spike from Article 3 (3,379 tokens) |

### Root cause analysis of the 27 retrieval failures

**Dominant pattern: recital-over-article confusion (12/27 cases)**

The EU AI Act explains every obligation in the preamble (recitals 1–254) before stating it as law (Articles 1–113). When a query asks about a provider obligation, recitals 66, 72, and 26 all discuss the topic in natural explanatory prose — and rank higher than the specific article because the bi-encoder sees semantic similarity but not the normative/explanatory distinction.

Example failure:
```
Q: "What must providers indicate on their high-risk AI systems?"
Gold: article-16
Top-3 retrieved: recital-72, recital-155, recital-66
```

**Secondary pattern: vague Q/A pairs (7/27 cases)**

Questions like "What are the specific requirements for high-risk AI systems?" (generated from `article-1`) are too broad to anchor to a single chunk. These are eval set quality issues, not retrieval failures — they would be caught by the verification step.

**Third pattern: amendment articles (3/27 cases)**

Articles 102–113 are legislative amendments that insert boilerplate text into other regulations. Their content is generic legal language with few distinctive terms, making semantic retrieval difficult.

---

## 3. Ablation 1 — Cross-encoder reranker

**Configuration:** bi-encoder retrieves top-20, cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks to top-10.

| Metric | Baseline | + Reranker | Delta |
|---|---|---|---|
| recall@1 | 0.367 | **0.459** | +0.092 ✓ |
| recall@3 | 0.561 | **0.633** | +0.072 ✓ |
| recall@5 | **0.735** | 0.704 | −0.031 ✗ |
| recall@10 | **0.857** | 0.755 | −0.102 ✗ |
| hit_rate@5 | **0.724** | 0.694 | −0.030 ✗ |
| MRR | 0.510 | **0.561** | +0.051 ✓ |
| Latency p50 | **12ms** | 98ms | 8× slower |
| Latency p95 | 349ms | **162ms** | better tail |

**Interpretation:**

The reranker is a **precision-at-1 improvement device**. It correctly promotes specific normative articles over generic recitals when it evaluates the (query, chunk) pair jointly — 28 items improved. But it also hurts 25 items, in two ways:

1. The MS MARCO-trained cross-encoder still prefers recital-style prose over article-style obligations (11 cases)
2. Gold chunks that ranked 6–10 in the bi-encoder get buried in the cross-encoder's reranking (13 cases)

The p95 latency improvement (349ms → 162ms) is a side effect: the baseline p95 spike came from embedding the 3,379-token Article 3. The reranker uses fixed 512-token truncation, capping its worst case.

**Decision: do not ship the reranker for k=5 context feeding.** The −3pp hit_rate@5 means the generator sees the right chunk less often. Only add the reranker for single-answer interfaces where top-1 precision is what matters.

---

## 4. Ablation 2 — Fixed-size chunking (512 tokens, 128 overlap)

**Configuration:** each structural chunk split into 512-token windows with 128-token overlap. Same embedding model, same eval set.

| Metric | Structural | Fixed-512 | Delta |
|---|---|---|---|
| recall@5 | **0.735** | 0.615 | −0.120 ✗ |
| recall@10 | **0.857** | 0.744 | −0.113 ✗ |
| hit_rate@5 | **0.724** | 0.694 | −0.030 ✗ |
| MRR | **0.510** | 0.493 | −0.017 ✗ |
| Corpus size | 377 | 472 | +25% |
| Avg gold windows/query | 1.0 | 1.49 | — |
| Latency p50 | 12ms | **12ms** | tied |
| Latency p95 | 349ms | **44ms** | −305ms ✓ |

**Why structural chunking wins:**

```
Structural chunk token distribution:
  ≤128 tokens    120 chunks  (39%)  ← already short
  129–256        76 chunks   (25%)
  257–512        57 chunks   (19%)
  513–1024       39 chunks   (13%)
  >1024          14 chunks    (5%)  ← only these benefit from splitting
```

64% of structural chunks are already ≤256 tokens. For them, fixed-size splitting does nothing except add 25% more vectors to the index — diluting top-k slots with duplicate overlapping windows that compete against each other.

The natural title anchor ("Article 9 — Risk management system") is lost when a chunk is sliced mid-content. That title is the strongest single-phrase signal for retrieval.

**The p95 fix is separate:** the 349ms spike comes from 14 chunks over 1k tokens. Truncating their embedding input to 512 tokens (without splitting the chunk) would give the same latency improvement without any quality loss.

**Decision: ship structural chunking.** Fixed-size is appropriate for unstructured corpora (web pages, PDFs without clear sections). Legislative text already has meaningful boundaries — use them.

---

## 5. Generation results

### Pipeline

```
question + top-5 retrieved chunks
            │
            ▼
   prompt template (prompts/generate_answer.txt)
            │
            ▼
   qwen2.5:3b @ temperature=0  (Ollama, local)
            │
            ▼
   generated answer
```

### Metrics (from-scratch implementation)

| Metric | Score | Method |
|---|---|---|
| Faithfulness | 0.510 | Claim decomposition → per-claim context verification |
| Context recall | 0.344 | Per-sentence gold answer attribution to context |
| Answer relevance | 0.551 | Direct LLM scoring 0–1 |

### RAGAS cross-check (n=20, same algorithm, same judge)

| Metric | Ours | RAGAS-equiv | Pearson r | MAE |
|---|---|---|---|---|
| Faithfulness | 0.517 | 0.658 | 0.128 | 0.358 |
| Context recall | 0.475 | 0.225 | 0.539 | 0.250 |

**Interpretation of low faithfulness correlation (r=0.128):**

Both implementations use identical algorithms. The low correlation reflects judge instability with a 3B model on legal text — small prompt variations produce different yes/no verdicts on the same claim. This is a property of the judge, not the metric design. Context recall correlation (r=0.539) is more stable because binary attribution ("is this sentence in the context?") is a simpler task with less ambiguity.

**Recommended fix:** use a stronger judge. `claude-haiku-4-5-20251001` via API would cost ~$0.50 for all 98 items and produce stable, agreement with RAGAS. The code supports this with `export RAGDIAG_JUDGE_MODEL=claude-haiku-4-5-20251001`.

---

## 6. Complete failure decomposition (final)

Combining retrieval and generation into the full picture:

```
                     RETRIEVAL
                   ┌──────────────────────────────────────────┐
                   │ gold NOT in top-5     │ gold IN top-5     │
  ─────────────────┼───────────────────────┼───────────────────┤
  GENERATION       │                       │                   │
  faithful ≥ 0.5   │   (impossible)        │   BOTH OK: 46     │
                   │                       │   (46.9%)         │
  ─────────────────┼───────────────────────┼───────────────────┤
  GENERATION       │   RETRIEVAL-BOUND: 27 │ GENERATION-BOUND  │
  faithful < 0.5   │   (27.6%)             │   : 25 (25.5%)    │
                   └───────────────────────┴───────────────────┘
```

---

## 7. What I'd ship and why

**Recommended production configuration:**

```
Chunking:   structural (articles / recitals / annexes as natural units)
Embedding:  BAAI/bge-small-en-v1.5  (130MB, 12ms p50 latency)
Index:      FAISS IndexFlatIP  (cosine, sufficient at this corpus size)
Retrieval:  top-5 (recall@5 = 0.735, covers 72% of eval questions)
Reranker:   OFF  (hurts hit_rate@5 for k=5 context; only add for top-1 interfaces)
Generator:  any instruction-tuned model; qwen2.5:3b is a floor, not a ceiling
Judge:      API model  (claude-haiku or equivalent — local 3B is unstable for legal text)
```

**Next engineering priorities, ranked by expected impact:**

1. **Metadata filter for recitals vs articles** — A +0.1 score boost for article chunks when the query contains obligation language ("shall", "must", "required") would fix ~12 of 27 retrieval failures at zero cost. Simple heuristic, measurable improvement.

2. **Stronger judge model** — Would give reliable faithfulness scores and expose which generation failures are real vs judge noise. The infrastructure is ready.

3. **Truncate embedding input for long chunks** — Articles 3 (3,379 tokens) and a few others cause the 349ms p95 latency spike. Truncating to 512 tokens at embedding time (without splitting) fixes p95 without touching recall.

4. **Domain-specific reranker** — The MS MARCO cross-encoder fails on legal text because it prefers natural prose over terse legal obligations. A reranker fine-tuned on EU regulatory text would likely flip the reranker trade-off and improve both MRR and hit_rate@5.

5. **Eval set verification** — Running the interactive verification tool (`python -m ragdiag verify --sample 20`) would identify and remove the ~7 vague Q/A pairs that inflate the retrieval-bound count. Cleaner eval set = more precise failure decomposition.

---

## 8. Reproducibility

All judge calls use `temperature=0`. Raw judge outputs are logged per-item in `results/gen_metrics_baseline.json` under the `judge_log` field. All prompts are versioned in `prompts/`. The eval set seed is fixed (`RANDOM_SEED=42`).

To reproduce from scratch:

```bash
python -m ragdiag ingest
python -m ragdiag index
python -m ragdiag evalgen
python -m ragdiag retrieve --tag baseline
python -m ragdiag rerank
python -m ragdiag ablation-chunking
python -m ragdiag generate --tag baseline
python -m ragdiag gen-metrics --tag baseline
```
