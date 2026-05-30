"""
Fixed-size chunker: splits chunk text into 512-token windows with 128-token overlap.

Produces the same schema as the structural chunker so they're drop-in comparable:
  chunk_id, type, number, title, text, cross_references, source_url, lang

For fixed chunks:
  chunk_id  = "fixed-{source_chunk_id}-{part_index}"
  type      = "fixed"
  number    = source chunk_id (for traceability)

The overlap keeps sentences from being cut mid-thought and improves recall on
boundary-spanning answers — but it increases corpus size.
"""

import json
import re
from pathlib import Path

import tiktoken

CHUNKS_PATH = Path("data/chunks/ai_act_en.json")
FIXED_CHUNKS_PATH = Path("data/chunks/ai_act_en_fixed512.json")

WINDOW_TOKENS = 512
OVERLAP_TOKENS = 128
ENCODING = "cl100k_base"  # same as GPT-4 / text-embedding-ada; close enough for splitting


def _split_into_windows(text: str, enc, window: int, overlap: int) -> list[str]:
    tokens = enc.encode(text)
    if len(tokens) <= window:
        return [text]

    windows = []
    step = window - overlap
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + window]
        windows.append(enc.decode(chunk_tokens))
        if start + window >= len(tokens):
            break
    return windows


_CROSS_REF_PATTERN = re.compile(
    r"\bArticle[s]?\s+(\d+(?:\s*(?:,|and|or|to)\s*\d+)*)"
    r"|\bAnnex\s+(I{1,3}|I?V|VI{0,3}|IX|X[I-V]?|XI{0,3})\b"
    r"|\bparagraph\s+(\d+)\b",
    re.IGNORECASE,
)


def _cross_refs(text: str) -> list[str]:
    seen, out = set(), []
    for m in _CROSS_REF_PATTERN.finditer(text):
        r = m.group(0).strip()
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def build(force: bool = False) -> list[dict]:
    if FIXED_CHUNKS_PATH.exists() and not force:
        print(f"[chunk_fixed] cached → {FIXED_CHUNKS_PATH}")
        return json.loads(FIXED_CHUNKS_PATH.read_text())

    structural = json.loads(CHUNKS_PATH.read_text())
    enc = tiktoken.get_encoding(ENCODING)

    fixed_chunks = []
    for src in structural:
        windows = _split_into_windows(src["text"], enc, WINDOW_TOKENS, OVERLAP_TOKENS)
        for i, window_text in enumerate(windows):
            fixed_chunks.append({
                "chunk_id": f"fixed-{src['chunk_id']}-{i}",
                "type": "fixed",
                "number": src["chunk_id"],   # source traceability
                "title": src.get("title"),
                "text": window_text,
                "cross_references": _cross_refs(window_text),
                "source_url": src["source_url"],
                "lang": src["lang"],
                # Keep source metadata for mapping back to gold
                "_source_chunk_id": src["chunk_id"],
                "_part_index": i,
                "_total_parts": len(windows),
            })

    FIXED_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXED_CHUNKS_PATH.write_text(json.dumps(fixed_chunks, ensure_ascii=False, indent=2))

    src_count = len(structural)
    token_counts = [len(enc.encode(c["text"])) for c in structural]
    avg_tokens = sum(token_counts) / len(token_counts)
    print(
        f"[chunk_fixed] {src_count} structural → {len(fixed_chunks)} fixed chunks "
        f"(avg src={avg_tokens:.0f} tok, window={WINDOW_TOKENS}, overlap={OVERLAP_TOKENS})"
    )
    return fixed_chunks
