"""
Structure-aware parser for the EU AI Act HTML (EUR-Lex format).

Document structure discovered:
  - Recitals: two-column <table> rows where left TD = "(N)" and right TD = text
  - Articles: <div class="eli-subdivision"> containing <p class="oj-ti-art">
  - Annexes: <div class="eli-container"> children with <p class="oj-doc-ti"> = "ANNEX N"

Each chunk is returned as a dict with the schema:
  {
    "chunk_id": str,            # e.g. "recital-1", "article-5", "annex-III"
    "type": "recital"|"article"|"annex",
    "number": str,              # "1", "5", "III"
    "title": str | None,        # article subtitle or annex title
    "text": str,                # clean body text
    "cross_references": list[str],  # article references extracted from text
    "source_url": str,
    "lang": str,
  }
"""

import re
import warnings
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SOURCE_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"

_CROSS_REF_PATTERN = re.compile(
    r"\bArticle[s]?\s+(\d+(?:\s*(?:,|and|or|to)\s*\d+)*)"
    r"|\bAnnex\s+(I{1,3}|I?V|VI{0,3}|IX|X[I-V]?|XI{0,3})\b"
    r"|\bparagraph\s+(\d+)\b",
    re.IGNORECASE,
)

_ROMAN_TO_INT = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13,
}


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_cross_refs(text: str) -> list[str]:
    refs = []
    for m in _CROSS_REF_PATTERN.finditer(text):
        refs.append(m.group(0).strip())
    # Deduplicate while preserving order
    seen = set()
    out = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _make_chunk(
    chunk_type: str,
    number: str,
    title: Optional[str],
    text: str,
    source_url: str,
    lang: str,
) -> dict:
    chunk_id = f"{chunk_type}-{number}"
    return {
        "chunk_id": chunk_id,
        "type": chunk_type,
        "number": number,
        "title": title,
        "text": text,
        "cross_references": _extract_cross_refs(text),
        "source_url": source_url,
        "lang": lang,
    }


def _parse_recitals(soup: BeautifulSoup, source_url: str, lang: str) -> list[dict]:
    chunks = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) != 1:
            continue
        tds = rows[0].find_all("td")
        if len(tds) != 2:
            continue
        left = _clean(tds[0].get_text(" "))
        m = re.fullmatch(r"\((\d+)\)", left)
        if not m:
            continue
        number = m.group(1)
        text = _clean(tds[1].get_text(" "))
        if not text:
            continue
        chunks.append(_make_chunk("recital", number, None, text, source_url, lang))
    return chunks


def _parse_articles(soup: BeautifulSoup, source_url: str, lang: str) -> list[dict]:
    chunks = []
    for div in soup.find_all("div", class_="eli-subdivision"):
        # Skip chapter-level wrappers — only take subdivisions nested inside another
        if not div.find_parent("div", class_="eli-subdivision"):
            continue
        ti_art = div.find("p", class_="oj-ti-art")
        if not ti_art:
            continue
        raw_num = _clean(ti_art.get_text(" "))
        m = re.search(r"Article\s+(\d+)", raw_num, re.IGNORECASE)
        if not m:
            continue
        number = m.group(1)

        sti = div.find("p", class_="oj-sti-art")
        title = _clean(sti.get_text(" ")) if sti else None
        # Strip the trailing backtick that appears in the source
        if title:
            title = title.rstrip("`").strip()

        # Simpler: join all text, then strip the header
        full = _clean(div.get_text(" "))
        # Remove "Article N" and subtitle from the front
        header_re = re.compile(
            r"^Article\s+\d+\s*" + re.escape(title or "") + r"\s*`?\s*",
            re.IGNORECASE,
        )
        body = header_re.sub("", full).strip()

        if not body:
            continue
        chunks.append(_make_chunk("article", number, title, body, source_url, lang))
    return chunks


def _parse_annexes(soup: BeautifulSoup, source_url: str, lang: str) -> list[dict]:
    chunks = []
    # Annexes: <div class="eli-container"> with a direct child <p class="oj-doc-ti"> matching "ANNEX"
    for container in soup.find_all("div", class_="eli-container"):
        # Only consider direct children
        direct_ps = [
            c for c in container.children
            if hasattr(c, "name") and c.name == "p"
        ]
        annex_header = None
        annex_title_tag = None
        for i, p in enumerate(direct_ps):
            cls = p.get("class", [])
            if "oj-doc-ti" in cls:
                txt = _clean(p.get_text(" "))
                m = re.match(r"ANNEX\s+([IVXLC]+|\d+)", txt, re.IGNORECASE)
                if m:
                    annex_header = p
                    number = m.group(1).upper()
                    # Next sibling p is the title
                    if i + 1 < len(direct_ps):
                        annex_title_tag = direct_ps[i + 1]
                    break

        if not annex_header:
            continue

        number = re.search(r"ANNEX\s+([IVXLC]+|\d+)", annex_header.get_text(), re.IGNORECASE).group(1).upper()
        title = _clean(annex_title_tag.get_text(" ")) if annex_title_tag else None

        # Full text of the annex container
        full = _clean(container.get_text(" "))
        # Strip the "ANNEX N" header from the front
        header_re = re.compile(
            r"^ANNEX\s+[IVXLC\d]+\s*" + re.escape(title or "") + r"\s*",
            re.IGNORECASE,
        )
        body = header_re.sub("", full).strip()
        if not body:
            continue

        chunks.append(_make_chunk("annex", number, title, body, source_url, lang))
    return chunks


def parse(html_path: Path, source_url: str = SOURCE_URL, lang: str = "EN") -> list[dict]:
    """Parse the EU AI Act HTML into structured chunks."""
    raw = html_path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")

    recitals = _parse_recitals(soup, source_url, lang)
    articles = _parse_articles(soup, source_url, lang)
    annexes = _parse_annexes(soup, source_url, lang)

    all_chunks = recitals + articles + annexes
    print(
        f"[parse] {len(recitals)} recitals  "
        f"{len(articles)} articles  "
        f"{len(annexes)} annexes  "
        f"→ {len(all_chunks)} total chunks"
    )
    return all_chunks
