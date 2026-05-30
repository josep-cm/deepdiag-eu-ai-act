"""Download the EU AI Act HTML from EUR-Lex with fallback to ELI endpoint."""

import hashlib
import sys
import time
from pathlib import Path

import requests

PRIMARY_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
)
FALLBACK_URL = "http://data.europa.eu/eli/reg/2024/1689/oj"

HEADERS = {
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (compatible; rag-diagnostics/0.1; "
        "+https://github.com/research)"
    ),
}
TIMEOUT = 60
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def _fetch(url: str, session: requests.Session) -> requests.Response:
    for attempt in range(1, 4):
        try:
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == 3:
                raise
            wait = 2 ** attempt
            print(f"  attempt {attempt} failed ({exc}); retrying in {wait}s…")
            time.sleep(wait)


def download(lang: str = "EN", force: bool = False) -> Path:
    """Download the AI Act HTML and return the local path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"ai_act_{lang.lower()}.html"

    if dest.exists() and not force:
        print(f"[download] cached → {dest}")
        return dest

    primary = PRIMARY_URL.replace("/EN/", f"/{lang}/")
    urls = [primary, FALLBACK_URL]

    with requests.Session() as session:
        for url in urls:
            print(f"[download] fetching {url} …")
            try:
                r = _fetch(url, session)
                content_type = r.headers.get("Content-Type", "")
                if "html" not in content_type and "xml" not in content_type:
                    print(f"  unexpected content-type: {content_type}; skipping")
                    continue
                raw = r.content
                dest.write_bytes(raw)
                digest = hashlib.sha256(raw).hexdigest()[:12]
                print(
                    f"[download] saved {len(raw):,} bytes  sha256={digest}  → {dest}"
                )
                return dest
            except requests.RequestException as exc:
                print(f"  failed: {exc}")

    print("[download] all sources failed", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    download()
