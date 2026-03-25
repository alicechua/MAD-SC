"""Etymology data fetching for the MAD-SC Lexicographer Agent.

Fetch Strategy (in priority order)
------------------------------------
1. OED (Oxford English Dictionary) — dated historical + modern quotations.
   Requires a cookie file at scripts/oed_cookie.json (UofT library session).
   Many entries lazy-load via JavaScript; falls through when 0 quotes returned.
2. Wiktionary — Etymology section text from the REST API (no auth required).
   Provides concise historical lineage, often sufficient for the LLM.
3. None — lexicographer LLM uses parametric knowledge only.

The returned EtymologyContext dict is passed verbatim into the lexicographer
LLM prompt so it can reason from real evidence rather than parametric memory.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_OED_COOKIE_FILE = Path(__file__).resolve().parent.parent / "scripts" / "oed_cookie.json"
_POLITE_DELAY = 0.5  # seconds between OED requests

# ---------------------------------------------------------------------------
# OED scraper (lightweight, adapted from scripts/lsc_data_pipeline.py)
# ---------------------------------------------------------------------------

_OED_PERIOD_YEARS: dict[str, int] = {
    "oe": 900, "loe": 1050, "eoe": 800,
    "eme": 1200, "me": 1350, "lme": 1470, "emode": 1580,
}


def _parse_oed_date(raw: str) -> Optional[int]:
    if not raw:
        return None
    s = raw.lower().strip()
    if s in _OED_PERIOD_YEARS:
        return _OED_PERIOD_YEARS[s]
    m = re.search(r"\d{4}", s)
    if m:
        return int(m.group())
    m = re.search(r"\d{3}", s)
    if m:
        return int(m.group())
    return None


_OED_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _load_oed_cookies() -> Optional[list[dict]]:
    """Load OED cookies from the Cookie-Editor JSON file in Playwright format."""
    if not _OED_COOKIE_FILE.exists():
        return None
    try:
        with open(_OED_COOKIE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        pw_cookies = []
        for c in raw:
            cookie: dict = {
                "name": c["name"],
                "value": c["value"],
                "domain": c.get("domain", ".oed.com"),
                "path": c.get("path", "/"),
            }
            if c.get("expires", -1) != -1:
                cookie["expires"] = int(c["expires"])
            same_site = c.get("sameSite", "Lax")
            if same_site in ("Strict", "Lax", "None"):
                cookie["sameSite"] = same_site
            pw_cookies.append(cookie)
        return pw_cookies
    except Exception as exc:
        log.debug("OED cookie load failed: %s", exc)
        return None


def _oed_parse_quotes(
    html: str,
    min_year: int = 0,
    max_year: int = 9999,
    max_quotes: int = 8,
) -> list[tuple[int, str]]:
    """Parse (year, text) tuples from a fully-rendered OED entry page HTML."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[int, str]] = []

    for li in soup.find_all("li", class_="quotation"):
        date_div = li.find("div", class_="quotation-date")
        if not date_div:
            continue
        date_span = date_div.find("span", class_="date")
        year = _parse_oed_date(date_span.get_text(strip=True) if date_span else "")
        if year is None or not (min_year <= year <= max_year):
            continue

        blockquote = li.find("blockquote", class_="quotation-text")
        if not blockquote:
            continue

        bq = BeautifulSoup(str(blockquote), "html.parser").find("blockquote")
        for span in bq.find_all("span", class_=re.compile(r"editorial.comment", re.I)):
            span.decompose()
        for mark in bq.find_all("mark"):
            mark.replace_with(mark.get_text())
        for cite in bq.find_all("cite"):
            cite.decompose()

        text = re.sub(r"\s+", " ", bq.get_text(separator=" ", strip=True)).strip()
        if 20 <= len(text) <= 600:
            results.append((year, text))

        if len(results) >= max_quotes:
            break

    return results


def _fetch_oed_context(word: str) -> Optional[dict]:
    """Fetch dated historical and modern quotations from OED via Playwright.

    Uses a headless Chromium browser so that JavaScript-rendered quotations
    are available for scraping. Returns dict with keys 'historical' and
    'modern', each a list of (year, text) tuples. Returns None if OED is
    unavailable or returns 0 quotes.
    """
    cookies = _load_oed_cookies()
    if cookies is None:
        return None

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        log.debug("Playwright not installed; OED scraping disabled")
        return None

    slug = word.lower().replace(" ", "_")
    entry_url: Optional[str] = None
    page_html: Optional[str] = None

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=_OED_UA)
        ctx.add_cookies(cookies)
        page = ctx.new_page()

        # Probe slug variants to find a live entry with quotations.
        for suffix in ("_n", "_v", "_adj", "_adv", "_prep", "_conj", ""):
            url = f"https://www.oed.com/dictionary/{slug}{suffix}"
            try:
                resp = page.goto(url, timeout=20_000, wait_until="domcontentloaded")
                # Redirect to login or search = no access
                if resp is None or "/dictionary/" not in page.url:
                    time.sleep(_POLITE_DELAY)
                    continue
                # Wait for JS quotations — networkidle is most reliable for OED.
                try:
                    page.wait_for_load_state("networkidle", timeout=20_000)
                except PWTimeout:
                    pass  # Capture whatever loaded
                html = page.content()
                parsed = BeautifulSoup(html, "html.parser")
                if not parsed.find("li", class_="quotation"):
                    # Detect a hard auth redirect (URL left the /dictionary/ path).
                    if "/dictionary/" not in page.url:
                        log.warning(
                            "OED: session cookie expired — refresh scripts/oed_cookie.json "
                            "via Cookie-Editor after logging in through the UofT library proxy."
                        )
                        break  # All slugs will fail; no point probing further.
                    log.debug("OED: no quotations found at %s", url)
                    time.sleep(_POLITE_DELAY)
                    continue
                entry_url = page.url
                page_html = html
                break
            except PWTimeout:
                log.debug("OED: timeout navigating to %s", url)
            except Exception as exc:
                log.debug("OED: error navigating to %s: %s", url, exc)
            time.sleep(_POLITE_DELAY)

        browser.close()

    if entry_url is None or page_html is None:
        log.debug("OED: no entry URL found for '%s'", word)
        return None

    historical = _oed_parse_quotes(page_html, max_year=1899, max_quotes=6)
    modern = _oed_parse_quotes(page_html, min_year=1900, max_quotes=6)

    if not historical and not modern:
        log.debug("OED: 0 quotes parsed for '%s'", word)
        return None

    log.info("OED: %d historical + %d modern quotes for '%s'",
             len(historical), len(modern), word)
    return {"historical": historical, "modern": modern}


# ---------------------------------------------------------------------------
# Wiktionary etymology fallback
# ---------------------------------------------------------------------------

def _fetch_wiktionary_etymology(word: str) -> Optional[str]:
    """Fetch the Etymology section text from English Wiktionary.

    Uses the REST API HTML endpoint (no JS, no auth). Returns a plain-text
    paragraph summarising the word's historical origin, or None if unavailable.
    """
    url = f"https://en.wiktionary.org/api/rest_v1/page/html/{quote(word, safe='')}"
    try:
        resp = requests.get(
            url, timeout=10,
            headers={"User-Agent": "MAD-SC-Research/1.0 (academic; contact alice@mail.utoronto.ca)"},
        )
        resp.raise_for_status()
    except Exception as exc:
        log.debug("Wiktionary fetch failed for '%s': %s", word, exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the English section first, then look for Etymology within it.
    english_section = None
    for h2 in soup.find_all("h2"):
        if re.search(r"english", h2.get_text(), re.I):
            english_section = h2
            break

    search_root = english_section if english_section else soup

    # Collect text from paragraphs that follow an Etymology header.
    for h in search_root.find_all_next(["h3", "h4"]) if english_section else soup.find_all(["h3", "h4"]):
        if not re.search(r"etymology", h.get_text(), re.I):
            continue
        paragraphs = []
        for sib in h.find_next_siblings():
            if sib.name in ("h2", "h3", "h4"):
                break
            if sib.name == "p":
                text = re.sub(r"\s+", " ", sib.get_text(strip=True))
                if text:
                    paragraphs.append(text)
            if len(paragraphs) >= 3:
                break
        if paragraphs:
            result = " ".join(paragraphs)
            log.info("Wiktionary: etymology found for '%s' (%d chars)", word, len(result))
            return result

    log.debug("Wiktionary: no etymology section found for '%s'", word)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_etymology_context(word: str) -> dict:
    """Retrieve the best available etymological context for a target word.

    Returns a dict with:
        source        : "oed" | "wiktionary" | "none"
        historical    : list of (year, text) tuples from OED (may be empty)
        modern        : list of (year, text) tuples from OED (may be empty)
        etymology_text: Wiktionary etymology paragraph (may be None)

    Priority: OED dated quotes → Wiktionary etymology → nothing.
    The lexicographer LLM uses whatever is available to synthesise the dossier.
    """
    # 1. Try OED
    oed = _fetch_oed_context(word)
    if oed:
        return {
            "source": "oed",
            "historical": oed["historical"],
            "modern": oed["modern"],
            "etymology_text": None,
        }

    # 2. Try Wiktionary
    etym = _fetch_wiktionary_etymology(word)
    if etym:
        return {
            "source": "wiktionary",
            "historical": [],
            "modern": [],
            "etymology_text": etym,
        }

    # 3. Nothing available
    return {
        "source": "none",
        "historical": [],
        "modern": [],
        "etymology_text": None,
    }


def format_etymology_context_for_prompt(ctx: dict) -> str:
    """Format an etymology context dict as a readable prompt string."""
    parts = [f"Source: {ctx['source'].upper()}"]

    if ctx["historical"]:
        parts.append("\nOED historical quotations (pre-1900):")
        for year, text in ctx["historical"]:
            parts.append(f"  [{year}] {text}")
    else:
        parts.append("\nOED historical quotations: (none retrieved)")

    if ctx["modern"]:
        parts.append("\nOED modern quotations (1900–present):")
        for year, text in ctx["modern"]:
            parts.append(f"  [{year}] {text}")
    else:
        parts.append("\nOED modern quotations: (none retrieved)")

    if ctx["etymology_text"]:
        parts.append(f"\nWiktionary etymology:\n  {ctx['etymology_text']}")

    if ctx["source"] == "none":
        parts.append("\n(No external data available — use parametric knowledge.)")

    return "\n".join(parts)
