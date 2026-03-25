#!/usr/bin/env python3
"""
LSC Data Collection Pipeline
=============================
Collects raw sentences demonstrating word usage across two eras:
  1. Modern Era  — via DuckDuckGo web search + article scraping
  2. Deep-Time Era (Old / Middle English) — OED (primary), Wiktionary, MED (fallbacks)

Outputs a unified JSON file at  data/lsc_context_data.json

OED access
----------
Place your exported browser cookies (from www.oed.com after UofT library login)
at  scripts/oed_cookie.json  (Cookie-Editor JSON format).
If the file is absent the pipeline falls back to Wiktionary + MED.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup, Tag
from ddgs import DDGS

import nltk

# ---------------------------------------------------------------------------
# Bootstrap NLTK data
# ---------------------------------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("lsc_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "LSC-Research-Bot/1.0 "
        "(Academic project, University of Toronto; "
        "contact: alice@mail.utoronto.ca)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json",
}
POLITE_DELAY = 1.5  # seconds between outbound requests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "lsc_context_data.json"
OED_COOKIE_FILE = Path(__file__).resolve().parent / "oed_cookie.json"


def _sleep():
    """Polite delay between HTTP requests."""
    time.sleep(POLITE_DELAY)


# ===================================================================
# 1. Modern Era Scraper
# ===================================================================
class ModernScraper:
    """Search the open web for modern-usage sentences of a target word."""

    def __init__(self, max_urls: int = 8, target_sentences: int = 10):
        self.max_urls = max_urls
        self.target_sentences = target_sentences

    # ----- public -----
    def collect(self, word: str) -> list[str]:
        """Return up to *target_sentences* modern sentences containing *word*."""
        log.info("ModernScraper: collecting for '%s'", word)
        urls = self._search(word)
        sentences: list[str] = []
        for url in urls:
            if len(sentences) >= self.target_sentences:
                break
            new = self._extract_sentences(url, word)
            sentences.extend(new)
            _sleep()
        sentences = sentences[: self.target_sentences]
        log.info(
            "ModernScraper: got %d modern sentence(s) for '%s'",
            len(sentences),
            word,
        )
        return sentences

    # ----- helpers -----
    def _search(self, word: str) -> list[str]:
        """Use DuckDuckGo to find URLs relevant to *word*."""
        try:
            results = DDGS().text(
                f"{word} usage example sentence",
                max_results=self.max_urls,
            )
            urls = [r["href"] for r in results if "href" in r]
            log.info("  Search returned %d URL(s)", len(urls))
            return urls
        except Exception as exc:
            log.warning("  DuckDuckGo search failed: %s", exc)
            return []

    def _extract_sentences(self, url: str, word: str) -> list[str]:
        """Fetch *url*, strip boilerplate, return sentences containing *word*."""
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            log.debug("  Fetch failed (%s): %s", url, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content tags
        for tag_name in ("script", "style", "nav", "footer", "header", "aside", "form"):
            for tag in soup.find_all(tag_name):
                tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)

        raw_sents = sent_tokenize(text)
        word_lower = word.lower()
        good: list[str] = []
        for s in raw_sents:
            s = s.strip()
            # Strip leading vote-count numbers (e.g. "600 281 Just as fast")
            s = re.sub(r"^\d+\s+\d+\s+", "", s)
            # Strip leading parenthesized numbers (e.g. "(10) The afflicted...")
            s = re.sub(r"^\(\d+\)\s*", "", s)
            # Strip leading "Advertisement " or "..."
            s = re.sub(r"^(Advertisement\s+|\.{2,}\s*)", "", s)
            s = s.strip()
            if word_lower not in s.lower():
                continue
            # Sanity: skip very short / very long artefacts
            if len(s) < 30 or len(s) > 600:
                continue
            # Skip sentences that look like page titles / nav boilerplate
            if re.match(r"^.{0,5}(Definition|Usage|Pronunciation|Sentence Examples)", s, re.I):
                continue
            good.append(s)
        return good


# ===================================================================
# 2a. OED Scraper (primary historical source)
# ===================================================================
class OEDScraper:
    """Fetch dated quotations from the Oxford English Dictionary.

    Requires browser session cookies exported from www.oed.com after
    authenticating via UofT library (or any institutional access).
    Cookie file format: Cookie-Editor JSON export (list of dicts with
    'name', 'value', 'domain' keys).

    Date parsing
    ------------
    OED uses labels like "OE", "lOE", "a1300", "c1400", "1481".
    These are mapped to approximate integer years so quotes can be
    filtered by era.
    """

    BASE_URL = "https://www.oed.com"
    SEARCH_URL = "https://www.oed.com/search/dictionary/?scope=Entries&q={word}"

    # Approximate midpoint years for OED period abbreviations
    _PERIOD_YEARS: dict[str, int] = {
        "oe": 900,
        "loe": 1050,
        "eoe": 800,
        "eme": 1200,
        "me": 1350,
        "lme": 1470,
        "emode": 1580,
    }

    def __init__(self, cookie_path: Path, polite_delay: float = POLITE_DELAY):
        self.polite_delay = polite_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.oed.com/",
        })
        self._load_cookies(cookie_path)

    def _load_cookies(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            for c in json.load(f):
                self.session.cookies.set(
                    c["name"], c["value"],
                    domain=c.get("domain", "www.oed.com"),
                )

    # ----- public -----

    def collect_historical(self, word: str, before_year: int = 1900) -> tuple[list[str], str]:
        """Return (quotes, source_label) for OED quotations before *before_year*."""
        entry_url = self._find_entry_url(word)
        if not entry_url:
            log.warning("OED: no entry found for '%s'", word)
            return [], "OED (not found)"
        time.sleep(self.polite_delay)
        quotes = self._fetch_quotes(entry_url, max_year=before_year - 1)
        log.info("OED: %d historical quote(s) for '%s' (before %d)", len(quotes), word, before_year)
        return quotes, "OED"

    def collect_modern(self, word: str, after_year: int = 1900) -> list[str]:
        """Return OED quotations from *after_year* onwards."""
        entry_url = self._find_entry_url(word)
        if not entry_url:
            return []
        time.sleep(self.polite_delay)
        quotes = self._fetch_quotes(entry_url, min_year=after_year)
        log.info("OED: %d modern quote(s) for '%s' (from %d)", len(quotes), word, after_year)
        return quotes

    # ----- entry URL resolution -----

    def _find_entry_url(self, word: str) -> Optional[str]:
        """Resolve a word to its OED entry URL.

        Strategy: try common POS slugs directly, then fall back to search.
        """
        slug = word.lower().replace(" ", "_")
        for suffix in ("_n", "_v", "_adj", "_adv", ""):
            url = f"{self.BASE_URL}/dictionary/{slug}{suffix}"
            try:
                resp = self.session.get(url, timeout=12, allow_redirects=True)
                if resp.status_code == 200 and "/dictionary/" in resp.url:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    if soup.find("li", class_="quotation"):
                        log.debug("OED: resolved '%s' → %s", word, resp.url)
                        return resp.url
            except Exception as exc:
                log.debug("OED direct slug failed (%s): %s", url, exc)
            time.sleep(0.4)

        return self._search_entry_url(word)

    def _search_entry_url(self, word: str) -> Optional[str]:
        search_url = self.SEARCH_URL.format(word=quote(word, safe=""))
        try:
            resp = self.session.get(search_url, timeout=12)
            resp.raise_for_status()
        except Exception as exc:
            log.debug("OED search failed: %s", exc)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=re.compile(r"/dictionary/")):
            href = a.get("href", "")
            if href and "#" not in href:
                return urljoin(self.BASE_URL, href)
        return None

    # ----- quote extraction -----

    def _fetch_quotes(
        self,
        url: str,
        min_year: int = 0,
        max_year: int = 9999,
    ) -> list[str]:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            log.warning("OED page fetch failed (%s): %s", url, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[str] = []

        for li in soup.find_all("li", class_="quotation"):
            # --- date ---
            date_div = li.find("div", class_="quotation-date")
            if not date_div:
                continue
            date_span = date_div.find("span", class_="date")
            year = self._parse_date(date_span.get_text(strip=True) if date_span else "")
            if year is None or not (min_year <= year <= max_year):
                continue

            # --- quote text ---
            blockquote = li.find("blockquote", class_="quotation-text")
            if not blockquote:
                continue

            # Work on a copy so we don't mutate the tree
            bq = BeautifulSoup(str(blockquote), "html.parser").find("blockquote")
            # Remove editorial comments (Latin originals, glosses, etc.)
            for span in bq.find_all("span", class_=re.compile(r"editorial.comment", re.I)):
                span.decompose()
            # Replace <mark> (keyword highlight) with its plain text
            for mark in bq.find_all("mark"):
                mark.replace_with(mark.get_text())
            # Remove any remaining citation noise
            for cite in bq.find_all("cite"):
                cite.decompose()

            text = re.sub(r"\s+", " ", bq.get_text(separator=" ", strip=True)).strip()

            if 20 <= len(text) <= 800:
                results.append(text)

        return results

    # ----- date parsing -----

    def _parse_date(self, raw: str) -> Optional[int]:
        """Map an OED date string to an approximate integer year."""
        if not raw:
            return None
        s = raw.lower().strip()

        # Period label lookup (exact match)
        if s in self._PERIOD_YEARS:
            return self._PERIOD_YEARS[s]

        # 4-digit year (possibly prefixed by a, c, >, <, l, e)
        m = re.search(r"\d{4}", s)
        if m:
            return int(m.group())

        # 3-digit year (rare but exists, e.g. c971)
        m = re.search(r"\d{3}", s)
        if m:
            return int(m.group())

        return None


# ===================================================================
# 2b. Deep-Time Era Scraper (Wiktionary + MED fallbacks)
# ===================================================================
class HistoricalScraper:
    """Fetch Old/Middle English quotations via Wiktionary & MED with DOM parsing, sanitization, and homograph filtering."""

    WIKTIONARY_API = "https://en.wiktionary.org/api/rest_v1/page/html/{word}"
    MED_SEARCH = (
        "https://quod.lib.umich.edu/m/middle-english-dictionary/dictionary"
        "?utf8=%E2%9C%93&search_field=hnf&q={word}"
    )

    def __init__(self, target_quotes: int = 10, use_llm_filter: bool = True):
        self.target_quotes = target_quotes
        self.use_llm_filter = use_llm_filter

        # OED scraper (primary) — only if cookie file exists
        self.oed: Optional[OEDScraper] = None
        if OED_COOKIE_FILE.exists():
            try:
                self.oed = OEDScraper(OED_COOKIE_FILE)
                log.info("HistoricalScraper: OED scraper enabled (cookie file found).")
            except Exception as exc:
                log.warning("HistoricalScraper: could not init OED scraper: %s", exc)

        # Initialize Google GenAI client if available for homograph filtering
        self.llm = None
        if self.use_llm_filter:
            api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
            if api_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    model_name = os.getenv("DEFAULT_MODEL_GAS", "gemini-2.0-flash-lite")
                    self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)
                    log.info("HistoricalScraper: LLM validation enabled for homograph filtering.")
                except ImportError:
                    log.warning("HistoricalScraper: langchain_google_genai not installed. LLM validation disabled.")

    # ----- public -----
    def collect(self, word: str, target_meaning: Optional[str] = None) -> tuple[list[str], str]:
        """Return (quotes, source_label).

        Priority: OED (if cookie file present) → Wiktionary → MED.
        """
        log.info("HistoricalScraper: collecting for '%s' (target meaning: %s)", word, target_meaning)

        # Strategy 1: OED
        if self.oed is not None:
            quotes, label = self.oed.collect_historical(word, before_year=1900)
            if quotes:
                # Run LLM meaning filter if available
                if self.llm and target_meaning:
                    quotes = [q for q in quotes if self._validate_meaning_llm(word, q, target_meaning)]
                if quotes:
                    return quotes[: self.target_quotes], label

        # Strategy 2: Wiktionary
        quotes = self._strategy_wiktionary(word, target_meaning)
        if quotes:
            return quotes[: self.target_quotes], "Wiktionary"

        # Strategy 3: MED
        log.info("  Wiktionary yielded nothing — falling back to MED")
        _sleep()
        quotes = self._strategy_med(word, target_meaning)
        if quotes:
            return quotes[: self.target_quotes], "MED (UMich)"

        log.warning("  No historical quotes found for '%s'", word)
        return [], "none"

    # ----- Sanitization & Validation -----
    def _sanitize_historical_quote(self, text: str) -> Optional[str]:
        """Filter out obvious artifacts, short/long strings, and lack of sentence structure."""
        if not text:
            return None
        text = re.sub(r"\s+", " ", text).strip()
        
        # Length check: historically, sentences are at least a few words and max a paragraph
        if len(text) < 20 or len(text) > 800:
            return None

        # Ensure it has basic spaces (i.e. not a single massive token)
        if " " not in text:
            return None
            
        lower_s = text.lower()
        
        # UI/boilerplate Artifacts to filter out completely
        artifacts = [
            "wiktionary", 
            "translations below need to be checked",
            "see instructions at",
            "definition:",
            "synonyms:",
            "antonyms:",
            "pronunciation:",
            "etymology",
            "isbn",
            "retrieved from",
            "quotations ▼",
        ]
        if any(art in lower_s for art in artifacts):
            return None
            
        return text

    def _validate_meaning_llm(self, word: str, quote: str, target_meaning: Optional[str]) -> bool:
        """Use LLM to check if the historical quote matches the target meaning (to avoid false homographs)."""
        if not target_meaning or not self.llm:
            return True  # Fallback to accepting if no meaning provided or LLM unavailable
            
        prompt = (
            f"You are a computational linguist evaluating historical text.\n"
            f"Does the following historical sentence use the word '{word}' in a sense related to '{target_meaning}' "
            f"(or as its semantic ancestor)? Answer EXACTLY with 'YES' or 'NO'.\n\n"
            f"Sentence: {quote}"
        )
        try:
            from langchain_core.messages import HumanMessage
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            return "YES" in resp.content.upper()
        except Exception as e:
            log.debug("LLM validation failed for quote, accepting by default: %s", e)
            return True

    # ----- Strategy A: Wiktionary REST API -----
    def _strategy_wiktionary(self, word: str, target_meaning: Optional[str]) -> list[str]:
        url = self.WIKTIONARY_API.format(word=quote(word, safe=""))
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            log.debug("  Wiktionary fetch failed: %s", exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        quotes: list[str] = []

        historical_sections = self._find_historical_sections(soup)

        for section in historical_sections:
            # Quotations are usually in <div class="h-quotation"> or <li> with class "quotation"
            for container in section.find_all(["li", "div"], class_=re.compile(r"quotation|quote", re.I)):
                # Clone node to avoid modifying the original soup
                node = container.__copy__() if hasattr(container, '__copy__') else container
                
                # STRICT DOM PARSING: explicitly remove <cite> tags (book titles, authors)
                for cite in node.find_all("cite"):
                    cite.decompose()
                
                # Explicitly remove translation spans/divs
                for trans in node.find_all(class_=re.compile(r"translation|summary|note", re.I)):
                    trans.decompose()

                # Find the actual quote text node (often <span class="e-quotation">, <q>, or <i>)
                q_node = node.find(["q", "span", "i"], class_=re.compile(r"quotation|quote", re.I))
                if not q_node:
                    q_node = node.find(["q", "i"])
                
                # Fallback to the whole container if no specific text node is found
                text_node = q_node if q_node else node
                text = text_node.get_text(separator=" ", strip=True)
                
                cleaned = self._sanitize_historical_quote(text)
                if cleaned:
                    quotes.append(cleaned)
                    
            # Fallback for plain <dd> that just have an <i> inside a definition list
            for dd in section.find_all("dd"):
                # Skip if already processed as part of a .quotation container
                if dd.find_parent(class_=re.compile(r"quotation|quote", re.I)):
                    continue
                    
                node = dd.__copy__() if hasattr(dd, '__copy__') else dd
                for cite in node.find_all("cite"):
                    cite.decompose()
                
                italic = node.find("i")
                if italic:
                    text = italic.get_text(separator=" ", strip=True)
                    cleaned = self._sanitize_historical_quote(text)
                    if cleaned:
                        quotes.append(cleaned)

        # Deduplicate, maintaining order
        seen: set[str] = set()
        unique: list[str] = []
        for q in quotes:
            if q not in seen:
                seen.add(q)
                if self._validate_meaning_llm(word, q, target_meaning):
                    unique.append(q)

        log.info("  Wiktionary: found %d valid quote(s) for '%s'", len(unique), word)
        return unique

    def _find_historical_sections(self, soup: BeautifulSoup) -> list[Tag]:
        """Return <section> or heading-delimited blocks for OE / ME."""
        targets = []
        # Try <section> wrappers first (modern Wiktionary HTML)
        for sec in soup.find_all("section"):
            heading = sec.find(re.compile(r"^h[1-6]$"))
            if heading and re.search(r"(Old\s+English|Middle\s+English|Etymology)", heading.get_text(), re.I):
                targets.append(sec)

        # Fallback: walk headings and grab everything until next same-level heading
        if not targets:
            for heading in soup.find_all(re.compile(r"^h[1-6]$")):
                htext = heading.get_text()
                if re.search(r"(Old\s+English|Middle\s+English)", htext, re.I):
                    level = int(heading.name[1])
                    container = BeautifulSoup("<div></div>", "html.parser").div
                    for sib in heading.find_next_siblings():
                        if isinstance(sib, Tag) and re.match(r"^h[1-6]$", sib.name):
                            if int(sib.name[1]) <= level:
                                break
                        container.append(sib.__copy__() if hasattr(sib, '__copy__') else sib) # type: ignore
                    targets.append(container)

        return targets

    # ----- Strategy B: Middle English Dictionary (UMich) -----
    def _strategy_med(self, word: str, target_meaning: Optional[str]) -> list[str]:
        search_url = self.MED_SEARCH.format(word=quote(word, safe=""))
        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            log.debug("  MED search failed: %s", exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        quotes: list[str] = []

        entry_link = soup.find("a", href=re.compile(r"/m/middle-english-dictionary/dictionary/MED"))
        if entry_link:
            entry_url = urljoin(search_url, entry_link["href"])
            _sleep()
            quotes = self._parse_med_entry(entry_url)
        else:
            quotes = self._extract_med_quotes_from_soup(soup)
            
        # Filter artifacts and false homographs
        valid_quotes: list[str] = []
        for q in quotes:
            cleaned = self._sanitize_historical_quote(q)
            if cleaned and self._validate_meaning_llm(word, cleaned, target_meaning):
                valid_quotes.append(cleaned)

        log.info("  MED: found %d valid quote(s) for '%s'", len(valid_quotes), word)
        return valid_quotes

    def _parse_med_entry(self, url: str) -> list[str]:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            log.debug("  MED entry fetch failed: %s", exc)
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        return self._extract_med_quotes_from_soup(soup)

    def _extract_med_quotes_from_soup(self, soup: BeautifulSoup) -> list[str]:
        """Pull quotation text from MED HTML with strict DOM parsing."""
        quotes: list[str] = []

        # MED uses various class names for citations
        for selector in ("blockquote", ".citation", ".quotation", ".cit", ".quote", "span.q", "span.cit", ".eg"):
            for el in soup.select(selector):
                node = el.__copy__() if hasattr(el, '__copy__') else el
                
                # Remove author/biblio tags typically found in MED
                for bib in node.find_all(class_=re.compile(r"bibl|author|title", re.I)):
                    bib.decompose()
                
                text = node.get_text(separator=" ", strip=True)
                quotes.append(text)

        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for q in quotes:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique


# ===================================================================
# 3. Pipeline orchestration
# ===================================================================
def run_pipeline(words: list[dict | str]) -> list[dict]:
    """Run both scrapers for every word; return list of result dicts.
    Accepts either a list of string words, or a list of dicts:
      [{"word": "bad", "target_meaning": "evil, not good"}, ...]
    """
    modern = ModernScraper(max_urls=8, target_sentences=10)
    historical = HistoricalScraper(target_quotes=10, use_llm_filter=True)

    results: list[dict] = []
    for item in words:
        if isinstance(item, dict):
            word = item["word"]
            target_meaning = item.get("target_meaning")
        else:
            word = str(item)
            target_meaning = None
            
        log.info("=" * 60)
        log.info("Processing word: %s", word)
        log.info("=" * 60)

        modern_ctx = modern.collect(word)
        _sleep()
        hist_ctx, hist_src = historical.collect(word, target_meaning=target_meaning)

        results.append(
            {
                "word": word,
                "modern_context": modern_ctx,
                "historical_context": hist_ctx,
                "historical_source": hist_src,
            }
        )
        _sleep()

    return results


def save_results(results: list[dict], path: Optional[Path] = None) -> Path:
    """Write results to JSON and return the path."""
    path = path or OUTPUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Results written to %s", path)
    return path


# ===================================================================
# 4. main()
# ===================================================================
def main():
    import csv
    
    # Path to LSC-CTD dataset
    tsv_path = PROJECT_ROOT / "data" / "LSC-CTD" / "blank_dataset.tsv"
    
    # Extract the 25 :engl words and their original meanings
    test_words = []
    seen_words = set()
    
    try:
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # Skip header
            for row in reader:
                if len(row) > 7:
                    raw_word = row[0].strip()
                    if raw_word.endswith(":engl"):
                        clean_word = raw_word.removesuffix(":engl").strip()
                        # Strip leading "to " for verb forms
                        if clean_word.startswith("to "):
                            clean_word = clean_word[3:]
                            
                        # Avoid duplicates (like 'corn' appearing 3x)
                        if clean_word not in seen_words:
                            # Use English translation of old meaning (column 7 / index 7) as target_meaning
                            old_meaning_eng = row[7].strip()
                            test_words.append({
                                "word": clean_word,
                                "target_meaning": old_meaning_eng
                            })
                            seen_words.add(clean_word)
    except Exception as e:
        log.error("Failed to parse LSC dataset to get 25 words: %s", e)
        return
        
    log.info("Starting LSC data pipeline for %d words", len(test_words))

    results = run_pipeline(test_words)
    out = save_results(results)

    # Quick summary
    print("\n" + "=" * 60)
    print("Pipeline complete — results at", out)
    print("=" * 60)
    for entry in results:
        print(
            f"  {entry['word']:12s}  "
            f"modern={len(entry['modern_context']):2d}  "
            f"historical={len(entry['historical_context']):2d}  "
            f"(source: {entry['historical_source']})"
        )
    print()


if __name__ == "__main__":
    main()
