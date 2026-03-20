#!/usr/bin/env python3
"""
LSC Data Collection Pipeline
=============================
Collects raw sentences demonstrating word usage across two eras:
  1. Modern Era  — via DuckDuckGo web search + article scraping
  2. Deep-Time Era (Old / Middle English) — via Wiktionary API + MED fallback

Outputs a unified JSON file at  data/lsc_context_data.json
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
# 2. Deep-Time Era Scraper
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
        """
        Return (quotes, source_label).
        Tries Wiktionary first; falls back to MED.
        """
        log.info("HistoricalScraper: collecting for '%s' (target meaning: %s)", word, target_meaning)

        quotes = self._strategy_wiktionary(word, target_meaning)
        if quotes:
            return quotes[: self.target_quotes], "Wiktionary"

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
