#!/usr/bin/env python3
"""
Helsinki Corpus Wrapper
========================
Extracts historical sentences from the Helsinki Corpus TEI XML for
a list of target words, then merges them with the standard
lsc_data_pipeline results (Modern + Wiktionary/MED).

Output: data/lsc_context_data_engl.json
"""

import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Allow importing the main pipeline from the same scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lsc_data_pipeline import run_pipeline, save_results  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("helsinki_wrapper")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HC_XML_PATH = PROJECT_ROOT / "data" / "helsinki_corpus" / "corpora" / "HC_XML_Master_v9f.xml"
OUTPUT_FILE = PROJECT_ROOT / "data" / "lsc_context_data_engl.json"
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# The 25 :engl words from LSC-CTD, cleaned to their base search forms.
# Multi-word entries like "to observe" → "observe", "perfect lady" kept as-is.
RAW_ENGL_WORDS = [
    "bad", "bead", "before", "canine", "car", "corn", "fast",
    "hardly", "holiday", "horn", "humour", "lure", "mouse", "must",
    "perfect lady", "premises", "shall", "spaghetti", "spanish",
    "target", "observe", "see", "sweat", "understand", "voyage",
]

# For pipeline search queries, strip leading "to " from verb forms
PIPELINE_WORDS = [w for w in RAW_ENGL_WORDS]

# For Helsinki search we also want historical spelling variants
# (simple list — extend if needed)
SPELLING_VARIANTS: dict[str, list[str]] = {
    "bad":       ["bad", "badde"],
    "bead":      ["bead", "bede", "gebed"],
    "before":    ["before", "bifore", "bifor", "beforen"],
    "canine":    ["canine"],
    "car":       ["car", "carre"],
    "corn":      ["corn", "corne"],
    "fast":      ["fast", "faste", "fæst"],
    "hardly":    ["hardly", "hardliche", "hardli"],
    "holiday":   ["holiday", "holidai", "haliday", "holi day", "halig"],
    "horn":      ["horn", "horne"],
    "humour":    ["humour", "humor", "humours"],
    "lure":      ["lure", "luren"],
    "mouse":     ["mouse", "mous", "mus"],
    "must":      ["must", "mot", "moste", "muste"],
    "perfect lady": ["perfect lady"],
    "premises":  ["premises", "premisses"],
    "shall":     ["shall", "shal", "schal", "sceal"],
    "spaghetti": ["spaghetti"],
    "spanish":   ["spanish", "spaynyssh", "spanisshe"],
    "target":    ["target", "targe"],
    "observe":   ["observe", "obserue"],
    "see":       ["see", "seo", "seon", "se"],
    "sweat":     ["sweat", "swete", "swæt"],
    "understand":["understand", "understande", "understonden"],
    "voyage":    ["voyage", "viage", "vyage"],
}

# Maximum sentences to keep per word from Helsinki
MAX_HC_SENTENCES = 10
# Minimum / maximum character length for a valid sentence chunk
MIN_CHUNK = 40
MAX_CHUNK = 800


# ===================================================================
# Helsinki Corpus parser
# ===================================================================
class HelsinkiParser:
    """Parse the Helsinki Corpus master XML file and extract sentences."""

    def __init__(self, xml_path: Path = HC_XML_PATH):
        self.xml_path = xml_path
        self._docs: list[tuple[str, str, list[str]]] = []  # (title, period, [para_texts])
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        log.info("Parsing Helsinki Corpus XML (this may take a moment)…")
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        for tei in root.findall(".//tei:TEI", TEI_NS):
            # Title
            title_el = tei.find(".//tei:titleStmt/tei:title", TEI_NS)
            title = (title_el.text or "").strip() if title_el is not None else ""

            # Period
            date_el = tei.find(".//tei:creation/tei:date", TEI_NS)
            period = ""
            if date_el is not None:
                period = date_el.get("when", date_el.get("notBefore", date_el.text or ""))

            # Paragraph texts
            body = tei.find(".//tei:body", TEI_NS)
            paras: list[str] = []
            if body is not None:
                for p in body.findall(".//tei:p", TEI_NS):
                    raw = ET.tostring(p, encoding="unicode", method="text")
                    # Collapse whitespace
                    text = re.sub(r"\s+", " ", raw).strip()
                    if text:
                        paras.append(text)

            self._docs.append((title, period, paras))

        log.info("  Loaded %d documents, %d total paragraphs",
                 len(self._docs),
                 sum(len(ps) for _, _, ps in self._docs))
        self._loaded = True

    def search(self, word: str) -> list[dict]:
        """
        Search for *word* (and its spelling variants) across all paragraphs.
        Returns list of {"text": ..., "source_title": ..., "period": ...}.
        """
        self._load()
        variants = SPELLING_VARIANTS.get(word, [word])
        # Build a compiled regex: match any variant as a whole word (case-insensitive)
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(v) for v in variants) + r")\b",
            re.IGNORECASE,
        )

        hits: list[dict] = []
        for title, period, paras in self._docs:
            for para in paras:
                if pattern.search(para):
                    # Try to extract just the sentence(s) containing the word
                    chunks = self._extract_chunks(para, pattern)
                    for chunk in chunks:
                        hits.append({
                            "text": chunk,
                            "source_title": title,
                            "period": period,
                        })
        log.info("  Helsinki: %d hit(s) for '%s' (variants: %s)",
                 len(hits), word, variants)
        return hits

    # Common Latin function words that never appear in OE/ME/ModE
    _LATIN_MARKERS = re.compile(
        r"\b(et|est|sunt|erat|eius|quod|quid|quia|quae|qui|quam|"
        r"autem|enim|cum|sed|sic|nec|tamen|aut|vel|nam|ipse|"
        r"ille|illa|illud|haec|hoc|ab|ex|domini|dominum|dominus|"
        r"deus|dei|meus|mea|tuum|tuus|nostrum|eorum|"
        r"fecit|dixit|inquit|inuocabo|laudans)\b",
        re.IGNORECASE,
    )
    _LATIN_THRESHOLD = 0.08  # reject if >8% of words are Latin markers

    @classmethod
    def _is_likely_latin(cls, text: str) -> bool:
        """Return True if *text* looks like it contains significant Latin."""
        words = text.split()
        if not words:
            return False
        latin_count = len(cls._LATIN_MARKERS.findall(text))
        return (latin_count / len(words)) > cls._LATIN_THRESHOLD

    @staticmethod
    def _extract_chunks(paragraph: str, pattern: re.Pattern) -> list[str]:
        """
        Split a paragraph into rough sentence-like chunks around the match.
        The Helsinki Corpus doesn't have <s> tags, so we split on
        common Old/Middle English punctuation (. ; : ·).
        """
        # Split on sentence-ending punctuation
        raw_chunks = re.split(r"(?<=[.;:·!?])\s+", paragraph)
        good: list[str] = []
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not pattern.search(chunk):
                continue
            if not (MIN_CHUNK <= len(chunk) <= MAX_CHUNK):
                continue
            # Skip chunks that look predominantly Latin
            if HelsinkiParser._is_likely_latin(chunk):
                continue
            good.append(chunk)
        # If none of the sub-chunks matched length filters, return a
        # trimmed version of the whole paragraph
        if not good and pattern.search(paragraph):
            trimmed = paragraph[:MAX_CHUNK]
            if len(trimmed) >= MIN_CHUNK and not HelsinkiParser._is_likely_latin(trimmed):
                good.append(trimmed + ("…" if len(paragraph) > MAX_CHUNK else ""))
        return good



# ===================================================================
# Merge logic
# ===================================================================
def merge_helsinki_into_results(
    pipeline_results: list[dict],
    hc_parser: HelsinkiParser,
) -> list[dict]:
    """Add Helsinki Corpus hits to each word's historical_context."""
    for entry in pipeline_results:
        word = entry["word"]
        hc_hits = hc_parser.search(word)

        if hc_hits:
            hc_texts = [h["text"] for h in hc_hits[:MAX_HC_SENTENCES]]
            existing = entry.get("historical_context", [])
            combined = existing + hc_texts

            # Deduplicate
            seen: set[str] = set()
            deduped: list[str] = []
            for t in combined:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)

            entry["historical_context"] = deduped

            # Update source label
            old_src = entry.get("historical_source", "none")
            if old_src and old_src != "none":
                entry["historical_source"] = f"{old_src} + Helsinki Corpus"
            else:
                entry["historical_source"] = "Helsinki Corpus"

    return pipeline_results


# ===================================================================
# main
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
                            # Use English translation of old meaning
                            old_meaning_eng = row[7].strip()
                            test_words.append({
                                "word": clean_word,
                                "target_meaning": old_meaning_eng
                            })
                            seen_words.add(clean_word)
    except Exception as e:
        log.error("Failed to parse LSC dataset to get 25 words: %s", e)
        return

    log.info("=" * 60)
    log.info("Helsinki Wrapper — running pipeline for %d :engl words", len(test_words))
    log.info("=" * 60)

    # 1. Run the standard pipeline (Modern + Wiktionary/MED) with target meanings
    pipeline_results = run_pipeline(test_words)

    # 2. Parse Helsinki Corpus and merge
    hc = HelsinkiParser()
    merged = merge_helsinki_into_results(pipeline_results, hc)

    # 3. Save
    out = save_results(merged, OUTPUT_FILE)

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline complete — results at", out)
    print("=" * 60)
    for entry in merged:
        print(
            f"  {entry['word']:18s}  "
            f"modern={len(entry['modern_context']):2d}  "
            f"historical={len(entry['historical_context']):2d}  "
            f"(source: {entry['historical_source']})"
        )
    print()


if __name__ == "__main__":
    main()
