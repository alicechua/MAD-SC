#!/usr/bin/env python3
"""
Enrich lsc_context_data_engl.json with OED historical quotations.

For each failing word where OED returns >=5 pre-1900 quotes, replaces
historical_context with a sampled spread of OED sentences.
Words with <5 OED quotes keep their existing (manually patched) data.
"""
import json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lsc_data_pipeline import OEDScraper, OED_COOKIE_FILE

DATA_FILE = ROOT / "data" / "lsc_context_data_engl.json"

# Slug overrides for words where default resolution hits wrong POS
SLUG_OVERRIDES: dict[str, str] = {
    "target":     "target_n1",
    "understand": "understand_v",
}

# Words to enrich with old meaning hint for quality filtering
FAILING_WORDS: dict[str, str] = {
    "before":       "in front of, spatial position",
    "see":          "physical sight, perceive with eyes",
    "voyage":       "travel, journey",
    "bad":          "evil, wicked, not good",
    "holiday":      "holy day, religious festival",
    "horn":         "animal horn, bony projection",
    "must":         "obligation, deontic modal auxiliary",
    "observe":      "worship, honor, revere, keep a custom",
    "shall":        "obligation, should, deontic future",
    "spanish":      "relating to Spain or Spanish people",
    "target":       "small round shield, targe",
    "understand":   "grasp meaning, comprehend",
    "car":          "cart, chariot, wheeled vehicle",
    "fast":         "solid, fixed, firmly attached",
    "hardly":       "boldly, firmly, vigorously",
    "perfect lady": "noble, completed lady",
}

MIN_QUOTES = 5   # minimum OED quotes to bother replacing
TARGET_N   = 12  # how many to keep after sampling


def sample_spread(quotes: list[str], n: int) -> list[str]:
    """Pick n evenly-spaced items from quotes (chronological order from OED)."""
    if len(quotes) <= n:
        return quotes
    step = len(quotes) / n
    return [quotes[int(i * step)] for i in range(n)]


def fetch_oed_quotes(oed: OEDScraper, word: str) -> list[str]:
    """Fetch historical OED quotes, respecting slug overrides."""
    override = SLUG_OVERRIDES.get(word)
    if override:
        # Temporarily patch the scraper to use the override URL directly
        url = f"https://www.oed.com{'' if override.startswith('/') else '/dictionary/'}{override}"
        time.sleep(oed.polite_delay)
        quotes = oed._fetch_quotes(url, max_year=1899)
        return quotes
    quotes, _ = oed.collect_historical(word, before_year=1900)
    return quotes


def main():
    oed = OEDScraper(OED_COOKIE_FILE)

    with open(DATA_FILE, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    entry_map = {e["word"]: e for e in data}

    print(f"{'Word':<16} {'OED quotes':>10}  {'Action'}")
    print("-" * 50)

    for word, _meaning in FAILING_WORDS.items():
        if word not in entry_map:
            print(f"{word:<16} {'N/A':>10}  MISSING from JSON — skipped")
            continue

        oed_quotes = fetch_oed_quotes(oed, word)
        time.sleep(0.5)

        if len(oed_quotes) < MIN_QUOTES:
            print(f"{word:<16} {len(oed_quotes):>10}  KEPT existing (too few OED quotes)")
            continue

        sampled = sample_spread(oed_quotes, TARGET_N)
        entry_map[word]["historical_context"] = sampled
        entry_map[word]["historical_source"] = "OED"
        print(f"{word:<16} {len(oed_quotes):>10}  REPLACED → {len(sampled)} sampled quotes")

    # Write back
    out = list(entry_map.values())
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {DATA_FILE}")


if __name__ == "__main__":
    main()
