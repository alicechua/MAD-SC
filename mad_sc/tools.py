"""External lookup tools for MAD-SC team agents.

These tools are bound to the team LLMs via .bind_tools() and invoked
dynamically during the debate when agents need external evidence.
"""

import urllib.parse

import nltk
import requests
from langchain_core.tools import tool

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

_WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
_WIKIPEDIA_SEARCH_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query&list=search&srsearch={}&srlimit=1&format=json"
)
_NGRAM_URL = (
    "https://books.google.com/ngrams/json"
    "?content={}&year_start={}&year_end={}&corpus=en-2019&smoothing=3"
)


@tool
def wikipedia_search(query: str) -> str:
    """Look up a Wikipedia summary for a query.

    Use this tool to retrieve historical context about cultural movements,
    technological inventions, or social events that may have driven a word's
    semantic change (e.g. 'history of streaming media', 'LGBT rights movement').

    Args:
        query: A descriptive search phrase or article title.

    Returns:
        A 1-3 paragraph summary from Wikipedia, or an error message.
    """
    # Try direct title lookup first
    encoded = urllib.parse.quote(query.replace(" ", "_"))
    try:
        resp = requests.get(
            _WIKIPEDIA_SUMMARY_URL.format(encoded),
            timeout=8,
            headers={"User-Agent": "MAD-SC/1.0 (semantic-change-research)"},
        )
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract", "").strip()
            if extract:
                title = data.get("title", query)
                return f"[Wikipedia: {title}]\n{extract}"
    except requests.RequestException:
        pass

    # Fall back to search API
    try:
        search_encoded = urllib.parse.quote(query)
        resp = requests.get(
            _WIKIPEDIA_SEARCH_URL.format(search_encoded),
            timeout=8,
            headers={"User-Agent": "MAD-SC/1.0 (semantic-change-research)"},
        )
        if resp.status_code == 200:
            results = resp.json().get("query", {}).get("search", [])
            if results:
                top_title = results[0]["title"]
                encoded2 = urllib.parse.quote(top_title.replace(" ", "_"))
                resp2 = requests.get(
                    _WIKIPEDIA_SUMMARY_URL.format(encoded2),
                    timeout=8,
                    headers={"User-Agent": "MAD-SC/1.0 (semantic-change-research)"},
                )
                if resp2.status_code == 200:
                    data2 = resp2.json()
                    extract2 = data2.get("extract", "").strip()
                    if extract2:
                        return f"[Wikipedia: {top_title}]\n{extract2}"
    except requests.RequestException:
        pass

    return f"Wikipedia: no result found for '{query}'."


@tool
def wordnet_query(word: str, pos: str = "n") -> str:
    """Query WordNet for synsets, definitions, hypernyms, and hyponyms.

    Use this tool to prove or disprove a broadening (Generalization) or
    narrowing (Specialization) change type. Hypernyms are broader/parent
    categories; hyponyms are narrower/child categories.

    Args:
        word: The target word to look up.
        pos:  Part of speech — "n" (noun), "v" (verb), "a" (adjective),
              "r" (adverb). Defaults to "n".

    Returns:
        A formatted string listing synsets, definitions, hypernym chains,
        and direct hyponyms for the word.
    """
    from nltk.corpus import wordnet as wn

    pos_map = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ, "r": wn.ADV}
    wn_pos = pos_map.get(pos.lower(), wn.NOUN)
    synsets = wn.synsets(word, pos=wn_pos)

    if not synsets:
        # Try without POS filter
        synsets = wn.synsets(word)
        if not synsets:
            return f"WordNet: no synsets found for '{word}'."

    lines = [f"WordNet results for '{word}' (pos={pos}):"]
    for ss in synsets[:4]:  # cap at 4 synsets
        lines.append(f"\nSynset: {ss.name()}")
        lines.append(f"  Definition: {ss.definition()}")

        # Hypernym chain (up to 3 levels up)
        hypernyms = ss.hypernyms()
        if hypernyms:
            chain = []
            current = hypernyms[0]
            for _ in range(3):
                chain.append(current.lemma_names()[0].replace("_", " "))
                parents = current.hypernyms()
                if not parents:
                    break
                current = parents[0]
            lines.append(f"  Hypernym chain: {' → '.join(chain)}")

        # Direct hyponyms (up to 6)
        hyponyms = ss.hyponyms()
        if hyponyms:
            hypo_names = [h.lemma_names()[0].replace("_", " ") for h in hyponyms[:6]]
            lines.append(f"  Hyponyms: {', '.join(hypo_names)}")

    return "\n".join(lines)


@tool
def ngram_frequency(word: str, start_year: int = 1900, end_year: int = 2019) -> str:
    """Retrieve Google Books Ngram frequency data for a word over a time range.

    Use this tool to show whether a word's usage spiked, declined, or remained
    stable during a period. A sharp spike after a specific year supports the
    claim that a new sense was adopted by the speech community.

    Args:
        word:       The word or phrase to look up.
        start_year: Start of the time window (default 1900).
        end_year:   End of the time window (default 2019).

    Returns:
        A human-readable summary of frequency trends: peak decade, direction,
        and notable spikes. Returns an error message if the API is unavailable.
    """
    url = _NGRAM_URL.format(urllib.parse.quote(word), start_year, end_year)
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "MAD-SC/1.0 (semantic-change-research)"},
        )
        if resp.status_code != 200:
            return f"Ngram API returned status {resp.status_code} for '{word}'."

        data = resp.json()
        if not data:
            return f"Ngram: no data found for '{word}' ({start_year}–{end_year})."

        series = data[0]
        timeseries = series.get("timeseries", [])
        ngram_word = series.get("ngram", word)
        years = list(range(start_year, end_year + 1))

        if not timeseries or len(timeseries) != len(years):
            return f"Ngram: unexpected data format for '{word}'."

        # Build summary
        paired = list(zip(years, timeseries))
        max_val = max(timeseries)
        min_val = min(timeseries)
        peak_year = paired[timeseries.index(max_val)][0]

        # Decade-level averages
        decade_avgs = {}
        for yr, freq in paired:
            decade = (yr // 10) * 10
            decade_avgs.setdefault(decade, []).append(freq)
        decade_summary = {
            d: sum(v) / len(v) for d, v in sorted(decade_avgs.items())
        }

        # Find biggest single-year jump
        jumps = [
            (years[i], timeseries[i] - timeseries[i - 1])
            for i in range(1, len(timeseries))
        ]
        big_jump = max(jumps, key=lambda x: x[1])

        lines = [
            f"Google Books Ngram frequency for '{ngram_word}' ({start_year}–{end_year}):",
            f"  Peak frequency: {max_val:.6f} in {peak_year}",
            f"  Min  frequency: {min_val:.6f}",
            f"  Largest single-year increase: {big_jump[1]:+.6f} in {big_jump[0]}",
            "  Decade averages:",
        ]
        for decade, avg in decade_summary.items():
            lines.append(f"    {decade}s: {avg:.6f}")

        return "\n".join(lines)

    except requests.RequestException as e:
        return f"Ngram API unavailable for '{word}': {e}"
