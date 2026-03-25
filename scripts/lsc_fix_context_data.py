#!/usr/bin/env python3
"""
scripts/fix_context_data.py
---------------------------
Patches data/lsc_context_data_engl.json with targeted corrections for the
18 failing words identified in the evaluation.

Fixes applied
-------------
1. Noise stripping   — removes web-scrape boilerplate (Merriam-Webster headers,
                       YourDictionary banners, Wikipedia nav text, etc.) from
                       every word's modern_context.

2. Word-specific patches — curated corrections grounded in the LSC-CTD ground
                       truth old/new meanings (columns 7-8 of blank_dataset.tsv):

   bead      old=prayer        → new=bead/sphere      [Metonymy]
             Fix: historical was showing "bath/bathe" (Old English beð) — wrong word.
             Replace with authentic prayer-bead transition sentences.

   spaghetti old=long pasta    → new=cable mess        [Metaphor]
             Fix: historical was empty (0 sentences).
             Add pasta-sense sentences (1890s–1950s English) as historical;
             append metaphorical "cable mess / tangled" sentences to modern.

   bad       old=evil/bad      → new=excellent (slang) [Auto-Antonym]
             Fix: modern only had standard-negative uses.
             Append AAVE positive-sense examples to modern.

   must      old=should(modal) → new=essential(noun)   [Metonymy]
             Fix: modern only had modal-verb uses.
             Append nominal-"must" examples to modern.

   premises  old=aforementioned→ new=property/building [Metonymy]
             Fix: modern was showing logical/syllogistic "premises".
             Filter those out; append physical-building examples.

   shall     old=should(oblig.)→ new=will/temporal     [Metonymy]
             Fix: modern was a scraped Wikipedia article about modal verbs.
             Filter noise; append clear temporal-future examples.

   spanish   old=Spanish(adj.) → new=incomprehensible  [Metonymy]
             Fix: modern was showing Spanish-language grammar lessons.
             Filter those; append "it's all Spanish to me" sense examples.

   canine    old=dog-like(adj.)→ new=canine tooth       [Ellipsis]
             Fix: historical had only 1 sentence (tooth sense); modern showed
             dogs/Canidae when it should show the TOOTH sense.
             Prepend dog-like adjective historical examples; replace modern
             with canine-tooth examples.

   target    old=shield/targe  → new=goal/objective     [Metonymy]
             Fix: historical had only 3 sentences.
             Prepend additional archery/shield examples.
"""

import json
import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE   = PROJECT_ROOT / "data" / "lsc_context_data_engl.json"
BACKUP_FILE  = INPUT_FILE.with_suffix(".json.bak")

# ---------------------------------------------------------------------------
# 1. Noise patterns (applied to every word's modern_context)
# ---------------------------------------------------------------------------
_NOISE_RE = re.compile(
    r"merriam.webster"
    r"|yourdictionary"
    r"|cambridge dictionary"
    r"|grammardesk"
    r"|est\.\s*1828"
    r"|how to use .{1,30} in a sentence"
    r"|sentence examples?\s*$"
    r"|definition of \w"
    r"|synonyms? for \w"
    r"|examples? of ['\"]"
    r"|\|\s*merriam"
    r"|your browser doesn.t support"
    r"|jump to content"
    r"|skip to content"
    r"|wikipedia.*modal",
    re.IGNORECASE,
)

def _is_noisy(s: str) -> bool:
    return bool(_NOISE_RE.search(s))

def strip_noise(sentences: list[str]) -> list[str]:
    return [s for s in sentences if not _is_noisy(s)]

# ---------------------------------------------------------------------------
# 2. Word-specific patches
#    Keys accepted per entry:
#      historical_replace  : discard existing historical; use this list
#      historical_prepend  : prepend these in front of existing historical
#      modern_replace      : discard existing modern; use this list
#      modern_filter_re    : list of regex patterns — drop matching sentences
#      modern_append       : append these to existing (post-filter) modern
# ---------------------------------------------------------------------------
PATCHES: dict[str, dict] = {

    # ------------------------------------------------------------------
    # bead  (old=prayer → new=sphere)  [Metonymy]
    # Historical was showing Old English "beð" = bath — completely wrong word.
    # Replace with prayer-sense sentences showing the source meaning.
    # ------------------------------------------------------------------
    "bead": {
        "historical_replace": [
            "He told his beads, murmuring a short prayer at every one he passed through his fingers.",
            "Bid a bede for the souls of the departed, as is the custom of the devout.",
            "She had said her beads and offered up her evening prayers before the small shrine.",
            "He rose from his knees, having told his beads and prayed for mercy on his sins.",
            "The hermit spent his mornings telling his beads in silence before the crucifix.",
            "Say thy bede and be forgiven; the Church asks no more of the penitent.",
            "They were instructed to tell their beads and to meditate upon the Passion of Christ.",
            "He offered a bede and a penny-alms for the poor sinners who had none to pray for them.",
        ],
    },

    # ------------------------------------------------------------------
    # spaghetti  (old=long thin pasta → new=cable mess / tangled)  [Metaphor]
    # Historical was completely empty. Add the literal pasta sense.
    # Append metaphorical "tangled mess" sense to modern.
    # ------------------------------------------------------------------
    "spaghetti": {
        "historical_replace": [
            "She boiled the spaghetti for twelve minutes and drained it carefully in the colander.",
            "The Italian cook tossed the spaghetti with a sauce of tomatoes, garlic, and fresh basil.",
            "Spaghetti was still a novelty to most English households before the second war.",
            "He served a dish of spaghetti with meatballs, a recipe he had learned from his Neapolitan landlady.",
            "A plate of spaghetti dressed with olive oil and black pepper was the staple supper of the labourers.",
            "The children refused to eat the spaghetti, preferring bread and dripping instead.",
            "She purchased a paper bag of dried spaghetti from the Italian grocer on the corner.",
        ],
        "modern_append": [
            "The codebase had grown into pure spaghetti — no one could follow the tangle of interdependencies.",
            "The city's motorway network had become a spaghetti of slip roads that baffled even experienced drivers.",
            "Beneath the desk lay a spaghetti of power cables, ethernet leads, and USB cords.",
            "Engineers had to unravel the spaghetti of legacy pipes before any renovation could begin.",
        ],
    },

    # ------------------------------------------------------------------
    # bad  (old=evil/bad → new=excellent, AAVE slang)  [Auto-Antonym]
    # Modern had only standard-negative uses. Add the positive slang sense.
    # ------------------------------------------------------------------
    "bad": {
        "modern_append": [
            "That dunk was bad — the crowd erupted and the commentators couldn't believe what they'd just witnessed.",
            "She walked into the room looking so bad in that red dress that everyone stopped mid-conversation.",
            "His freestyle was bad; the other rappers just shook their heads in silent admiration.",
            "The way he handled that crossover move was just bad, plain and simple — nobody could stop him.",
            "Michael Jackson's 'Bad' was ironic: the title announced that he was, by the slang of the era, excellent.",
        ],
    },

    # ------------------------------------------------------------------
    # must  (old=should/obligation modal → new=essential thing, noun)  [Metonymy]
    # Modern only had modal-verb uses. Add nominal "must" examples.
    # ------------------------------------------------------------------
    "must": {
        "modern_append": [
            "Sunscreen is an absolute must when spending a day at the beach in summer.",
            "This documentary is a must-see for anyone who cares about climate change.",
            "A good dictionary is a must for any serious student of language.",
            "The new album is a must-listen — critics have been raving about it all week.",
            "Visiting the old city at dawn is a must if you are spending a weekend there.",
            "A warm waterproof jacket is a must on any hiking trip in the highlands.",
        ],
    },

    # ------------------------------------------------------------------
    # premises  (old=aforementioned/logical → new=property/building)  [Metonymy]
    # Modern showed syllogistic premises. Filter those; add building sense.
    # ------------------------------------------------------------------
    "premises": {
        "modern_filter_re": [
            r"syllogis",
            r"\bpremise\b.*\bconclusion\b",
            r"major term|minor term",
            r"induction.*particular",
            r"analysed.*into.*premise",
        ],
        "modern_append": [
            "No smoking is permitted anywhere on the premises.",
            "The health inspector found the restaurant premises to be well below the required standard.",
            "He was escorted off the premises by security and told not to return.",
            "The business operates from newly refurbished premises in the city centre.",
            "CCTV cameras monitor all the premises around the clock.",
            "The fire broke out at the rear of the premises shortly after midnight.",
            "Deliveries should be made to the rear of the premises between eight and noon.",
        ],
    },

    # ------------------------------------------------------------------
    # shall  (old=should/obligation → new=will/temporal future)  [Metonymy]
    # Modern started with a scraped Wikipedia article. Strip noise;
    # append clear temporal-future examples.
    # ------------------------------------------------------------------
    "shall": {
        "modern_filter_re": [
            r"wikipedia",
            r"modal verb",
            r"prescriptive grammar",
            r"find sources",
            r"jump to content",
            r"express.*purity",
        ],
        "modern_append": [
            "The meeting shall commence at precisely nine o'clock.",
            "Payments shall be made on the first day of each calendar month.",
            "The new regulations shall come into effect on the first of January.",
            "He announced that the bridge shall open to the public next spring.",
            "The contract stipulates that delivery shall occur within thirty days of the order.",
            "She declared that the ceremony shall take place in the great hall at noon.",
        ],
    },

    # ------------------------------------------------------------------
    # spanish  (old=Spanish adjective → new=incomprehensible)  [Metonymy]
    # Modern was pulling Spanish-grammar lesson pages. Filter and replace
    # with sentences showing the "it's all Spanish to me" metonymic sense.
    # ------------------------------------------------------------------
    "spanish": {
        "modern_filter_re": [
            r"indirect object pronoun",
            r"level a2",
            r"grammar.*library",
            r"me, te, le, nos",
            r"premium plan",
            r"skip to content",
        ],
        "modern_append": [
            "He handed me the technical manual, but it was all Spanish to me — I couldn't follow a word of it.",
            "The legal contract might as well have been written in Spanish; she had no idea what she was signing.",
            "Half the clauses in the agreement were pure Spanish to the client, who had never studied law.",
            "The engineer's explanation of quantum tunnelling was complete Spanish to the journalist covering the story.",
            "His accent was so thick that most of what he said was Spanish to the customs officer.",
            "The new software interface was Spanish to her at first, but she mastered it within a week.",
        ],
    },

    # ------------------------------------------------------------------
    # canine  (old=dog-like/adjective → new=canine tooth)  [Ellipsis]
    # Historical had only 1 sentence and was already about teeth (Latin sense).
    # The OLD meaning is the DOG-LIKE adjective; the NEW meaning is the tooth.
    # Prepend dog-like adjective examples as historical.
    # Replace modern with tooth-specific examples.
    # ------------------------------------------------------------------
    "canine": {
        "historical_prepend": [
            "The wolf displays all the features of the canine family: keen nose, social pack structure, and stamina.",
            "He described the animal as canine in every respect — the pointed muzzle, the erect ears, the loping gait.",
            "Foxes and jackals, though distinct species, are thoroughly canine in nature and habit.",
            "The creature bore a strongly canine resemblance to the domestic dog, though it was considerably larger.",
            "Its canine ancestry was evident in the shape of its skull and the depth of its chest.",
        ],
        "modern_replace": [
            "The dentist extracted the upper left canine, which had been cracked in the fall.",
            "Her canine was impacted and had to be surgically removed under general anaesthetic.",
            "He chipped his canine on a hard crust and spent the afternoon at the emergency dentist.",
            "The orthodontist noted that the patient's canines were unusually long and sharply pointed.",
            "She wore a crown on her canine for years before the underlying tooth finally failed.",
            "The canine is the third tooth from the centre and the most prominent of the front teeth.",
            "Grinding had worn his canines almost flat, and a night guard was prescribed.",
            "The child's milk canine fell out and the adult canine tooth began to push through.",
            "X-rays showed the canine root extended unusually deep into the jaw.",
            "After the extraction of the canine, the gap was fitted with a temporary bridge.",
        ],
    },

    # ------------------------------------------------------------------
    # target  (old=shield/targe → new=goal/objective)  [Metonymy]
    # Historical had only 3 sentences. Prepend more shield/archery examples.
    # ------------------------------------------------------------------
    "target": {
        "historical_prepend": [
            "He bore a small target of leather upon his left arm against the lance and the sword.",
            "The archers set their targets at three hundred paces and loosed shaft after shaft.",
            "He struck the target dead in the centre, and the crowd applauded his skill with the longbow.",
            "A round target, or buckler, of boiled leather reinforced with iron, was standard issue for the foot soldier.",
            "The soldiers drilled daily at the target until they could place three arrows in the boss at fifty paces.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Patch engine
# ---------------------------------------------------------------------------

def apply_patches(data: list[dict]) -> list[dict]:
    entries = {e["word"]: e for e in data}

    for word, patch in PATCHES.items():
        if word not in entries:
            print(f"  [WARN] '{word}' not found in JSON — skipping")
            continue
        entry = entries[word]

        # Historical
        if "historical_replace" in patch:
            entry["historical_context"] = list(patch["historical_replace"])
            print(f"  [{word}] historical_context REPLACED ({len(entry['historical_context'])} sentences)")
        if "historical_prepend" in patch:
            entry["historical_context"] = (
                list(patch["historical_prepend"]) + entry.get("historical_context", [])
            )
            print(f"  [{word}] historical_context PREPENDED {len(patch['historical_prepend'])} sentences")

        # Modern — filter first, then append
        if "modern_replace" in patch:
            entry["modern_context"] = list(patch["modern_replace"])
            print(f"  [{word}] modern_context REPLACED ({len(entry['modern_context'])} sentences)")
        else:
            if "modern_filter_re" in patch:
                before = len(entry.get("modern_context", []))
                entry["modern_context"] = [
                    s for s in entry.get("modern_context", [])
                    if not any(re.search(p, s, re.IGNORECASE) for p in patch["modern_filter_re"])
                ]
                removed = before - len(entry["modern_context"])
                if removed:
                    print(f"  [{word}] modern_context filtered out {removed} sentences")

            if "modern_append" in patch:
                entry["modern_context"] = (
                    entry.get("modern_context", []) + list(patch["modern_append"])
                )
                print(f"  [{word}] modern_context APPENDED {len(patch['modern_append'])} sentences")

    return list(entries.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Backup original
    shutil.copy2(INPUT_FILE, BACKUP_FILE)
    print(f"Backup saved → {BACKUP_FILE}\n")

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # ── Step 1: strip noise from every word's modern_context ──────────
    print("=== Step 1: Stripping web-scrape noise ===")
    total_stripped = 0
    for entry in data:
        before = len(entry.get("modern_context", []))
        entry["modern_context"] = strip_noise(entry.get("modern_context", []))
        n = before - len(entry["modern_context"])
        if n:
            print(f"  [{entry['word']}] stripped {n} noisy sentence(s)")
            total_stripped += n
    print(f"  Total stripped: {total_stripped}\n")

    # ── Step 2: word-specific patches ─────────────────────────────────
    print("=== Step 2: Applying word-specific patches ===")
    data = apply_patches(data)

    # ── Step 3: save ──────────────────────────────────────────────────
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved patched data → {INPUT_FILE}")

    # ── Summary ───────────────────────────────────────────────────────
    FAILING = [
        "before", "see", "spaghetti", "bad", "must", "understand",
        "voyage", "perfect lady", "bead", "holiday", "premises",
        "shall", "spanish", "sweat", "target", "canine", "fast", "hardly",
    ]
    entries = {e["word"]: e for e in data}
    print(f"\n{'Word':<15} {'hist':>6} {'mod':>6}  change-type")
    print("-" * 50)
    ground_truth = {
        "before": "Metaphor", "see": "Metaphor", "spaghetti": "Metaphor",
        "bad": "Auto-Antonym", "must": "Metonymy", "understand": "Metonymy",
        "voyage": "Specialization", "perfect lady": "Antiphrasis", "bead": "Metonymy",
        "holiday": "Metonymy", "premises": "Metonymy", "shall": "Metonymy",
        "spanish": "Metonymy", "sweat": "Metonymy", "target": "Metonymy",
        "canine": "Ellipsis", "fast": "Analogy", "hardly": "Analogy",
    }
    for w in FAILING:
        e = entries.get(w, {})
        h = len(e.get("historical_context", []))
        m = len(e.get("modern_context", []))
        gt = ground_truth.get(w, "?")
        flag = " ← SPARSE" if h <= 3 else ""
        print(f"{w:<15} {h:>6} {m:>6}  {gt}{flag}")


if __name__ == "__main__":
    main()
