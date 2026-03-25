"""SemEval-2020 Task 1 English data loader.

Streams the parallel lemma + token .gz files from corpus1 (1810–1860) and
corpus2 (1960–2010) to extract readable context sentences for a target word.

How matching works
------------------
The SemEval lemma files tag *only* the target word with a broad POS suffix
(e.g., ``edge_nn``, ``record_nn``).  All other tokens are plain lemmas without
tags.  Matching is therefore performed on the lemma file, and the corresponding
line from the parallel token file — which contains natural, punctuated prose —
is returned to the agents.

Sampling strategy
-----------------
Sentences are selected via **reservoir sampling** (Algorithm R) so that the
returned sample is uniformly random over all matches in the corpus — not just
the first N chronological hits.  A deterministic seed derived from the target
word is used by default so results are reproducible; pass an explicit ``seed``
to override.

Environment variables
---------------------
SEMEVAL_DIR          Path to the semeval2020_ulscd_eng directory.
                     Default: <project_root>/data/semeval2020_ulscd_eng
SEMEVAL_MAX_SAMPLES  Maximum sentences returned per corpus (default: 10).
"""

from __future__ import annotations

import gzip
import os
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODULE_DIR   = Path(__file__).parent       # mad_sc/
_PROJECT_ROOT = _MODULE_DIR.parent          # MAD-SC/

SEMEVAL_DIR: Path = Path(
    os.getenv("SEMEVAL_DIR",
              str(_PROJECT_ROOT / "data" / "semeval2020_ulscd_eng"))
)

DEFAULT_MAX_SAMPLES: int = int(os.getenv("SEMEVAL_MAX_SAMPLES", "10"))

# Fixed period labels used by the pipeline, UI, and prompts.
CORPUS1_LABEL = "Corpus 1 (1810\u20131860)"
CORPUS2_LABEL = "Corpus 2 (1960\u20132010)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_targets() -> list[str]:
    """Return the SemEval English target words from targets.txt, sorted.

    Words are in the POS-tagged form (e.g., ``edge_nn``, ``circle_vb``).
    Returns an empty list if targets.txt is absent.
    """
    targets_path = SEMEVAL_DIR / "targets.txt"
    if not targets_path.exists():
        return []
    return sorted(
        line.strip()
        for line in targets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def get_semeval_contexts(
    target_word: str,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    seed: int | None = None,
) -> tuple[list[str], list[str]]:
    """Retrieve context sentences for *target_word* from both SemEval corpora.

    Sentences are drawn via reservoir sampling so the returned set is a
    uniform random sample of *all* matching sentences — not just the first N.

    Parameters
    ----------
    target_word : str
        Target word in the POS-tagged form used in targets.txt (e.g.,
        ``edge_nn``, ``circle_vb``).  A bare lemma (e.g., ``edge``) is also
        accepted and will match any POS variant present in the lemma files.
    max_samples : int
        Maximum sentences to return per corpus.
    seed : int | None
        RNG seed for reservoir sampling.  When *None* (default) a
        deterministic seed derived from *target_word* is used so repeated
        calls return the same sample.  Pass a different integer to vary it.

    Returns
    -------
    (sentences_c1, sentences_c2) : tuple[list[str], list[str]]
        Readable token-file sentences from corpus1 and corpus2 respectively.
        Returns ``([], [])`` with a printed warning if the data directory or
        the expected .gz files are missing.
    """
    if not SEMEVAL_DIR.exists():
        print(
            f"[data_loader] WARNING: SemEval directory '{SEMEVAL_DIR}' not found. "
            "Place the semeval2020_ulscd_eng folder under data/ and retry."
        )
        return [], []

    c1_lemma = SEMEVAL_DIR / "corpus1" / "lemma" / "ccoha1.txt.gz"
    c1_token  = SEMEVAL_DIR / "corpus1" / "token"  / "ccoha1.txt.gz"
    c2_lemma = SEMEVAL_DIR / "corpus2" / "lemma" / "ccoha2.txt.gz"
    c2_token  = SEMEVAL_DIR / "corpus2" / "token"  / "ccoha2.txt.gz"

    for path in (c1_lemma, c1_token, c2_lemma, c2_token):
        if not path.exists():
            print(f"[data_loader] WARNING: Expected file not found: {path}")
            return [], []

    # Derive a reproducible per-word seed when the caller does not supply one.
    effective_seed = seed if seed is not None else (hash(target_word) % (2 ** 31))

    pattern = _build_pattern(target_word)
    sentences_c1 = _extract_sentences(c1_lemma, c1_token, pattern, max_samples, effective_seed)
    sentences_c2 = _extract_sentences(c2_lemma, c2_token, pattern, max_samples, effective_seed + 1)
    return sentences_c1, sentences_c2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_pattern(target_word: str) -> re.Pattern:
    """Compile a whole-token regex for *target_word* against the lemma file.

    The lemma file is space-separated with no punctuation.  Target words are
    stored as ``word_pos`` (e.g., ``edge_nn``).  We use ``\\b`` word-boundary
    anchors: because ``_`` is a ``\\w`` character, ``\\bedge_nn\\b`` correctly
    matches the full token ``edge_nn`` but not ``edge`` or ``edge_nnn``.

    If *target_word* has no POS suffix (no trailing ``_<letters>``), the
    pattern is broadened to match any POS variant (``edge_nn``, ``edge_vb``, …).
    """
    word = target_word.strip()
    if re.search(r"_[a-z]+$", word):
        # Fully qualified form: match exactly (e.g., edge_nn)
        return re.compile(rf"\b{re.escape(word)}\b")
    else:
        # Bare lemma: match word followed by any POS suffix
        return re.compile(rf"\b{re.escape(word)}_[a-z]+\b")


def _extract_sentences(
    lemma_gz: Path,
    token_gz: Path,
    pattern: re.Pattern,
    max_samples: int,
    seed: int = 0,
) -> list[str]:
    """Stream lemma and token gz files using reservoir sampling (Algorithm R).

    Unlike a simple first-N scan, reservoir sampling gives every matching
    sentence an equal probability of appearing in the returned set, regardless
    of where it sits in the corpus.  This avoids the systematic bias of
    returning only early-corpus (often older-style) sentences even when the
    corpus file itself spans the full period.

    Time complexity: O(n) where n = total matching sentences.
    Space complexity: O(max_samples).
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    count = 0

    with (
        gzip.open(lemma_gz, "rt", encoding="utf-8", errors="ignore") as lf,
        gzip.open(token_gz, "rt", encoding="utf-8", errors="ignore") as tf,
    ):
        for lemma_line, token_line in zip(lf, tf):
            if pattern.search(lemma_line):
                sentence = token_line.strip()
                if not sentence:
                    continue
                count += 1
                if len(reservoir) < max_samples:
                    reservoir.append(sentence)
                else:
                    # Replace a random slot with decreasing probability.
                    j = rng.randint(0, count - 1)
                    if j < max_samples:
                        reservoir[j] = sentence

    return reservoir
