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

Environment variables
---------------------
SEMEVAL_DIR          Path to the semeval2020_ulscd_eng directory.
                     Default: <project_root>/data/semeval2020_ulscd_eng
SEMEVAL_MAX_SAMPLES  Maximum sentences returned per corpus (default: 10).
"""

from __future__ import annotations

import gzip
import os
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
) -> tuple[list[str], list[str]]:
    """Retrieve context sentences for *target_word* from both SemEval corpora.

    Searches the lemma files for the target word and returns the corresponding
    readable sentences from the parallel token files.

    Parameters
    ----------
    target_word : str
        Target word in the POS-tagged form used in targets.txt (e.g.,
        ``edge_nn``, ``circle_vb``).  A bare lemma (e.g., ``edge``) is also
        accepted and will match any POS variant present in the lemma files.
    max_samples : int
        Maximum sentences to return per corpus.

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

    pattern = _build_pattern(target_word)
    sentences_c1 = _extract_sentences(c1_lemma, c1_token, pattern, max_samples)
    sentences_c2 = _extract_sentences(c2_lemma, c2_token, pattern, max_samples)
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
) -> list[str]:
    """Stream lemma and token gz files line-by-line in parallel.

    When a lemma line matches *pattern*, the corresponding token-file line
    (readable prose) is collected.  Stops once *max_samples* are gathered.
    """
    results: list[str] = []
    with (
        gzip.open(lemma_gz, "rt", encoding="utf-8", errors="ignore") as lf,
        gzip.open(token_gz, "rt", encoding="utf-8", errors="ignore") as tf,
    ):
        for lemma_line, token_line in zip(lf, tf):
            if pattern.search(lemma_line):
                sentence = token_line.strip()
                if sentence:
                    results.append(sentence)
                    if len(results) >= max_samples:
                        break
    return results
