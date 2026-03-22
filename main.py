"""MAD-SC CLI entry point.

Loads context sentences from the SemEval-2020 Task 1 English corpus for a
target word and runs the tri-agent debate pipeline.

Run
---
    source .venv/bin/activate
    python main.py [target_word]

    # Examples:
    python main.py edge_nn
    python main.py record_nn
    python main.py           # defaults to the first word in targets.txt
"""

import json
import sys

from dotenv import load_dotenv

from mad_sc.data_loader import (
    CORPUS1_LABEL,
    CORPUS2_LABEL,
    get_semeval_contexts,
    get_targets,
)
from mad_sc.graph import compile_graph

load_dotenv()


def main() -> None:
    # ------------------------------------------------------------------
    # Pick target word and flags from CLI args
    # ------------------------------------------------------------------
    import argparse
    parser = argparse.ArgumentParser(description="MAD-SC CLI demo")
    parser.add_argument("word", nargs="?", default=None, help="Target word (e.g. edge_nn)")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--grounding", dest="use_grounding", action="store_true", default=None,
                     help="Enable pre-debate BERT grounding")
    grp.add_argument("--no-grounding", dest="use_grounding", action="store_false",
                     help="Disable pre-debate BERT grounding")
    args = parser.parse_args()

    targets = get_targets()
    if args.word:
        word = args.word
    elif targets:
        word = targets[0]
    else:
        word = "edge_nn"

    # ------------------------------------------------------------------
    # Load SemEval context sentences
    # ------------------------------------------------------------------
    sentences_old, sentences_new = get_semeval_contexts(word)

    if not sentences_old and not sentences_new:
        print(
            "[main] No corpus sentences found for this word. "
            "Check that data/semeval2020_ulscd_eng/ exists and the word "
            "appears in the corpus.\n"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build and invoke the LangGraph pipeline
    # ------------------------------------------------------------------
    grounding_kwargs = {} if args.use_grounding is None else {"use_grounding": args.use_grounding}
    graph = compile_graph(**grounding_kwargs)

    initial_state = {
        "word": word,
        "t_old": CORPUS1_LABEL,
        "t_new": CORPUS2_LABEL,
        "sentences_old": sentences_old,
        "sentences_new": sentences_new,
        "arg_change": "",
        "arg_stable": "",
        "verdict": None,
    }

    print("=" * 65)
    print(f"  MAD-SC  |  word: '{word}'")
    print(f"  {CORPUS1_LABEL}  →  {CORPUS2_LABEL}")
    print(f"  Sentences: {len(sentences_old)} (corpus1) / {len(sentences_new)} (corpus2)")
    print("=" * 65)

    result = graph.invoke(initial_state)

    print("\n┌─ ARG_CHANGE (Team Support) " + "─" * 37)
    print(result["arg_change"])

    print("\n┌─ ARG_STABLE (Team Refuse) " + "─" * 38)
    print(result["arg_stable"])

    print("\n┌─ JUDGE VERDICT " + "─" * 48)
    print(json.dumps(result["verdict"], indent=2))
    print("─" * 65)


if __name__ == "__main__":
    main()
