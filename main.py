"""MAD-SC CLI entry point.

Loads context sentences from the SemEval-2020 Task 1 English corpus for a
target word and runs the tri-agent debate pipeline.

Run
---
    source .venv/bin/activate

    # Single-round parallel debate (original behaviour):
    python main.py [target_word]
    python main.py edge_nn

    # Multi-round rebuttal debate (agents see each other's arguments):
    python main.py [target_word] --mode multi --rounds 3
    python main.py attack_nn --mode multi --rounds 2
"""

import argparse
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
from mad_sc.graph_multi import compile_multi_round_graph
from mad_sc.log_utils import append_debate_log

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAD-SC: Multi-Agent Debate for Semantic Change detection."
    )
    parser.add_argument(
        "word",
        nargs="?",
        default=None,
        help="Target word (e.g. 'attack_nn'). Defaults to first entry in targets.txt.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help=(
            "Debate mode. 'single' (default): both teams argue in parallel, "
            "neither sees the other's argument. "
            "'multi': sequential rebuttal rounds where each team reads and "
            "counters the opponent's latest argument."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Number of rebuttal rounds for --mode multi (default: 3). "
            "Ignored in single mode."
        ),
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--grounding",
        dest="use_grounding",
        action="store_true",
        default=None,
        help="Enable pre-debate BERT grounding.",
    )
    grp.add_argument(
        "--no-grounding",
        dest="use_grounding",
        action="store_false",
        help="Disable pre-debate BERT grounding.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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
    mode = args.mode
    num_rounds = max(1, args.rounds)
    grounding_kwargs = {} if args.use_grounding is None else {"use_grounding": args.use_grounding}

    if mode == "multi":
        graph = compile_multi_round_graph(num_rounds=num_rounds, **grounding_kwargs)
    else:
        graph = compile_graph(**grounding_kwargs)

    # Derive word_type from the trailing _nn / _vb suffix.
    word_type = "verb" if word.endswith("_vb") else "noun"

    initial_state = {
        "word": word,
        "word_type": word_type,
        "t_old": CORPUS1_LABEL,
        "t_new": CORPUS2_LABEL,
        "sentences_old": sentences_old,
        "sentences_new": sentences_new,
        "arg_change": "",
        "arg_stable": "",
        "num_rounds": num_rounds,
        "current_round": 1,
        "debate_history": [],
        "verdict": None,
    }

    mode_label = (
        f"multi-round ({num_rounds} rebuttal round{'s' if num_rounds > 1 else ''})"
        if mode == "multi"
        else "single-round (parallel)"
    )

    print("=" * 65)
    print(f"  MAD-SC  |  word: '{word}'  |  mode: {mode_label}")
    print(f"  {CORPUS1_LABEL}  →  {CORPUS2_LABEL}")
    print(f"  Sentences: {len(sentences_old)} (corpus1) / {len(sentences_new)} (corpus2)")
    print("=" * 65)

    result = graph.invoke(initial_state)

    if mode == "multi":
        # Print each round's arguments separately so the back-and-forth is clear.
        history = result.get("debate_history", [])
        for entry in history:
            r = entry["round"]
            print(f"\n┌─ ROUND {r}/{num_rounds} — Team Support (Arg_change) " + "─" * 25)
            print(entry["arg_change"])
            print(f"\n┌─ ROUND {r}/{num_rounds} — Team Refuse (Arg_stable) " + "─" * 26)
            print(entry["arg_stable"])
    else:
        print("\n┌─ ARG_CHANGE (Team Support) " + "─" * 37)
        print(result["arg_change"])
        print("\n┌─ ARG_STABLE (Team Refuse) " + "─" * 38)
        print(result["arg_stable"])

    print("\n┌─ JUDGE VERDICT " + "─" * 48)
    print(json.dumps(result["verdict"], indent=2))
    print("─" * 65)

    # Persist the full debate trail for later investigation.
    append_debate_log(result, debate_mode=mode + "_round", num_rounds=num_rounds)
    print(f"[main] Debate log updated → debate_logs.json")


if __name__ == "__main__":
    main()
