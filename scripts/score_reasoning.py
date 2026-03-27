#!/usr/bin/env python3
"""
LLM-as-Judge Reasoning Scorer
==============================
Reads saved trace files and asks the LLM to grade each trace on four rubric
axes, independently of whether the final verdict was correct.

Rubric axes (0–5 each, applied to each team argument AND the judge reasoning):
  • specificity       – Are concrete corpus examples quoted (with text + year)?
  • temporal_relevance – Does the argument reference diachronic patterns
                         (differences between old and new periods)?
  • logical_validity  – Is the conclusion consistent with the cited evidence?
  • taxonomy_adherence – Does the prediction use Blank's taxonomy correctly?

The LLM returns a JSON object for each component (support, refuse, judge).

Usage
-----
    # Score traces in the default eval_results/traces/ directory
    python scripts/score_reasoning.py

    # Point at a specific run
    python scripts/score_reasoning.py --traces-dir evals/eval_results_lexicographer_full/traces

    # Limit to specific words (useful for spot-checking)
    python scripts/score_reasoning.py --words mouse corn horn

    # Save full report to JSON
    python scripts/score_reasoning.py --output reasoning_scores.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_TRACES_DIR = PROJECT_ROOT / "eval_results" / "traces"

# ────────────────────────────────────────────────────────────────────────────
# Rubric prompt
# ────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a linguistic expert evaluating the quality of arguments in a \
multi-agent debate about diachronic semantic change.

Score each component on FOUR axes using integers 0–5:

  specificity        — Does the text quote specific corpus sentences with year/period labels?
                       0 = no quotes at all, 5 = ≥3 well-attributed quotations.

  temporal_relevance — Does the text explicitly contrast OLD-period vs. NEW-period usage?
                       0 = no temporal framing, 5 = clear before/after analysis.

  logical_validity   — Is the conclusion consistent with the evidence presented?
                       0 = contradicts own evidence, 5 = tightly reasoned, no leaps.

  taxonomy_adherence — Is the predicted change type (from Blank's taxonomy: Metaphor,
                       Metonymy, Analogy, Generalization, Specialization, Ellipsis,
                       Antiphrasis, Auto-Antonym, Synecdoche) correctly applied or
                       argued for, with its definition clearly invoked?
                       0 = wrong or absent taxonomy reference, 5 = precise and justified.

Return ONLY valid JSON in this exact schema (no extra keys):
{
  "support": {
    "specificity": <int 0-5>,
    "temporal_relevance": <int 0-5>,
    "logical_validity": <int 0-5>,
    "taxonomy_adherence": <int 0-5>,
    "notes": "<one sentence>"
  },
  "refuse": {
    "specificity": <int 0-5>,
    "temporal_relevance": <int 0-5>,
    "logical_validity": <int 0-5>,
    "taxonomy_adherence": <int 0-5>,
    "notes": "<one sentence>"
  },
  "judge": {
    "specificity": <int 0-5>,
    "temporal_relevance": <int 0-5>,
    "logical_validity": <int 0-5>,
    "taxonomy_adherence": <int 0-5>,
    "notes": "<one sentence>"
  }
}
"""


def _build_user_message(
    word: str,
    ground_truth: str,
    predicted: str,
    arg_change: str,
    arg_stable: str,
    reasoning: str,
) -> str:
    return f"""\
TARGET WORD: {word}
GROUND TRUTH TYPE: {ground_truth}
PIPELINE PREDICTION: {predicted or "STABLE (no change type predicted)"}

──── TEAM SUPPORT ARGUMENT (argues for semantic change) ────
{arg_change or "(empty)"}

──── TEAM REFUSE ARGUMENT (argues for semantic stability) ────
{arg_stable or "(empty)"}

──── JUDGE REASONING ────
{reasoning or "(empty)"}

Score all three components as described. Return only the JSON object.
"""


# ────────────────────────────────────────────────────────────────────────────
# LLM caller
# ────────────────────────────────────────────────────────────────────────────

def _get_llm():
    """Return a ChatGoogleGenerativeAI instance using the project's LLM backend."""
    from mad_sc.nodes import _get_llm as nodes_get_llm  # reuse project factory
    return nodes_get_llm(temperature=0.1)


def score_trace(llm, trace: dict, max_retries: int = 5) -> dict | None:
    """Call the LLM to score a single trace. Returns the parsed scores dict or None."""
    from langchain_core.messages import HumanMessage, SystemMessage

    word = trace.get("word", "?")
    arg_change = trace.get("arg_change", "")
    arg_stable = trace.get("arg_stable", "")
    reasoning = (trace.get("verdict") or {}).get("reasoning", "")
    ground_truth = trace.get("ground_truth_type", "?")
    predicted = trace.get("predicted_type")

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(
            word, ground_truth, predicted, arg_change, arg_stable, reasoning
        )),
    ]

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            text = response.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            scores = json.loads(text)
            # Validate expected keys
            if all(k in scores for k in ("support", "refuse", "judge")):
                return scores
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error for '{word}' (attempt {attempt+1}): {e}")
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                wait = 60 * (attempt + 1)
                print(f"  [RATE LIMIT] sleeping {wait}s…")
                time.sleep(wait)
            else:
                print(f"  [ERROR] '{word}': {e}")
                return None

    print(f"  [FAIL] '{word}' exhausted {max_retries} retries.")
    return None


# ────────────────────────────────────────────────────────────────────────────
# Reporting
# ────────────────────────────────────────────────────────────────────────────

_AXES = ["specificity", "temporal_relevance", "logical_validity", "taxonomy_adherence"]


def _axis_mean(results: list[dict], component: str, axis: str) -> float:
    vals = [
        r["scores"][component][axis]
        for r in results
        if r.get("scores") and isinstance(r["scores"].get(component), dict)
    ]
    return sum(vals) / len(vals) if vals else 0.0


def _total(scores_component: dict) -> float:
    return sum(scores_component.get(ax, 0) for ax in _AXES)


def print_report(results: list[dict]) -> None:
    print(f"\n{'=' * 75}")
    print("  LLM-as-Judge Reasoning Scores  (0–5 per axis, 0–20 total)")
    print(f"{'=' * 75}")
    header = f"  {'Word':<20} {'Sup':>5} {'Ref':>5} {'Judge':>6}  Mode"
    print(header)
    print(f"  {'-' * 65}")
    for r in results:
        word = r["word"]
        mode = r.get("error_mode", "-")
        sc = r.get("scores")
        if sc:
            sup_t = _total(sc.get("support", {}))
            ref_t = _total(sc.get("refuse", {}))
            jdg_t = _total(sc.get("judge", {}))
            print(f"  {word:<20} {sup_t:>5.1f} {ref_t:>5.1f} {jdg_t:>6.1f}  {mode}")
        else:
            print(f"  {word:<20} {'N/A':>5} {'N/A':>5} {'N/A':>6}  {mode}")
    print(f"  {'-' * 65}")

    # Per-axis aggregate
    print(f"\n  Aggregate (n={len(results)})")
    for component in ("support", "refuse", "judge"):
        label = {"support": "Team Support", "refuse": "Team Refuse", "judge": "Judge"}[component]
        print(f"\n  {label}:")
        for ax in _AXES:
            mean = _axis_mean(results, component, ax)
            print(f"    {ax:<25s} {mean:.2f} / 5.00")

    # Cross-tabulate by error mode
    print("\n  Mean judge total score by error mode:")
    modes: dict[str, list[float]] = {}
    for r in results:
        if not r.get("scores"):
            continue
        m = r.get("error_mode", "unknown")
        modes.setdefault(m, []).append(_total(r["scores"].get("judge", {})))
    for m, vals in sorted(modes.items()):
        mean = sum(vals) / len(vals) if vals else 0
        print(f"    {m:<30s} {mean:.1f} / 20.0  (n={len(vals)})")
    print()


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Reasoning Scorer")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=DEFAULT_TRACES_DIR,
        help="Directory containing per-word JSON trace files (default: eval_results/traces/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save full JSON report",
    )
    parser.add_argument(
        "--words",
        nargs="+",
        default=None,
        help="Restrict scoring to specific words",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between LLM calls (API politeness)",
    )
    args = parser.parse_args()

    traces_dir = args.traces_dir
    if not traces_dir.exists():
        print(f"Error: traces directory not found: {traces_dir}")
        sys.exit(1)

    trace_files = sorted(traces_dir.glob("*.json"))
    if args.words:
        word_set = set(args.words)
        trace_files = [f for f in trace_files if f.stem in word_set]

    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        sys.exit(1)

    print(f"Scoring {len(trace_files)} traces from: {traces_dir}")
    llm = _get_llm()

    results = []
    for i, tf in enumerate(trace_files, 1):
        with open(tf, encoding="utf-8") as f:
            trace = json.load(f)

        word = trace.get("word", tf.stem)
        print(f"[{i}/{len(trace_files)}] Scoring '{word}'…")

        scores = score_trace(llm, trace)
        results.append({
            "word": word,
            "ground_truth_type": trace.get("ground_truth_type"),
            "predicted_type": trace.get("predicted_type"),
            "error_mode": trace.get("error_mode"),
            "scores": scores,
        })

        if i < len(trace_files):
            time.sleep(args.delay)

    print_report(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Full report saved to: {args.output}")


if __name__ == "__main__":
    main()
