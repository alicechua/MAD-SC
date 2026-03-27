#!/usr/bin/env python3
"""
Evidence Grounding & Faithfulness Analyser
==========================================
Reads saved trace files from an evaluation run and computes two sets of
metrics without making any LLM calls:

  1. Evidence grounding rate — how well-cited are each team's arguments?
       • quoted_strings   : number of passages in quotation marks (" " or ' ')
       • year_refs        : number of 4-digit year tokens (1800–2020)
       • citations_score  : average of the two (normalised 0–10)

  2. Reasoning faithfulness — does the judge's reasoning reflect both sides?
       • support_overlap  : word-overlap ratio with Team Support argument
       • refuse_overlap   : word-overlap ratio with Team Refuse argument
       • faithfulness     : harmonic mean of the two overlaps (0–1)

Usage
-----
    # Analyse the default eval_results/traces/ directory
    python scripts/analyze_grounding.py

    # Point at a specific run
    python scripts/analyze_grounding.py --traces-dir evals/eval_results_lexicographer_full/traces

    # Save JSON output
    python scripts/analyze_grounding.py --output grounding_report.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TRACES_DIR = PROJECT_ROOT / "eval_results" / "traces"

# Regex for 4-digit years plausible in historical-linguistic context
_YEAR_RE = re.compile(r"\b(1[89]\d\d|20[01]\d)\b")

# Regex for quoted passages (double or single quotes, at least 3 chars inside)
_QUOTE_RE = re.compile(r'["\u201c\u201d]([^"]{3,})["\u201c\u201d]|\'([^\']{3,})\'')

# Common function words to exclude from faithfulness overlap
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "it", "this", "that", "as",
    "its", "their", "from", "by", "be", "been", "has", "have", "had",
    "not", "no", "which", "who", "what", "when", "where", "how", "both",
    "also", "more", "than", "so", "if", "while", "although", "however",
    "word", "sense", "meaning", "use", "used", "usage",
}


# ────────────────────────────────────────────────────────────────────────────
# Grounding helpers
# ────────────────────────────────────────────────────────────────────────────

def count_quoted_strings(text: str) -> int:
    """Count distinct quoted passages in *text*."""
    return len(_QUOTE_RE.findall(text))


def count_year_refs(text: str) -> int:
    """Count 4-digit year tokens (1800–2020) in *text*."""
    return len(_YEAR_RE.findall(text))


def grounding_score(text: str) -> dict:
    """Return a grounding dict for a single argument text."""
    q = count_quoted_strings(text)
    y = count_year_refs(text)
    # Normalise each to 0–5 (cap at 5 for ≥5 citations/years) then average
    score = (min(q, 5) + min(y, 5)) / 2
    return {"quoted_strings": q, "year_refs": y, "citations_score": round(score, 2)}


# ────────────────────────────────────────────────────────────────────────────
# Faithfulness helpers
# ────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def word_overlap_ratio(source: str, reference: str) -> float:
    """Fraction of unique content words in *source* that appear in *reference*."""
    src_tokens = _tokenise(source)
    ref_tokens = _tokenise(reference)
    if not src_tokens:
        return 0.0
    return len(src_tokens & ref_tokens) / len(src_tokens)


def faithfulness_score(reasoning: str, arg_change: str, arg_stable: str) -> dict:
    """Measure how much the judge's reasoning reflects each team's argument."""
    sup = word_overlap_ratio(reasoning, arg_change)
    ref = word_overlap_ratio(reasoning, arg_stable)
    # Harmonic mean — low on either side → low overall
    if sup + ref == 0:
        hm = 0.0
    else:
        hm = 2 * sup * ref / (sup + ref)
    return {
        "support_overlap": round(sup, 3),
        "refuse_overlap": round(ref, 3),
        "faithfulness": round(hm, 3),
    }


# ────────────────────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────────────────────

def analyse_trace(trace: dict) -> dict:
    """Compute grounding + faithfulness metrics for one trace."""
    word = trace.get("word", "?")
    arg_change = trace.get("arg_change", "")
    arg_stable = trace.get("arg_stable", "")
    reasoning = (trace.get("verdict") or {}).get("reasoning", "")

    support_grounding = grounding_score(arg_change)
    refuse_grounding = grounding_score(arg_stable)
    faith = faithfulness_score(reasoning, arg_change, arg_stable)

    return {
        "word": word,
        "ground_truth_type": trace.get("ground_truth_type"),
        "predicted_type": trace.get("predicted_type"),
        "error_mode": trace.get("error_mode"),
        "team_support_grounding": support_grounding,
        "team_refuse_grounding": refuse_grounding,
        "faithfulness": faith,
    }


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_report(results: list[dict]) -> None:
    print(f"\n{'=' * 70}")
    print("  Evidence Grounding & Faithfulness Report")
    print(f"{'=' * 70}")
    print(f"  {'Word':<20} {'Sup-Q':>5} {'Sup-Y':>5} {'Sup-S':>5}  "
          f"{'Ref-Q':>5} {'Ref-Y':>5} {'Ref-S':>5}  {'Faith':>6}  Mode")
    print(f"  {'-' * 68}")
    for r in results:
        sg = r["team_support_grounding"]
        rg = r["team_refuse_grounding"]
        f = r["faithfulness"]
        mode = r.get("error_mode", "-")
        print(f"  {r['word']:<20} "
              f"{sg['quoted_strings']:>5} {sg['year_refs']:>5} {sg['citations_score']:>5.1f}  "
              f"{rg['quoted_strings']:>5} {rg['year_refs']:>5} {rg['citations_score']:>5.1f}  "
              f"{f['faithfulness']:>6.3f}  {mode}")
    print(f"  {'-' * 68}")

    # Aggregate
    sup_scores = [r["team_support_grounding"]["citations_score"] for r in results]
    ref_scores = [r["team_refuse_grounding"]["citations_score"] for r in results]
    faith_scores = [r["faithfulness"]["faithfulness"] for r in results]
    print(f"\n  Aggregate (n={len(results)})")
    print(f"    Mean Team Support grounding score : {_mean(sup_scores):.2f} / 5.00")
    print(f"    Mean Team Refuse  grounding score : {_mean(ref_scores):.2f} / 5.00")
    print(f"    Mean faithfulness                 : {_mean(faith_scores):.3f} / 1.000")
    print()

    # Faithfulness breakdown by error mode
    modes = {}
    for r in results:
        m = r.get("error_mode", "unknown")
        modes.setdefault(m, []).append(r["faithfulness"]["faithfulness"])
    print("  Faithfulness by error mode:")
    for m, vals in sorted(modes.items()):
        print(f"    {m:<30s} mean={_mean(vals):.3f}  n={len(vals)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evidence Grounding & Faithfulness Analyser")
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
        help="Optional path to save full JSON report (e.g. grounding_report.json)",
    )
    parser.add_argument(
        "--words",
        nargs="+",
        default=None,
        help="Restrict analysis to specific words",
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

    results = []
    for tf in trace_files:
        with open(tf, encoding="utf-8") as f:
            trace = json.load(f)
        results.append(analyse_trace(trace))

    print_report(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Full report saved to: {args.output}")


if __name__ == "__main__":
    main()
