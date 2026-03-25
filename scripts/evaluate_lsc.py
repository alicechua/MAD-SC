#!/usr/bin/env python3
"""
LSC Evaluation Harness
======================
Orchestrates the MAD-SC agentic pipeline over `:engl` words from the
LSC-CTD dataset, evaluates predictions against ground truth, and produces
classification metrics, a confusion matrix, and per-word reasoning traces.

Usage
-----
    # Full run on all 25 :engl words
    python scripts/evaluate_pipeline.py

    # Quick test on specific words
    python scripts/evaluate_pipeline.py --words corn horn

    # Custom output directory
    python scripts/evaluate_pipeline.py --output-dir eval_results_test
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `mad_sc` is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mad_sc.graph import compile_graph  # noqa: E402
from mad_sc.graph_multi import compile_multi_round_graph  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("lsc_eval")

# ---------------------------------------------------------------------------
# Paths (adjust if your layout differs)
# ---------------------------------------------------------------------------
CONTEXT_JSON = PROJECT_ROOT / "data" / "lsc_context_data_engl.json"
GROUND_TRUTH_TSV = PROJECT_ROOT / "data" / "LSC-CTD" / "blank_dataset.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_results"

# ---------------------------------------------------------------------------
# Column indices in blank_dataset.tsv (0-indexed)
# Adjust these if the TSV schema changes.
# ---------------------------------------------------------------------------
TSV_COL_WORD = 0     # "Words"   — e.g. "corn:engl"
TSV_COL_TYPE = 6     # "Type"    — e.g. "Specialization"

# ---------------------------------------------------------------------------
# Taxonomy mapping — coarse-grained grouping
# Maps both ground-truth labels AND pipeline labels into 3 super-categories.
# ---------------------------------------------------------------------------
COARSE_MAP: dict[str, str] = {
    # Ground-truth types
    "Specialization":         "Narrowing",
    "Ellipsis":               "Narrowing",
    "Generalization":         "Broadening",
    "Analogy":                "Broadening",
    "Metaphor":               "Transfer",
    "Metonymy":               "Transfer",
    "Auto-Antonym":           "Transfer",
    "Antiphrasis":            "Transfer",
    # Pipeline output types (old coarse labels still accepted for backward compat)
    "Co-hyponymous Transfer": "Transfer",
}

# Fine-grained label set — all valid Blank's taxonomy change types
FINE_GRAINED_LABELS = [
    "Metaphor", "Metonymy", "Analogy", "Generalization",
    "Specialization", "Ellipsis", "Antiphrasis", "Auto-Antonym", "Synecdoche",
]

# Fuzzy alias map for label normalization (Improvement 4)
LABEL_ALIASES: dict[str, str] = {
    "metaphor": "Metaphor",
    "metaphorical": "Metaphor",
    "metonym": "Metonymy",
    "metonymic": "Metonymy",
    "analogy": "Analogy",
    "analogical": "Analogy",
    "generali": "Generalization",
    "broadening": "Generalization",
    "speciali": "Specialization",
    "narrowing": "Specialization",
    "ellipsis": "Ellipsis",
    "antiphrasis": "Antiphrasis",
    "auto-antonym": "Auto-Antonym",
    "autantonym": "Auto-Antonym",
    "synecdoche": "Synecdoche",
}

# Pipeline timeout per word (seconds)
PIPELINE_TIMEOUT = 120


# ===================================================================
# 1. Data Ingestion
# ===================================================================

def load_ground_truth(
    tsv_path: Path = GROUND_TRUTH_TSV,
    lang_suffix: str = ":engl",
) -> dict[str, str]:
    """
    Parse the LSC-CTD TSV and return {clean_word: change_type} for all
    entries matching *lang_suffix*.

    Word cleaning:
        "corn:engl"        → "corn"
        "to observe:engl"  → "observe"
    """
    truth: dict[str, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) <= max(TSV_COL_WORD, TSV_COL_TYPE):
                continue
            raw_word = row[TSV_COL_WORD].strip()
            change_type = row[TSV_COL_TYPE].strip()
            if not raw_word.endswith(lang_suffix):
                continue
            clean = raw_word.removesuffix(lang_suffix).strip()
            # Strip leading "to " for verb forms
            if clean.startswith("to "):
                clean = clean[3:]
            truth[clean] = change_type
    log.info("Ground truth: loaded %d :engl entries", len(truth))
    return truth


def load_context_data(json_path: Path = CONTEXT_JSON) -> list[dict]:
    """Load the context JSON produced by the data pipeline."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Context data: loaded %d word entries", len(data))
    return data


def align_words(
    context_data: list[dict],
    ground_truth: dict[str, str],
) -> list[dict]:
    """
    Return a list of aligned records with keys:
        word, modern_context, historical_context, ground_truth_type
    Skips any word not present in both sources.
    """
    aligned = []
    for entry in context_data:
        word = entry["word"]
        if word in ground_truth:
            aligned.append({
                "word": word,
                "modern_context": entry.get("modern_context", []),
                "historical_context": entry.get("historical_context", []),
                "ground_truth_type": ground_truth[word],
            })
        else:
            log.warning("  Word '%s' not found in ground truth — skipping", word)
    log.info("Aligned %d words for evaluation", len(aligned))
    return aligned


# ===================================================================
# 2. Pipeline Orchestration
# ===================================================================

def run_pipeline_for_word(
    graph,
    word: str,
    sentences_old: list[str],
    sentences_new: list[str],
    num_rounds: int = 1,
) -> dict:
    """
    Invoke the MAD-SC pipeline for a single word.

    Returns the full result dict from the LangGraph invocation, or a
    sentinel dict on failure.
    """
    initial_state = {
        "word": word,
        "t_old": "Historical Era (Old/Middle English)",
        "t_new": "Modern Era (Contemporary English)",
        "sentences_old": sentences_old[:10],  # cap at 10 sentences
        "sentences_new": sentences_new[:10],
        "arg_change": "",
        "arg_stable": "",
        "num_rounds": num_rounds,
        "current_round": 1,
        "debate_history": [],
        "verdict": None,
    }

    max_retries = 10
    for attempt in range(max_retries):
        try:
            result = graph.invoke(initial_state)
            return result
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                log.warning("  Rate limit hit for '%s', sleeping 60s… (attempt %d/%d)",
                            word, attempt + 1, max_retries)
                time.sleep(60)
            else:
                log.error("  Pipeline failed for '%s': %s", word, exc)
                traceback.print_exc()
                return {
                    "word": word,
                    "arg_change": f"ERROR: {exc}",
                    "arg_stable": f"ERROR: {exc}",
                    "verdict": {
                        "word": word,
                        "verdict": "ERROR",
                        "change_type": None,
                        "causal_driver": None,
                        "break_point_year": None,
                        "reasoning": f"Pipeline error: {exc}",
                    },
                }
    log.error("  '%s' failed after %d retries (rate limit).", word, max_retries)
    return {
        "word": word,
        "arg_change": "ERROR: rate limit",
        "arg_stable": "ERROR: rate limit",
        "verdict": {
            "word": word,
            "verdict": "ERROR",
            "change_type": None,
            "causal_driver": None,
            "break_point_year": None,
            "reasoning": f"Rate limit: exhausted {max_retries} retries.",
        },
    }


def extract_predicted_type(result: dict) -> Optional[str]:
    """Extract the predicted change_type from a pipeline result dict.

    Applies label normalization (Improvement 4): if the raw output is a
    near-miss like 'metaphorical transfer', maps it to the canonical label.
    """
    verdict = result.get("verdict")
    if not verdict or not isinstance(verdict, dict):
        return None
    raw = verdict.get("change_type")
    if raw is None:
        return None

    # Direct match: already a valid label
    if raw in FINE_GRAINED_LABELS:
        return raw

    # Fuzzy normalization: check aliases
    raw_lower = raw.lower()
    for alias, canonical in LABEL_ALIASES.items():
        if alias in raw_lower:
            return canonical

    return raw  # return as-is; will be marked UNKNOWN in coarse mapping if unrecognized


# ===================================================================
# 3. Evaluation Metrics
# ===================================================================

def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    label: str = "Fine-grained",
) -> dict:
    """
    Compute classification metrics and print a summary.
    Returns a dict with accuracy, macro-precision, macro-recall, macro-f1,
    and the confusion matrix.

    Uses sklearn if available; falls back to a manual implementation.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )

        # Get all unique labels (sorted for reproducibility)
        labels = sorted(set(y_true) | set(y_pred))

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True,
        )
        report_str = classification_report(
            y_true, y_pred, labels=labels, zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        print(f"\n{'=' * 65}")
        print(f"  {label} Evaluation")
        print(f"{'=' * 65}")
        print(f"  Accuracy: {acc:.3f}")
        print(f"\n{report_str}")

        # Print confusion matrix
        print(f"Confusion Matrix (rows=true, cols=predicted):")
        print(f"  Labels: {labels}")
        for i, row in enumerate(cm):
            print(f"  {labels[i]:30s} {list(row)}")
        print()

        return {
            "accuracy": acc,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    except ImportError:
        log.warning("sklearn not installed — computing basic accuracy only")
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        acc = correct / len(y_true) if y_true else 0.0
        print(f"\n  {label} Accuracy: {acc:.3f} ({correct}/{len(y_true)})")
        return {"accuracy": acc}


def coarsen(label: Optional[str]) -> str:
    """Map a fine-grained label to the coarse 3-class scheme."""
    if label is None:
        return "UNKNOWN"
    return COARSE_MAP.get(label, "UNKNOWN")


# ===================================================================
# 4. Trace Logging
# ===================================================================

def save_trace(
    output_dir: Path,
    word: str,
    result: dict,
    ground_truth_type: str,
    predicted_type: Optional[str],
):
    """Save full pipeline trace for a single word."""
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace = {
        "word": word,
        "ground_truth_type": ground_truth_type,
        "predicted_type": predicted_type,
        "coarse_true": coarsen(ground_truth_type),
        "coarse_pred": coarsen(predicted_type),
        "arg_change": result.get("arg_change", ""),
        "arg_stable": result.get("arg_stable", ""),
        "debate_history": result.get("debate_history", []),
        "verdict": result.get("verdict", {}),
        "timestamp": datetime.now().isoformat(),
    }

    with open(trace_dir / f"{word}.json", "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)


def save_summary(
    output_dir: Path,
    records: list[dict],
    fine_metrics: dict,
    coarse_metrics: dict,
):
    """Save the overall evaluation summary to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_words": len(records),
        "fine_grained_metrics": _serializable(fine_metrics),
        "coarse_grained_metrics": _serializable(coarse_metrics),
        "per_word": [
            {
                "word": r["word"],
                "ground_truth": r["ground_truth_type"],
                "predicted": r.get("predicted_type"),
                "coarse_true": coarsen(r["ground_truth_type"]),
                "coarse_pred": coarsen(r.get("predicted_type")),
                "correct_fine": r["ground_truth_type"] == r.get("predicted_type"),
                "correct_coarse": coarsen(r["ground_truth_type"]) == coarsen(r.get("predicted_type")),
            }
            for r in records
        ],
    }
    path = output_dir / "eval_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Summary saved to %s", path)


def _serializable(obj):
    """Make numpy/int64 objects JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


# ===================================================================
# 5. Main Orchestration
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="LSC Pipeline Evaluation Harness")
    parser.add_argument(
        "--words",
        nargs="+",
        default=None,
        help="Subset of words to evaluate (default: all 25 :engl words)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for results and traces (default: eval_results/)",
    )
    parser.add_argument(
        "--context-json",
        type=Path,
        default=CONTEXT_JSON,
        help="Path to lsc_context_data_engl.json",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=GROUND_TRUTH_TSV,
        help="Path to blank_dataset.tsv",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between pipeline invocations (API politeness)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip words that already have a successful trace in the output directory",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Debate mode: 'single' (parallel, default) or 'multi' (rebuttal rounds)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rebuttal rounds for --mode multi (default: 3)",
    )
    grounding_group = parser.add_mutually_exclusive_group()
    grounding_group.add_argument(
        "--grounding",
        dest="use_grounding",
        action="store_true",
        default=None,
        help="Enable pre-debate BERT grounding (SED/TD). Overrides USE_GROUNDING env var.",
    )
    grounding_group.add_argument(
        "--no-grounding",
        dest="use_grounding",
        action="store_false",
        help="Disable pre-debate BERT grounding. Overrides USE_GROUNDING env var.",
    )
    lex_group = parser.add_mutually_exclusive_group()
    lex_group.add_argument(
        "--lexicographer",
        dest="use_lexicographer",
        action="store_true",
        default=None,
        help="Enable Lexicographer Agent (Definition Dossier). Overrides USE_LEXICOGRAPHER env var.",
    )
    lex_group.add_argument(
        "--no-lexicographer",
        dest="use_lexicographer",
        action="store_false",
        help="Disable Lexicographer Agent. Overrides USE_LEXICOGRAPHER env var.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load and align data
    # ------------------------------------------------------------------
    ground_truth = load_ground_truth(args.ground_truth)
    context_data = load_context_data(args.context_json)
    aligned = align_words(context_data, ground_truth)

    # Filter to requested subset if specified
    if args.words:
        word_set = set(args.words)
        aligned = [a for a in aligned if a["word"] in word_set]
        log.info("Filtered to %d requested word(s): %s", len(aligned), args.words)

    if not aligned:
        log.error("No words to evaluate — exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Compile pipeline graph
    # ------------------------------------------------------------------
    mode = args.mode
    num_rounds = max(1, args.rounds)

    # use_grounding=None means the flag wasn't passed → fall back to compile_graph default
    graph_kwargs = {}
    if args.use_grounding is not None:
        graph_kwargs["use_grounding"] = args.use_grounding
    if args.use_lexicographer is not None:
        graph_kwargs["use_lexicographer"] = args.use_lexicographer
    log.info("Compiling MAD-SC LangGraph pipeline… (grounding=%s, lexicographer=%s, mode=%s, rounds=%d)",
             graph_kwargs.get("use_grounding", "env/default"),
             graph_kwargs.get("use_lexicographer", "env/default"),
             mode, num_rounds)

    if mode == "multi":
        graph = compile_multi_round_graph(num_rounds=num_rounds, **graph_kwargs)
    else:
        graph = compile_graph(**graph_kwargs)

    # ------------------------------------------------------------------
    # 3. Run pipeline on each word
    # ------------------------------------------------------------------
    records = []
    for i, entry in enumerate(aligned, 1):
        word = entry["word"]
        log.info("=" * 60)
        log.info("[%d/%d] Running pipeline for '%s' (truth: %s)",
                 i, len(aligned), word, entry["ground_truth_type"])
        log.info("=" * 60)

        trace_path = args.output_dir / "traces" / f"{word}.json"

        # Resume logic: skip if a successful trace already exists
        if args.resume and trace_path.exists():
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    trace = json.load(f)
                verdict_obj = trace.get("verdict", {})
                v_status = verdict_obj.get("verdict")
                
                # If it's a valid completed output (not ERROR or empty)
                if v_status and v_status not in ["ERROR", "N/A"]:
                    log.info("  Found existing successful trace for '%s' — skipping pipeline", word)
                    result = {
                        "arg_change": trace.get("arg_change", ""),
                        "arg_stable": trace.get("arg_stable", ""),
                        "debate_history": trace.get("debate_history", []),
                        "verdict": verdict_obj
                    }
                    predicted = trace.get("predicted_type")
                    entry["predicted_type"] = predicted
                    entry["pipeline_result"] = result
                    records.append(entry)
                    continue
            except Exception as e:
                log.warning("  Failed to read existing trace for '%s', re-running: %s", word, e)

        result = run_pipeline_for_word(
            graph,
            word=word,
            sentences_old=entry["historical_context"],
            sentences_new=entry["modern_context"],
            num_rounds=num_rounds,
        )

        predicted = extract_predicted_type(result)
        entry["predicted_type"] = predicted
        entry["pipeline_result"] = result
        records.append(entry)

        print(f"\n--- Debate Thread for '{word}' ---")
        if mode == "multi":
            for r_entry in result.get("debate_history", []):
                r = r_entry.get("round", "")
                print(f"\n[ROUND {r}/{num_rounds} — Team Support]:\n{r_entry.get('arg_change', '')}\n")
                print(f"[ROUND {r}/{num_rounds} — Team Refuse]:\n{r_entry.get('arg_stable', '')}\n")
        else:
            print(f"[TEAM SUPPORT (Change)]:\n{result.get('arg_change', '')}\n")
            print(f"[TEAM REFUSE (Stable)]:\n{result.get('arg_stable', '')}\n")
        
        verdict_obj = result.get("verdict", {})
        v_status = verdict_obj.get("verdict", "N/A")
        v_reasoning = verdict_obj.get("reasoning", "")
        print(f"[LLM JUDGE]:\nVerdict: {v_status}")
        print(f"Reasoning:\n{v_reasoning}\n")
        print("-" * 65)

        # Log per-word result
        verdict_label = (result.get("verdict") or {}).get("verdict", "N/A")
        log.info(
            "  Result: verdict=%s  predicted=%s  truth=%s  %s",
            verdict_label,
            predicted or "None",
            entry["ground_truth_type"],
            "✓" if predicted == entry["ground_truth_type"] else "✗",
        )

        # Save trace
        save_trace(
            args.output_dir, word, result,
            entry["ground_truth_type"], predicted,
        )

        # Politeness delay between API calls
        if i < len(aligned):
            time.sleep(args.delay)

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    y_true_fine = [r["ground_truth_type"] for r in records]
    y_pred_fine = [r.get("predicted_type") or "NONE" for r in records]

    y_true_coarse = [coarsen(t) for t in y_true_fine]
    y_pred_coarse = [coarsen(p) if p != "NONE" else "UNKNOWN" for p in y_pred_fine]

    fine_metrics = compute_metrics(y_true_fine, y_pred_fine, "Fine-grained")
    coarse_metrics = compute_metrics(y_true_coarse, y_pred_coarse, "Coarse-grained")

    # ------------------------------------------------------------------
    # 5. Save summary
    # ------------------------------------------------------------------
    save_summary(args.output_dir, records, fine_metrics, coarse_metrics)

    print(f"\nAll results saved to: {args.output_dir}/")
    print(f"  eval_summary.json  — metrics + per-word results")
    print(f"  traces/            — full pipeline outputs per word")


if __name__ == "__main__":
    main()
