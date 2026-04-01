import os
import sys
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure the parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from mad_sc.data_loader import (
    CORPUS1_LABEL,
    CORPUS2_LABEL,
    get_semeval_contexts,
    get_targets,
    SEMEVAL_DIR
)
from mad_sc.graph import compile_graph
from mad_sc.graph_multi import compile_multi_round_graph
from mad_sc.state import JudgeVerdict

def load_truth():
    truth_file = SEMEVAL_DIR / "truth" / "binary.txt"
    truth = {}
    with open(truth_file, "r") as f:
        for line in f:
            word, label = line.strip().split("\t")
            truth[word] = int(label)
    return truth

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MAD-SC on SemEval-2020 Task 1."
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of random targets to sample (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
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
        metavar="N",
        help="Number of rebuttal rounds for --mode multi (default: 3)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        metavar="N",
        help="Number of corpus sentences shown to agents per period (default: 10)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to write output files (default: current working directory)",
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

    # Set before importing data_loader so the env var is picked up correctly.
    os.environ["SEMEVAL_MAX_SAMPLES"] = str(args.samples)

    load_dotenv()
    truth = load_truth()
    targets = get_targets()
    
    if not targets:
        print("No targets found.")
        return

    if args.seed is not None:
        random.seed(args.seed)
        os.environ["LLM_SEED"] = str(args.seed)

    if args.n is not None:
        targets = random.sample(targets, min(args.n, len(targets)))
        print(f"Sampled {len(targets)} random targets: {targets}")

    mode = args.mode
    num_rounds = max(1, args.rounds)
    graph_kwargs = {}
    if args.use_grounding is not None:
        graph_kwargs["use_grounding"] = args.use_grounding
    if args.use_lexicographer is not None:
        graph_kwargs["use_lexicographer"] = args.use_lexicographer
    if mode == "multi":
        graph = compile_multi_round_graph(num_rounds=num_rounds, **graph_kwargs)
    else:
        graph = compile_graph(**graph_kwargs)

    mode_label = (
        f"multi-round ({num_rounds} rebuttal round{'s' if num_rounds > 1 else ''})"
        if mode == "multi"
        else "single-round (parallel)"
    )

    # Build timestamped output paths so runs never overwrite each other.
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    debate_logs_file   = out_dir / f"debate_logs_{run_ts}.json"
    results_file       = out_dir / f"evaluation_results_{run_ts}.json"
    
    y_true = []
    y_pred = []
    debate_logs = {}
    
    print(f"Evaluating MAD-SC on SemEval-2020 Task 1 | mode: {mode_label} | {args.samples} samples per corpus...")
    print(f"Output directory: {out_dir}")
    print("-" * 65)
    
    for i, word in enumerate(targets, 1):
        true_label = truth.get(word)
        if true_label is None:
            print(f"[{i}/{len(targets)}] Warning: {word} not found in truth file. Skipping.")
            continue
            
        sentences_old, sentences_new = get_semeval_contexts(word)
        
        parts = word.split("_")
        clean_word = parts[0]
        word_type = "noun" if len(parts) > 1 and parts[1] == "nn" else ("verb" if len(parts) > 1 and parts[1] == "vb" else "word")
        
        initial_state = {
            "word": clean_word,
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
        
        
        import time
        max_retries = 10
        pred_label = 0
        
        # Proactively sleep 30 seconds per word to avoid 15 RPM limit 
        # (multi-round debates make many LLM calls per word)
        time.sleep(30)
        
        for attempt in range(max_retries):
            try:
                result = graph.invoke(initial_state)
                verdict_dict = result["verdict"]
                arg_change = result.get("arg_change", "")
                arg_stable = result.get("arg_stable", "")
                
                # verdict can be a dict or a JudgeVerdict object
                v_status = verdict_dict.get("verdict") if isinstance(verdict_dict, dict) else verdict_dict.verdict
                v_reasoning = verdict_dict.get("reasoning") if isinstance(verdict_dict, dict) else verdict_dict.reasoning
                pred_label = 1 if v_status == "CHANGE DETECTED" else 0
                
                print(f"\n--- Debate Thread for '{word}' ({mode_label}) ---")
                if mode == "multi":
                    for entry in result.get("debate_history", []):
                        r = entry["round"]
                        print(f"\n[ROUND {r}/{num_rounds} — Team Support]:\n{entry['arg_change']}\n")
                        print(f"[ROUND {r}/{num_rounds} — Team Refuse]:\n{entry['arg_stable']}\n")
                else:
                    print(f"[TEAM SUPPORT (Change)]:\n{arg_change}\n")
                    print(f"[TEAM REFUSE (Stable)]:\n{arg_stable}\n")
                print(f"[LLM JUDGE]:\nVerdict: {v_status}")
                print(f"Reasoning:\n{v_reasoning}\n")
                print("-" * 65)

                debate_logs[word] = {
                    "arg_change": arg_change,
                    "arg_stable": arg_stable,
                    "debate_history": result.get("debate_history", []),
                    "verdict": v_status,
                    "reasoning": v_reasoning,
                    "debate_mode": mode,
                    "num_rounds": num_rounds,
                }
                
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    print(f"[{i:02d}/{len(targets)}] Rate limit hit, sleeping for 60s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(60)
                else:
                    print(f"[{i:02d}/{len(targets)}] {word}: Error during inference: {e}")
                    break
        else:
            print(f"[{i:02d}/{len(targets)}] {word}: Failed after {max_retries} retries.")
            
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        match = "✓" if true_label == pred_label else "✗"
        print(f"[{i:02d}/{len(targets)}] {word:15} | True: {true_label} | Pred: {pred_label} | {match}")
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("-" * 65)
    print("Evaluation Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Save debate logs
    with open(debate_logs_file, "w") as f:
        json.dump(debate_logs, f, indent=2)
    print(f"Debate transcripts saved to {debate_logs_file}")
    
    try:
        from scripts.export_to_markdown import export_debate_to_md
        md_file = out_dir / f"debate_logs_{run_ts}.md"
        export_debate_to_md(str(debate_logs_file), str(md_file))
    except Exception as e:
        print(f"Could not export Markdown: {e}")
    
    # Save results
    with open(results_file, "w") as f:
        json.dump({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "y_true": y_true,
            "y_pred": y_pred,
            "targets": list(targets),
            "debate_mode": mode,
            "num_rounds": num_rounds,
            "samples_per_corpus": args.samples,
            "run_timestamp": run_ts,
        }, f, indent=2)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
