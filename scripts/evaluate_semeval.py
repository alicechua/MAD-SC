import os
import sys
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure the parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set max samples to 10 before importing data_loader
os.environ["SEMEVAL_MAX_SAMPLES"] = "10"

from dotenv import load_dotenv
from mad_sc.data_loader import (
    CORPUS1_LABEL,
    CORPUS2_LABEL,
    get_semeval_contexts,
    get_targets,
    SEMEVAL_DIR
)
from mad_sc.graph import compile_graph
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
    load_dotenv()
    truth = load_truth()
    targets = get_targets()
    
    if not targets:
        print("No targets found.")
        return

    graph = compile_graph()
    
    y_true = []
    y_pred = []
    debate_logs = {}
    
    print(f"Evaluating MAD-SC on SemEval-2020 Task 1 ({os.environ['SEMEVAL_MAX_SAMPLES']} samples per corpus)...")
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
            "verdict": None,
        }
        
        import time
        max_retries = 10
        pred_label = 0
        
        # Proactively sleep 15 seconds per word to avoid 15 RPM limit 
        # (each graph run is ~3 LLM calls)
        time.sleep(15)
        
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
                
                print(f"\n--- Debate Thread for '{word}' ---")
                print(f"[TEAM SUPPORT (Change)]:\n{arg_change}\n")
                print(f"[TEAM REFUSE (Stable)]:\n{arg_stable}\n")
                print(f"[LLM JUDGE]:\nVerdict: {v_status}")
                print(f"Reasoning:\n{v_reasoning}\n")
                print("-" * 65)

                debate_logs[word] = {
                    "arg_change": arg_change,
                    "arg_stable": arg_stable,
                    "verdict": v_status,
                    "reasoning": v_reasoning
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
    
    # Save debate logs to a separate file
    debate_logs_file = Path("debate_logs.json")
    with open(debate_logs_file, "w") as f:
        json.dump(debate_logs, f, indent=2)
    print(f"Debate transcripts saved to {debate_logs_file}")
    
    # Save results to a file
    results_file = Path("evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "y_true": y_true,
            "y_pred": y_pred,
            "targets": list(targets)
        }, f, indent=2)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
