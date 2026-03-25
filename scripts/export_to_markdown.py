import json
import sys
import os

def extract_text(content):
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        )
    if hasattr(content, "content"):
        return extract_text(content.content)
    if isinstance(content, dict) and "content" in content:
        return extract_text(content["content"])
    return str(content)

def export_debate_to_md(json_path, out_path=None):
    if not out_path:
        out_path = os.path.splitext(json_path)[0] + ".md"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(out_path, 'w', encoding='utf-8') as out:
        out.write(f"# Debate Logs: {os.path.basename(json_path)}\n\n")
        out.write(f"Generated from `{json_path}`.\n\n---\n\n")

        for word, details in data.items():
            out.write(f"## Target Word: `{word}`\n\n")
            
            # Print Verdict
            verdict = details.get("verdict", "N/A")
            out.write(f"**Verdict**: {verdict}  \n")
            
            # Print configuration
            num_rounds = details.get("num_rounds", 1)
            debate_mode = details.get("debate_mode", "unknown")
            timestamp = details.get("timestamp", "")
            out.write(f"**Debate Mode**: {debate_mode} | **Rounds**: {num_rounds} | **Time**: {timestamp}\n\n")

            history = details.get("debate_history", [])
            
            if not history:
                # Single round fallback
                arg_change = extract_text(details.get("arg_change", ""))
                arg_stable = extract_text(details.get("arg_stable", ""))
                
                out.write("### Opening Statements\n\n")
                out.write("#### 🔴 Team Support (Hypothesis: Change)\n\n")
                out.write(f"{arg_change}\n\n")
                out.write("#### 🟢 Team Refuse (Hypothesis: Stable)\n\n")
                out.write(f"{arg_stable}\n\n")
            else:
                for round_info in history:
                    r_num = round_info.get("round", "?")
                    if str(r_num) == "0":
                         out.write(f"### Opening Statements\n\n")
                    else:
                         out.write(f"### Rebuttal - Round {r_num}\n\n")
                    
                    arg_change = extract_text(round_info.get("arg_change", ""))
                    arg_stable = extract_text(round_info.get("arg_stable", ""))
                    
                    if arg_change:
                        out.write("#### 🔴 Team Support (Hypothesis: Change)\n\n")
                        out.write(f"{arg_change}\n\n")
                    
                    if arg_stable:
                        out.write("#### 🟢 Team Refuse (Hypothesis: Stable)\n\n")
                        out.write(f"{arg_stable}\n\n")

            # Judge Reasoning
            out.write("### ⚖️ LLM Judge\n\n")
            reasoning = details.get("reasoning", "No reasoning available.")
            out.write(f"{reasoning}\n\n")
            
            change_type = details.get("change_type")
            if change_type:
                out.write(f"- **Change Type**: {change_type}\n")
            
            causal_driver = details.get("causal_driver")
            if causal_driver:
                out.write(f"- **Causal Driver**: {causal_driver}\n")
                
            break_point = details.get("break_point_year")
            if break_point:
                out.write(f"- **Break-point Year**: {break_point}\n")

            out.write("\n---\n\n")

    print(f"✅ Successfully exported {json_path} to {out_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/export_to_markdown.py <path_to_json_in_results>")
        sys.exit(1)
    
    export_debate_to_md(sys.argv[1])
