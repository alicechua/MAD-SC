#!/bin/bash
# 5x5 Multi-Model Baseline Benchmark Runner (macOS Bash 3.2 Compatible)

# Setup virtual environment
source .venv/bin/activate

# Base configuration
SEED=67
DELAY=8
MODE="single"
export USE_SELF_CONSISTENCY=false

# Define models and their backends separated by a space
MODELS=(
    # "gemini-3.1-flash-lite-preview google_ai_studio" # Daily quota exhausted!
    # "deepseek-ai/DeepSeek-V3.2-fast nebius"
    # "deepseek-ai/DeepSeek-R1-0528-fast nebius"
    "Qwen/Qwen3.5-397B-A17B-fast nebius"
    "nvidia/nemotron-3-super-120b-a12b nebius"
)

echo "========================================="
echo "Starting 5x5 Benchmark Execution"
echo "========================================="

for ENTRY in "${MODELS[@]}"; do
    # Extract model and backend manually since no associative arrays in Bash 3.2
    MODEL=$(echo "$ENTRY" | awk '{print $1}')
    BACKEND=$(echo "$ENTRY" | awk '{print $2}')
    
    echo "Starting evaluation suite for model: $MODEL (Backend: $BACKEND)"
    
    # Sanitize model name for directory path
    DIR_NAME=$(echo "$MODEL" | tr '/' '_')
    
    for RUN in {1..5}; do
        OUT_DIR="evals/baseline/$DIR_NAME/run$RUN"
        LOG_FILE="/tmp/eval_${DIR_NAME}_run${RUN}.log"
        mkdir -p "$OUT_DIR"
        
        echo "  - Spawning Run $RUN -> $OUT_DIR (Log: $LOG_FILE)"
        
        # Configure the environment dynamically for this specific background task
        (
            export LLM_BACKEND="$BACKEND"
            if [ "$BACKEND" = "nebius" ]; then
                export NEBIUS_MODEL="$MODEL"
                export JUDGE_MODEL_NEBIUS="$MODEL"
            elif [ "$BACKEND" = "google_ai_studio" ]; then
                export DEFAULT_MODEL_GAS="$MODEL"
                export JUDGE_MODEL_GAS="$MODEL"
            fi
            
            python3 scripts/evaluate_lsc.py \
                --output-dir "$OUT_DIR" \
                --context-json data/lsc_context_data_engl.json \
                --mode "$MODE" \
                --resume \
                --no-grounding \
                --no-lexicographer \
                --seed "$SEED" \
                --delay "$DELAY" > "$LOG_FILE" 2>&1
        ) &
    done
    
    # Wait for the 5 parallel runs for this specific model to finish before starting the next model to avoid massive rate limits!
    echo "Waiting for 5 parallel instances of $MODEL to complete..."
    wait
    echo "$MODEL evaluations complete!"
    echo "-----------------------------------------"
done

echo "All baseline evaluations completed successfully!"
