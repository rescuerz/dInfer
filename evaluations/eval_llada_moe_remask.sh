#!/bin/bash

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ============================================================================
# Model Configuration
# ============================================================================
model_path='/home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused'

# ============================================================================
# Decoder Configuration
# ============================================================================
# Decoder type: 'threshold', 'hierarchy', or 'slide_window_rcr'
parallel_decoding='slide_window_rcr'

# Threshold parameters (used by all decoders)
threshold=0.9

# SlideWindowRCRDecoder specific parameters
medium_threshold=0.8
low_threshold=0.62
window_size=3
decline_threshold=0.1

# # Enable flags for remask strategies (True/False)
# enable_low_threshold=False
# enable_decline_threshold=False
# enable_consecutive_decline=False

# ============================================================================
# Generation Configuration
# ============================================================================
block_length=32

# ============================================================================
# Cache Configuration
# ============================================================================
cache='dual'  # or 'prefix' or ''
warmup_times=0
prefix_look=0
after_look=0

# ============================================================================
# Other Configuration
# ============================================================================
cont_weight=0
use_credit=False
use_compile=True
use_shift=False
tp_size=4
gpus='0-1-2-3'  # Use hyphen as separator (will be converted to comma in code)
parallel='tp'

# ============================================================================
# Output Configuration
# ============================================================================
base_output_path='./res_remask'

# ============================================================================
# Task Configuration
# ============================================================================
tasks="gsm8k_llada_moe"

# ============================================================================
# Function to build strategy folder name based on enabled flags
# ============================================================================
get_strategy_folder() {
    local en_low=$1
    local en_dec=$2
    local en_cons=$3
    
    local folder=""
    
    if [ "$en_low" = "True" ]; then
        folder="${folder}enLow"
    fi
    
    if [ "$en_dec" = "True" ]; then
        if [ -n "$folder" ]; then
            folder="${folder}_"
        fi
        folder="${folder}enDec"
    fi
    
    if [ "$en_cons" = "True" ]; then
        if [ -n "$folder" ]; then
            folder="${folder}_"
        fi
        folder="${folder}enCons"
    fi
    
    # If no strategy enabled, use "noRemask"
    if [ -z "$folder" ]; then
        folder="noRemask"
    fi
    
    echo "$folder"
}

# ============================================================================
# Run Evaluations - Iterate over all combinations of enable flags
# ============================================================================
for enable_low_threshold in False True; do
    for enable_decline_threshold in False True; do
        for enable_consecutive_decline in False True; do
            for length in 64 128 256; do
                # Get strategy folder name
                strategy_folder=$(get_strategy_folder "$enable_low_threshold" "$enable_decline_threshold" "$enable_consecutive_decline")
                
                # Construct output path
                output_path="${base_output_path}/${strategy_folder}/length${length}_block${block_length}_th${threshold}_med${medium_threshold}_low${low_threshold}_win${window_size}_dec${decline_threshold}"
                
                echo "========================================================================"
                echo "Running evaluation:"
                echo "  Strategy: ${strategy_folder}"
                echo "  gen_length=${length}"
                echo "  block_length=${block_length}"
                echo "  parallel_decoding=${parallel_decoding}"
                echo "  threshold=${threshold}"
                echo "  medium_threshold=${medium_threshold}"
                echo "  low_threshold=${low_threshold}"
                echo "  window_size=${window_size}"
                echo "  decline_threshold=${decline_threshold}"
                echo "  enable_low_threshold=${enable_low_threshold}"
                echo "  enable_decline_threshold=${enable_decline_threshold}"
                echo "  enable_consecutive_decline=${enable_consecutive_decline}"
                echo "  cache=${cache}"
                echo "  output_path=${output_path}"
                echo "========================================================================"
                
                for task in ${tasks}; do
                    echo "Evaluating task: ${task}"
                    
                    # Build model_args
                    model_args="model_path=${model_path}"
                    model_args="${model_args},gen_length=${length}"
                    model_args="${model_args},block_length=${block_length}"
                    model_args="${model_args},threshold=${threshold}"
                    model_args="${model_args},medium_threshold=${medium_threshold}"
                    model_args="${model_args},low_threshold=${low_threshold}"
                    model_args="${model_args},window_size=${window_size}"
                    model_args="${model_args},decline_threshold=${decline_threshold}"
                    model_args="${model_args},enable_low_threshold=${enable_low_threshold}"
                    model_args="${model_args},enable_decline_threshold=${enable_decline_threshold}"
                    model_args="${model_args},enable_consecutive_decline=${enable_consecutive_decline}"
                    model_args="${model_args},show_speed=True"
                    model_args="${model_args},save_dir=${output_path}"
                    model_args="${model_args},parallel_decoding=${parallel_decoding}"
                    model_args="${model_args},cache=${cache}"
                    model_args="${model_args},warmup_times=${warmup_times}"
                    model_args="${model_args},use_compile=${use_compile}"
                    model_args="${model_args},tp_size=${tp_size}"
                    model_args="${model_args},parallel=${parallel}"
                    model_args="${model_args},gpus=${gpus}"
                    model_args="${model_args},cont_weight=${cont_weight}"
                    model_args="${model_args},use_credit=${use_credit}"
                    model_args="${model_args},prefix_look=${prefix_look}"
                    model_args="${model_args},after_look=${after_look}"
                    model_args="${model_args},use_shift=${use_shift}"
                    
                    # Get absolute path for include_path to avoid lm_eval path resolution bug
                    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
                    
                    python eval_dinfer_remask.py \
                        --tasks ${task} \
                        --confirm_run_unsafe_code \
                        --model dInfer_eval \
                        --model_args "${model_args}" \
                        --output_path ${output_path} \
                        --include_path "${SCRIPT_DIR}/tasks" \
                        --apply_chat_template
                    
                    echo "Completed task: ${task}"
                    echo ""
                done
            done
        done
    done
done

echo "========================================================================"
echo "All evaluations completed!"
echo "========================================================================"
