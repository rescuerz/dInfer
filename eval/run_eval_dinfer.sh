#!/bin/bash

# dInfer Countdown/Sudoku Evaluation Script

model_path="/path/to/your/model"  # e.g., inclusionAI/LLaDA-MoE-7B-A1B-Instruct
dataset="countdown"  # countdown or sudoku
gen_length=256
block_length=32
threshold=0.9
subsample=256
output_dir="results/"
cache="dual"  # "", "prefix", or "dual"

# Optional: absolute path to dataset directory
# data_dir="/path/to/dInfer/dataset"

torchrun --nproc_per_node=1 eval_dinfer.py \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --gen_length ${gen_length} \
    --block_length ${block_length} \
    --threshold ${threshold} \
    --subsample ${subsample} \
    --output_dir ${output_dir} \
    --cache ${cache}
    # --data_dir ${data_dir}

# After generation, parse results:
# python parse_and_get_acc.py --directory ${output_dir}
