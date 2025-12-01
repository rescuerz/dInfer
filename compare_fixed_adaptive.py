#!/usr/bin/env python3
"""
dInfer 固定vs自适应专家激活比较脚本

比较固定专家数和动态专家激活的性能差异。
"""

import os
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

from dinfer.model import FusedOlmoeForCausalLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM, FixedParallelDecoder


def compare_fixed_vs_adaptive(model, tokenizer, input_ids, args):
    """比较固定专家激活和自适应专家激活的差异

    Args:
        model: FusedOlmoeForCausalLM 模型
        tokenizer: tokenizer
        input_ids: 输入 token ids，shape [1, seq_len]
        args: 命令行参数
    """
    print("\n" + "="*80)
    print("Comparison Mode: Fixed vs Adaptive Expert Activation")
    print("="*80)

    # 1. 运行固定专家激活
    print("\n🔹 Running FIXED expert activation...")
    print("-"*80)

    # 创建解码器和扩散LLM（固定模式）
    decoder_fixed = ThresholdParallelDecoder(
        temperature=0.0,
        threshold=args.threshold,
        mask_id=args.mask_id,
        eos_id=args.eos_id
    )
    # decoder_fixed = FixedParallelDecoder(
    #     temperature=0.0,
    #     steps=args.steps,
    #     mask_id=args.mask_id
    # )

    dllm_fixed = BlockWiseDiffusionLLM(
        model=model,
        decoder=decoder_fixed,
        iterator_factory=BlockIteratorFactory(True),
        early_stop=False,
        cache_factory=KVCacheFactory('dual'),
        enable_adaptive_moe=False  # 固定模式
    )

    start_time_fixed = time.time()
    res_fixed = dllm_fixed.generate(
        input_ids,
        gen_length=args.gen_length,
        block_length=args.block_length
    )
    end_time_fixed = time.time()
    time_fixed = end_time_fixed - start_time_fixed
    nfe_fixed = dllm_fixed.num_forwards

    tokens_fixed = res_fixed[0, input_ids.shape[1]:]
    text_fixed = tokenizer.decode(tokens_fixed, skip_special_tokens=True)
    print(f"\n🔸 Generated Text (FIXED):")
    print("-" * 30)
    print(f"   {text_fixed}")
    print("-" * 30)

    # 清理固定模式的状态，避免影响自适应模式
    del dllm_fixed, decoder_fixed
    torch.cuda.empty_cache()

    # 2. 运行自适应专家激活
    print(f"\n🔹 Running ADAPTIVE expert activation (initial={args.initial_num_experts}, max={args.max_num_experts})...")
    print("-"*80)

    # 创建解码器和扩散LLM（自适应模式）
    decoder_adaptive = ThresholdParallelDecoder(
        temperature=0.0,
        threshold=args.threshold,
        mask_id=args.mask_id,
        eos_id=args.eos_id
    )
    # decoder_adaptive = FixedParallelDecoder(
    #     temperature=0.0,
    #     steps=args.steps,
    #     mask_id=args.mask_id
    # )

    dllm_adaptive = BlockWiseDiffusionLLM(
        model=model,
        decoder=decoder_adaptive,
        iterator_factory=BlockIteratorFactory(True),
        early_stop=False,
        cache_factory=KVCacheFactory('dual'),
        enable_adaptive_moe=True,  # 启用自适应
        growth_strategy=args.growth_strategy,
        max_num_experts=args.max_num_experts,
        initial_num_experts=args.initial_num_experts,
        update_interval=args.update_interval,
        verbose=args.verbose  # 传递 verbose 参数
    )

    start_time_adaptive = time.time()
    res_adaptive = dllm_adaptive.generate(
        input_ids,
        gen_length=args.gen_length,
        block_length=args.block_length
    )
    end_time_adaptive = time.time()
    time_adaptive = end_time_adaptive - start_time_adaptive
    nfe_adaptive = dllm_adaptive.num_forwards

        
    tokens_adaptive = res_adaptive[0, input_ids.shape[1]:]
    text_adaptive = tokenizer.decode(tokens_adaptive, skip_special_tokens=True)
    print(f"\nGenerated Text (ADAPTIVE):")
    print("-" * 30)
    print(f"   {text_adaptive}")
    print("-" * 30)
    # 3. 计算性能指标
    num_tokens = args.gen_length
    tps_fixed = num_tokens / time_fixed if time_fixed > 0 else 0
    tps_adaptive = num_tokens / time_adaptive if time_adaptive > 0 else 0

    speedup = time_fixed / time_adaptive if time_adaptive > 0 else 0


    print(f"\nPerformance Metrics:")
    print(f"\n  FIXED Mode:")
    print(f"     Tokens generated:  {num_tokens}")
    print(f"     Time taken:        {time_fixed:.2f}s")
    print(f"     NFE:               {nfe_fixed}")
    print(f"     TPS:               {tps_fixed:.2f} tokens/sec")

    print(f"\n  ADAPTIVE Mode (initial={args.initial_num_experts}):")
    print(f"     Tokens generated:  {num_tokens}")
    print(f"     Time taken:        {time_adaptive:.2f}s")
    print(f"     NFE:               {nfe_adaptive}")
    print(f"     TPS:               {tps_adaptive:.2f} tokens/sec")

    print(f"\n  Speedup:           {speedup:.2f}x")
    print(f"  Time reduction:    {(1 - time_adaptive/time_fixed)*100:.1f}%")

    # 4. 打印详细日志（如果启用了 verbose）
    if args.verbose and hasattr(dllm_adaptive.diff_iteration, 'step_logs'):
        print("\n" + "="*80)
        print("📋 Detailed Step-by-Step Logs (ADAPTIVE Mode)")
        print("="*80)

        step_logs = dllm_adaptive.diff_iteration.step_logs
        for i, log in enumerate(step_logs):
            # if i % 8 == 0:
            print("="*40)
            print(f"Step {log['step']}:")
            print("="*40) 
            # 打印专家配置
            if log['num_experts_per_tok_global'] is not None:
                print(f"\n  Expert Configuration in Block [{log['block_range'][0]}:{log['block_range'][1]}]:")
                # 打印当前块的专家配置
                if log['experts_per_tok_in_block'] is not None:
                    experts_in_block = log['experts_per_tok_in_block'][0]  # [block_size]
                    print(f"    num_experts_per_tok (current block): {experts_in_block.tolist()}")
                global_experts = log['num_experts_per_tok_global'][0]  # [total_len]
                prompt_len = input_ids.shape[1]
                
                # 显示prompt和生成部分的专家配置分布
                prompt_experts = global_experts[:prompt_len]
                gen_experts = global_experts[prompt_len:]

                # print(f"    Prompt part experts: {prompt_experts.tolist()}")
                print(f"Generation part experts: {gen_experts.tolist()}")

            # 打印当前序列状态（只显示生成部分）
            # if log['sequence_snapshot'] is not None:
            #     seq = log['sequence_snapshot'][0]  # [total_len]
            #     prompt_len = input_ids.shape[1]
            #     gen_part = seq[prompt_len:]  # 只显示生成部分

            #     # 统计MASK token数量
            #     num_mask = (gen_part == args.mask_id).sum().item()
            #     num_decoded = len(gen_part) - num_mask

            #     print(f"\nSequence Status (Generated part: {num_decoded}/{len(gen_part)} decoded):")
            #     decoded_tokens = [tokenizer.decode([tok], skip_special_tokens=True) if tok != args.mask_id else '[MASK]'
            #                     for tok in gen_part[:].tolist()] 
            #     decoded_str = ' '.join(decoded_tokens)
            #     print(f"    {decoded_str}")
                


def main():
    parser = argparse.ArgumentParser(
        description='Compare Fixed vs Adaptive Expert Activation in dInfer'
    )

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the fused MoE model')
    parser.add_argument('--prompt', type=str,
                        default="Lily can run 12 kilometers per hour for 4 hours. "
                                "After that, she can run 6 kilometers per hour. "
                                "How many kilometers can she run in 8 hours?",
                        help='Input prompt for text generation')

    # Generation arguments
    parser.add_argument('--gen_length', type=int, default=512,
                        help='Length of generated sequence (default: 512)')
    parser.add_argument('--block_length', type=int, default=64,
                        help='Block length for generation (default: 64)')
    parser.add_argument('--steps', type=int, default=128,
                        help='Number of diffusion steps (default: 128)')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Confidence threshold for decoder (default: 0.9)')

    # Special token IDs
    parser.add_argument('--mask_id', type=int, default=156895,
                        help='Mask token ID (default: 156895)')
    parser.add_argument('--eos_id', type=int, default=156892,
                        help='EOS token ID (default: 156892)')

    # Adaptive expert configuration
    parser.add_argument('--growth_strategy', type=str, default='linear',
                        choices=['linear', 'exponential'],
                        help='Expert growth strategy (default: linear)')
    parser.add_argument('--initial_num_experts', type=int, default=1,
                        help='Initial number of experts for MASK tokens (default: 1)')
    parser.add_argument('--max_num_experts', type=int, default=8,
                        help='Maximum number of experts per token (default: 8)')
    parser.add_argument('--update_interval', type=int, default=8,
                        help='Update expert count every N steps (default: 8)')

    # Output control
    parser.add_argument('--show_text', action='store_true',
                        help='Show generated text (default: False)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed step-by-step logging for adaptive mode (default: False)')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on (default: cuda:0)')

    args = parser.parse_args()

    # ========== 步骤1: 环境初始化 ==========
    print("\n" + "="*80)
    print("dInfer Fixed vs Adaptive Expert Activation Comparison")
    print("="*80)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'

    device = torch.device('cuda:0')

    # 初始化分布式环境
    distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    distributed.initialize_model_parallel(1, backend='nccl')

    # ========== 步骤2: 加载模型 ==========
    print(f"\nLoading model from: {args.model_path}")

    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        model = FusedOlmoeForCausalLM(config=model_config).eval()
        model.load_weights(args.model_path, torch_dtype=torch.bfloat16)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # ========== 步骤3: 准备输入 ==========
    print(f"\nPrompt: {args.prompt}")

    m = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_text)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    print(f"\nConfiguration:")
    print(f"   Generation length:    {args.gen_length}")
    print(f"   Block length:         {args.block_length}")
    print(f"   Diffusion steps:      {args.steps}")
    print(f"   Threshold:            {args.threshold}")
    print(f"   Growth strategy:      {args.growth_strategy}")
    print(f"   Initial experts:      {args.initial_num_experts}")
    print(f"   Max experts:          {args.max_num_experts}")
    print(f"   Update interval:      {args.update_interval}")

    # ========== 步骤4: 运行比较 ==========
    compare_fixed_vs_adaptive(model, tokenizer, input_ids, args)


if __name__ == "__main__":
    main()
