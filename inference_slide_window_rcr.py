"""
Inference script for SlideWindowRCRDecoder

This script demonstrates how to use the SlideWindowRCRDecoder with BlockWiseDiffusionLLM.
"""
import os
import torch
import time
import argparse
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config

from dinfer.model import LLaDAMoeModelLM 
from dinfer import BlockIteratorFactory, KVCacheFactory  
from dinfer import ThresholdParallelDecoder, SlideWindowRCRDecoder, BlockWiseDiffusionLLM  


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with SlideWindowRCRDecoder')
    parser.add_argument('--model_path', type=str, 
                        default="/home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused",
                        help='Path to the model')
    parser.add_argument('--decoder', type=str, default='slide_window_rcr',
                        choices=['threshold', 'slide_window_rcr'],
                        help='Decoder type to use')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='High confidence threshold for transfer')
    parser.add_argument('--medium_threshold', type=float, default=0.8,
                        help='Medium confidence threshold (for slide_window_rcr)')
    parser.add_argument('--low_threshold', type=float, default=0.62,
                        help='Low confidence threshold for direct remask')
    parser.add_argument('--window_size', type=int, default=3,
                        help='Sliding window size for confidence history')
    parser.add_argument('--decline_threshold', type=float, default=0.1,
                        help='Threshold for confidence decline detection')
    parser.add_argument('--gen_length', type=int, default=256,
                        help='Maximum generation length')
    parser.add_argument('--block_length', type=int, default=64,
                        help='Block length for diffusion')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='Temperature for sampling')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt (optional)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("========== Step 1: Load Tokenizer ==========")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Get special token IDs from tokenizer
    mask_id = tokenizer.convert_tokens_to_ids('<|mask|>') if '<|mask|>' in tokenizer.get_vocab() else 156895
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 156892
    print(f"mask_id: {mask_id}, eos_id: {eos_id}")

    print("========== Step 2: Initialize Distributed Environment ==========")
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    device = torch.device('cuda:0')  
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'

    distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    distributed.initialize_model_parallel(1, backend='nccl')

    print("========== Step 3: Load Model ==========")
    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=model_config).eval()
        model.load_weights(args.model_path, torch_dtype=torch.bfloat16)
        model = model.to(device)

    print("========== Step 4: Configure Decoder ==========")
    if args.decoder == 'threshold':
        print(f"Using ThresholdParallelDecoder with threshold={args.threshold}")
        decoder = ThresholdParallelDecoder(
            temperature=args.temperature, 
            threshold=args.threshold, 
            mask_id=mask_id, 
            eos_id=eos_id
        )
    elif args.decoder == 'slide_window_rcr':
        print(f"Using SlideWindowRCRDecoder with:")
        print(f"  threshold={args.threshold}")
        print(f"  medium_threshold={args.medium_threshold}")
        print(f"  low_threshold={args.low_threshold}")
        print(f"  window_size={args.window_size}")
        print(f"  decline_threshold={args.decline_threshold}")
        decoder = SlideWindowRCRDecoder(
            temperature=args.temperature,
            threshold=args.threshold,
            medium_threshold=args.medium_threshold,
            low_threshold=args.low_threshold,
            window_size=args.window_size,
            decline_threshold=args.decline_threshold,
            mask_id=mask_id,
            eos_id=eos_id,
            debug=True,  # Enable debug mode
            tokenizer=tokenizer  # Pass tokenizer for debug output
        )

    print("========== Step 5: Create Diffusion LLM ==========")
    dllm = BlockWiseDiffusionLLM(
        model, 
        decoder, 
        BlockIteratorFactory(True), 
        cache_factory=KVCacheFactory('dual')
    )

    print("========== Step 6: Prepare Input ==========")
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "The vending machine sells drinks for 80 cents each. However, it gives you a 20-cent refund for each empty bottle you return. James has 2 dollars (200 cents). Assuming he can buy a drink, drink it, and immediately return the bottle for the refund (and repeat), how many drinks can he drink in total?"
    
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_text)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    print(f"Prompt: {prompt}")
    print(f"Input length: {input_ids.shape[1]} tokens")

    print("========== Step 7: Generate ==========")
    # Reset statistics before generation
    if args.decoder == 'slide_window_rcr':
        decoder.reset_stats()
    
    start_time = time.time()
    res = dllm.generate(input_ids, gen_length=args.gen_length, block_length=args.block_length)
    end_time = time.time()
    
    generation_time = end_time - start_time
    num_generated = res.shape[1] - input_ids.shape[1]
    tokens_per_second = num_generated / generation_time if generation_time > 0 else 0
    
    print(f"\nGeneration time: {generation_time:.2f} seconds")
    print(f"Generated tokens: {num_generated}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Number of forward passes: {dllm.num_forwards}")
    
    # Print remask statistics for SlideWindowRCRDecoder
    if args.decoder == 'slide_window_rcr':
        stats = decoder.get_stats()
        print(f"Remask statistics:")
        print(f"  Low confidence remask (conf < {args.low_threshold}): {stats['remask_low_conf_count']}")
        print(f"  Declining threshold remask (decline > {args.decline_threshold}): {stats['remask_declining_count']}")
        print(f"  Declining consecutive remask (every step decreasing): {stats['remask_consecutive_declining_count']}")
        print(f"  Total remask count: {stats['total_remask_count']}")

    print("========== Step 8: Decode Output ==========")
    output_text = tokenizer.decode(res[0, input_ids.shape[1]:], skip_special_tokens=False)
    print("\n--- Generated Output ---")
    print(output_text)


if __name__ == '__main__':
    main()
