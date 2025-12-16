import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed as vllm_dist
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

from dinfer.model import LLaDAMoeModelLM, LLaDA2MoeModelLM, LLaDAModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM

from data_utils import CTDDataset, SudokuDataset

DATASET_MAP = {
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.total_size = len(self.dataset)
        self.num_samples = len(self.dataset) // self.num_replicas + int(
            rank < (self.total_size % self.num_replicas)
        )
        self.shuffle = shuffle
        self.seed = seed


def evaluate(dllm, tokenizer, dataloader, gen_length, block_length, device, rank):
    wall_times = []
    all_generations = []
    total_forwards = 0

    for batch in tqdm(dataloader, disable=(rank != 0)):
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        start_time = time.time()
        prev_forwards = dllm.num_forwards
        out = dllm.generate(input_ids, gen_length=gen_length, block_length=block_length)
        nfe = dllm.num_forwards - prev_forwards
        total_forwards += nfe
        wall_times.append(time.time() - start_time)

        generated_texts = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=False)

        for j in range(len(gt_answers)):
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            })

        if rank == 0:
            idx = 0
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print(f"Generation: {generated_texts[idx][:500]}...")
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")
            print(f"NFE: {nfe}, Time: {wall_times[-1]:.2f}s")

    return {
        "wall_time": sum(wall_times) / len(wall_times) if wall_times else 0,
        "generations": all_generations,
        "total_processed": len(all_generations),
        "total_forwards": total_forwards,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["countdown", "sudoku"], required=True)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--subsample", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--cache", type=str, default="dual", choices=["", "prefix", "dual"])
    args = parser.parse_args()

    local_rank = setup_ddp()
    init_seed(42)

    device = torch.device(f'cuda:{local_rank}')

    # Auto-detect mask_id/eos_id based on model path
    model_path_lower = args.model_path.lower()
    if "moe" in model_path_lower or "mini" in model_path_lower:
        mask_id = 156895
        eos_id = 156892
        is_moe = True
    else:
        mask_id = 126336
        eos_id = 126081
        is_moe = False

    if local_rank == 0:
        print(f"Model: {args.model_path}")
        print(f"is_moe: {is_moe}, mask_id: {mask_id}, eos_id: {eos_id}")

    # Initialize vllm distributed for MoE models
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    vllm_port = str(45600 + local_rank)
    os.environ['MASTER_PORT'] = vllm_port
    vllm_dist.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    vllm_dist.initialize_model_parallel(1, backend='nccl')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load model
    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        if 'moe' in model_path_lower:
            model = LLaDAMoeModelLM(config=config).eval()
        elif 'mini' in model_path_lower:
            model = LLaDA2MoeModelLM(config=config).eval()
        else:
            model = LLaDAModelLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).eval()

        if is_moe:
            model.load_weights(args.model_path, torch_dtype=torch.bfloat16)
        model = model.to(device)

        # Create decoder and dllm
        decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=mask_id, eos_id=eos_id)
        cache_factory = KVCacheFactory(args.cache) if args.cache else None
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True),
                                      cache_factory=cache_factory, early_stop=True)

        # Load dataset
        dataset = DATASET_MAP[args.dataset](tokenizer, subsample=args.subsample, add_reasoning=True, data_dir=args.data_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Fixed to 1 for dInfer
            sampler=CustomDistributedSampler(dataset, shuffle=False),
            collate_fn=dataset.collate_fn,
        )

        # Evaluate
        metrics = evaluate(dllm, tokenizer, dataloader, args.gen_length, args.block_length, device, local_rank)

        # Save results (only rank 0)
        if local_rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            model_name = os.path.basename(os.path.normpath(args.model_path))
            if args.suffix:
                model_name = f"{model_name}_{args.suffix}"
            filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.block_length}_{local_rank}_generations.json"

            with open(filename, "w") as f:
                json.dump({
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                        "total_forwards": metrics["total_forwards"],
                    },
                    "args": vars(args),
                }, f, indent=2)

            print(f"Results saved to {filename}")
            print(f"Avg wall time: {metrics['wall_time']:.2f}s, Total forwards: {metrics['total_forwards']}")

    cleanup_ddp()
