import os
import torch
import time
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config

from dinfer.model import LLaDAMoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import MCMCThresholdParallelDecoder, BlockMCMCDiffusionLLM

print("========== 步骤1: 模型路径和Tokenizer加载 ==========")
m = "/home/zhounan/models/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused"
tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)

print("========== 步骤2: 设备和分布式环境初始化 ==========")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
device = torch.device('cuda:0')
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12346'

distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
distributed.initialize_model_parallel(1, backend='nccl')

print("========== 步骤3: 配置并行策略和加载模型 ==========")
parallel_config = ParallelConfig(enable_expert_parallel=True)
with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
    model_config = AutoConfig.from_pretrained(m, trust_remote_code=True)
    model = LLaDAMoeModelLM(config=model_config).eval()
    model.load_weights(m, torch_dtype=torch.bfloat16)
    model = model.to(device)

print("========== 步骤4: 配置解码器和MCMC扩散LLM ==========")
decoder = MCMCThresholdParallelDecoder(0.9, threshold=0.9, mask_id=156895, eos_id=156892)

dllm = BlockMCMCDiffusionLLM(
    model, decoder, BlockIteratorFactory(True),
    cache_factory=None,  # 不使用KV Cache (简化版本)
    enable_mcmc=True,
    n_mcmc_steps=2,
    mcmc_alpha=4.0,
    mcmc_temperature=0.9,
    tokenizer=tokenizer,
    verbose=True
)

print("========== 步骤5: 准备输入和生成 ==========")
prompt = "The vending machine sells drinks for 80 cents each. However, it gives you a 20-cent refund for each empty bottle you return. James has 2 dollars (200 cents). Assuming he can buy a drink, drink it, and immediately return the bottle for the refund (and repeat), how many drinks can he drink in total?"
m = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(prompt)['input_ids']
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

print("========== 步骤6: 执行生成（带MCMC精炼） ==========")
start_time = time.time()
res = dllm.generate(input_ids, gen_length=256, block_length=32)
end_time = time.time()
print(f"生成耗时: {end_time - start_time:.2f} 秒")

print("========== 步骤7: 解码并输出结果 ==========")
print(f"DEBUG: res.shape = {res.shape}")
print(f"DEBUG: input_ids.shape = {input_ids.shape}")
print(f"DEBUG: res[0] first 10 tokens = {res[0, :10].tolist()}")
print(f"DEBUG: res[0] last 10 tokens = {res[0, -10:].tolist()}")
generated_part = res[0, input_ids.shape[1]:]
print(f"DEBUG: generated_part.shape = {generated_part.shape}")
print(f"DEBUG: generated_part first 20 tokens = {generated_part[:20].tolist()}")
print(f"DEBUG: Number of mask tokens (156895) in generated: {(generated_part == 156895).sum().item()}")
print(f"DEBUG: Number of eos tokens (156892) in generated: {(generated_part == 156892).sum().item()}")
print(f"DEBUG: Unique token count in generated: {torch.unique(generated_part).shape[0]}")

decoded_output = tokenizer.decode(res[0, input_ids.shape[1]:], skip_special_tokens=False)
print(f"\n生成的文本 (长度={len(decoded_output)}):")
print(decoded_output)
