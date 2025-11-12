"""
dInfer推理示例脚本

此脚本演示如何使用dInfer框架进行LLaDA-MoE模型的推理。
包含模型加载、分布式初始化、解码器配置和生成过程。

依赖项:
- transformers: HuggingFace模型加载
- vllm: 分布式并行和配置管理
- dinfer: dInfer核心组件(模型、解码器、缓存管理)
"""

import os
import torch
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config


# - FusedOlmoeForCausalLM: 来自 dinfer.model.__init__.py
# - BlockIteratorFactory, KVCacheFactory: 来自 dinfer.decoding.utils
# - ThresholdParallelDecoder, BlockWiseDiffusionLLM: 来自 dinfer.decoding
from dinfer.model import FusedOlmoeForCausalLM  # 融合MoE模型
from dinfer import BlockIteratorFactory, KVCacheFactory  # 迭代器和缓存工厂
from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM  # 解码器和扩散LLM

# ========== 步骤1: 模型路径和Tokenizer加载 ==========
# 模型路径: 使用transfer.py转换后的FusedMoE格式模型
m = "/home/shenyl/hf/model/inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused"
tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)

# ========== 步骤2: 设备和分布式环境初始化 ==========
# 设置CUDA设备顺序以避免设备索引问题
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# 只使用GPU 0，避免访问不存在的设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')  # 使用GPU 0
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12346'

# 初始化分布式环境
# 参数: (world_size, rank, init_method, local_rank, backend)
distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
# 初始化模型并行(Tensor Parallel, TP=1)
distributed.initialize_model_parallel(1, backend='nccl')

# ========== 步骤3: 配置并行策略和加载模型 ==========
# 启用专家并行(Expert Parallel, EP)以提高MoE模型性能
parallel_config = ParallelConfig(enable_expert_parallel=True)
with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
    # 加载模型配置
    model_config = AutoConfig.from_pretrained(m, trust_remote_code=True)
    # 创建FusedMoE模型实例(评估模式)
    model = FusedOlmoeForCausalLM(config=model_config).eval()
    # 加载模型权重(使用bfloat16精度)
    model.load_weights(m, torch_dtype=torch.bfloat16)
    # 将模型移动到指定设备
    model = model.to(device)

# ========== 步骤4: 配置解码器和扩散LLM ==========
# 创建阈值并行解码器
# 参数: (rank, threshold, mask_id, eos_id)
# - threshold=0.9: 置信度阈值,高于此值的token被接受
# - mask_id=156895: 掩码token ID
# - eos_id=156892: 结束token ID
decoder = ThresholdParallelDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892)

# 创建块级扩散LLM
# 参数:
# - model: FusedMoE模型
# - decoder: 并行解码器
# - BlockIteratorFactory(True): 块迭代器工厂(启用软扩散)
# - cache_factory: KV缓存工厂('dual'=双缓存策略)
dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(True), cache_factory=KVCacheFactory('dual'))

# ========== 步骤5: 准备输入和生成 ==========
# 测试问题: 数学推理任务
prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours?"
# 构建对话格式消息
m = [{"role": "user", "content": prompt}, ]
# 应用聊天模板(添加生成提示符)
prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
# Tokenize输入文本
input_ids = tokenizer(prompt)['input_ids']
# 转换为张量并移动到设备,添加batch维度
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

# ========== 步骤6: 执行生成 ==========
# 生成参数:
# - gen_length=1024: 最大生成长度
# - block_length=64: 块长度(扩散块大小)
res = dllm.generate(input_ids, gen_length=1024, block_length=64)

# ========== 步骤7: 解码并输出结果 ==========
# 只解码生成的部分(跳过输入部分)
# skip_special_tokens=False: 保留特殊token以便调试
print(tokenizer.decode(res[0, input_ids.shape[1]:], skip_special_tokens=False))