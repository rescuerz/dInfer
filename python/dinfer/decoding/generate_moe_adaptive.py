"""
自适应专家激活的生成函数 (dInfer版本)

该模块实现了基于dInfer框架的自适应专家激活策略:
- 根据token的remask历史动态调整激活的专家数量
- 支持多种增长策略(线性、指数)
- 支持多种已解码token处理方案
- 与dInfer的BlockWiseDiffusionLLM集成
"""

import torch
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class AdaptiveExpertConfig:
    """自适应专家激活的配置类"""

    # 增长策略: 'linear' (1→2→3→4...→8) 或 'exponential' (1→2→4→8)
    growth_strategy: Literal['linear', 'exponential'] = 'linear'

    # 已解码token的专家数处理: 'keep' (保持不变), 'reset_to_one' (重置为1), 'reset_to_max' (重置为8)
    decoded_token_strategy: Literal['keep', 'reset_to_one', 'reset_to_max'] = 'keep'

    # prompt部分的专家数
    prompt_num_experts: int = 8

    # MASK token初始专家数
    initial_num_experts: int = 1

    # 最大专家数
    max_num_experts: int = 8

    # 每隔多少步更新一次专家数 (对应llada_moe中的8步更新)
    update_interval: int = 8

    # 是否启用调试日志
    debug: bool = False


def update_expert_count(
    remask_count: torch.Tensor,
    growth_strategy: str,
    max_num_experts: int,
    initial_num_experts: int
) -> torch.Tensor:
    """
    根据remask次数和增长策略更新每个token的专家数

    Args:
        remask_count: 每个token被remask的次数, shape [B, L]
        growth_strategy: 增长策略 'linear' 或 'exponential'
        max_num_experts: 最大专家数
        initial_num_experts: 初始专家数

    Returns:
        更新后的专家数, shape [B, L]
    """
    if growth_strategy == 'linear':
        # 线性增长: initial_num_experts + remask_count
        # 例如：initial=1时，1→2→3→4→...→8
        new_experts = initial_num_experts + remask_count
    elif growth_strategy == 'exponential':
        # 指数增长: initial_num_experts * (2^remask_count)
        # 例如：initial=1时，1→2→4→8
        new_experts = initial_num_experts * (2 ** remask_count)
    else:
        raise ValueError(f"Unknown growth strategy: {growth_strategy}")

    # 限制在最大专家数内
    new_experts = torch.clamp(new_experts, min=initial_num_experts, max=max_num_experts)

    return new_experts.long()


class AdaptiveExpertManager:
    """
    自适应专家激活管理器

    负责在生成过程中管理每个token的专家数量，并根据remask历史动态调整。
    """

    def __init__(self, config: AdaptiveExpertConfig):
        """
        初始化管理器

        Args:
            config: 自适应专家配置
        """
        self.config = config
        self.num_experts_per_tok = None
        self.remask_count = None
        self.prompt_len = None
        self.step_counter = 0

    def initialize(self, batch_size: int, total_len: int, prompt_len: int, device: torch.device):
        """
        初始化专家数矩阵和remask计数器

        Args:
            batch_size: 批次大小
            total_len: 总序列长度 (prompt + generation)
            prompt_len: prompt长度
            device: 设备
        """
        self.prompt_len = prompt_len
        self.step_counter = 0

        # 初始化专家数矩阵: [B, L]
        # prompt部分: config.prompt_num_experts
        # MASK部分: config.initial_num_experts
        self.num_experts_per_tok = torch.full(
            (batch_size, total_len),
            self.config.initial_num_experts,
            dtype=torch.long,
            device=device
        )
        self.num_experts_per_tok[:, :prompt_len] = self.config.prompt_num_experts

        # remask_count: 记录每个token被remask的次数 [B, L]
        self.remask_count = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)

    def get_num_experts_per_tok(self) -> torch.Tensor:
        """
        获取当前的专家数矩阵

        Returns:
            num_experts_per_tok: [B, L] 每个token的专家数
        """
        return self.num_experts_per_tok

    def update(self, mask_index: torch.Tensor, transfer_index: torch.Tensor):
        """
        在每个生成步骤后更新专家数

        Args:
            mask_index: 当前MASK位置, shape [B, L]
            transfer_index: 本步解码的位置, shape [B, L]
        """
        self.step_counter += 1

        # 每隔update_interval步更新一次专家数
        if self.step_counter % self.config.update_interval == 0:
            # 1. 所有当前仍为MASK的token (除了刚被解码的), remask_count + 1
            still_mask = mask_index & (~transfer_index)
            self.remask_count[still_mask] += 1

            # 2. 根据增长策略更新这些token的专家数
            self.num_experts_per_tok[still_mask] = update_expert_count(
                self.remask_count[still_mask],
                self.config.growth_strategy,
                self.config.max_num_experts,
                self.config.initial_num_experts
            )

        # 3. 处理刚被解码的token的专家数
        if self.config.decoded_token_strategy == 'keep':
            # 保持不变 (不做任何操作)
            pass
        elif self.config.decoded_token_strategy == 'reset_to_one':
            # 重置为1
            self.num_experts_per_tok[transfer_index] = 1
            self.remask_count[transfer_index] = 0
        elif self.config.decoded_token_strategy == 'reset_to_max':
            # 重置为最大值
            self.num_experts_per_tok[transfer_index] = self.config.max_num_experts
            self.remask_count[transfer_index] = 0

        # 调试输出
        if self.config.debug and self.step_counter % 10 == 0:
            prompt_index = torch.zeros_like(mask_index)
            prompt_index[:, :self.prompt_len] = True
            print(f"Step {self.step_counter}:")
            print(f"  Remaining MASK: {mask_index.sum().item()}")
            print(f"  Decoded this step: {transfer_index.sum().item()}")
            print(f"  Expert count stats - min: {self.num_experts_per_tok[~prompt_index].min().item()}, "
                  f"max: {self.num_experts_per_tok[~prompt_index].max().item()}, "
                  f"mean: {self.num_experts_per_tok[~prompt_index].float().mean().item():.2f}")

    def get_statistics(self) -> dict:
        """
        获取统计信息

        Returns:
            统计字典，包含平均专家数、最小/最大专家数等
        """
        if self.num_experts_per_tok is None:
            return {}

        # 只统计生成部分（排除prompt）
        gen_experts = self.num_experts_per_tok[:, self.prompt_len:]

        return {
            'avg_experts': gen_experts.float().mean().item(),
            'min_experts': gen_experts.min().item(),
            'max_experts': gen_experts.max().item(),
            'total_steps': self.step_counter,
        }
