# generate_uniform.py 动态专家激活修改

## 修改目的
为 `BlockWiseDiffusionLLM` 类添加动态专家激活机制支持，实现 per-token 动态专家数调整。

## 主要修改

### 1. 新增 `update_expert_count` 函数 (L12-43)

```python
def update_expert_count(
    remask_count: torch.Tensor,
    num_experts_per_tok: torch.Tensor,
    growth_strategy: str,
    max_num_experts: int,
    initial_num_experts: int
) -> torch.Tensor
```

**功能**: 根据 remask 次数和增长策略更新每个 token 的专家数

**增长策略**:
- `linear`: `initial_num_experts + remask_count` (例如: 1→2→3→4)
- `exponential`: `initial_num_experts * (2^remask_count)` (例如: 1→2→4→8)

### 2. 新增 `AdaptiveMoEDiffusionIteration` 类 (L334-434)

继承自 `BaseDiffusionIteration`，专门处理动态专家激活。

**核心特性**:
- 在每次 `model()` 调用时传入 `num_experts_per_tok` 参数
- 追踪 `remask_count` (每个 token 被 remask 的次数)
- 每 N 步自动更新专家数 (默认 8 步)
- 识别已解码 token 并更新相应位置的专家数

**关键方法**:
```python
def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
    # 1. 调用 model 时传入 num_experts_per_tok
    logits = model(x.data, num_experts_per_tok=self.num_experts_per_tok).logits

    # 2. 识别被解码的 token
    transfer_index = (old_block == mask_id) & (new_block != mask_id)

    # 3. 每 update_interval 步更新专家数
    if self.num_forwards % self.update_interval == 0:
        still_mask_block = (new_block == mask_id) & (~transfer_index)
        self.remask_count[still_mask_indices] += 1
        self.num_experts_per_tok[still_mask_indices] = update_expert_count(...)
```

### 3. 修改 `BlockWiseDiffusionLLM.__init__` (L589-611)

**新增参数**:
- `enable_adaptive_moe`: 是否启用动态专家激活 (默认 False)
- `growth_strategy`: 专家增长策略 (默认 'linear')
- `max_num_experts`: 最大专家数 (默认 8)
- `initial_num_experts`: 初始专家数 (默认 1)
- `update_interval`: 更新间隔 (默认 8 步)

**逻辑**:
```python
if enable_adaptive_moe:
    self.diff_iteration = AdaptiveMoEDiffusionIteration(...)
elif use_shift:
    self.diff_iteration = ShiftDiffusionIteration()
else:
    self.diff_iteration = BaseDiffusionIteration()
```

### 4. 修改 `BlockWiseDiffusionLLM.generate` (L622-663)

**初始化动态专家激活**:
```python
if self.enable_adaptive_moe:
    # 初始化 num_experts_per_tok: [B, L]
    # - prompt 部分: max_num_experts
    # - MASK 部分: initial_num_experts
    num_experts_per_tok = torch.full((B, L), initial_num_experts, ...)
    num_experts_per_tok[:, :prompt_len] = max_num_experts

    # 初始化 remask_count: [B, L]
    remask_count = torch.zeros(B, L, ...)

    # 设置到 diff_iteration
    self.diff_iteration.num_experts_per_tok = num_experts_per_tok
    self.diff_iteration.remask_count = remask_count
```

## 使用示例

```python
# 不启用动态专家激活 (默认行为)
llm = BlockWiseDiffusionLLM(
    model=model,
    decoder=decoder,
    iterator_factory=iterator_factory
)

# 启用动态专家激活 - 线性增长策略
llm = BlockWiseDiffusionLLM(
    model=model,
    decoder=decoder,
    iterator_factory=iterator_factory,
    enable_adaptive_moe=True,
    growth_strategy='linear',
    max_num_experts=8,
    initial_num_experts=1,
    update_interval=8
)

# 启用动态专家激活 - 指数增长策略
llm = BlockWiseDiffusionLLM(
    model=model,
    decoder=decoder,
    iterator_factory=iterator_factory,
    enable_adaptive_moe=True,
    growth_strategy='exponential',
    max_num_experts=8,
    initial_num_experts=2,
    update_interval=4
)

# 生成
output = llm.generate(prompt, gen_length=128, block_length=128)
```

## 参数传递链路

```
BlockWiseDiffusionLLM.generate()
  ↓ 初始化 num_experts_per_tok, remask_count
AdaptiveMoEDiffusionIteration.forward()
  ↓ num_experts_per_tok 传递
model(x, num_experts_per_tok=num_experts_per_tok)
  ↓
FusedOlmoeForCausalLM.forward(num_experts_per_tok=...)
  ↓
OlmoeModel.forward(num_experts_per_tok=...)
  ↓
OlmoeDecoderLayer.forward(num_experts_per_tok=...)
  ↓
OlmoeMoE.forward(num_experts_per_tok=...)
  ↓
FusedMoE.forward_impl(num_experts_per_tok=...)
  ↓
adaptive_topk(num_experts_per_tok=...)  # vLLM 底层实现
```

## 兼容性

- ✅ **向后兼容**: 默认 `enable_adaptive_moe=False`，不影响现有代码
- ✅ **可选功能**: 通过参数控制是否启用
- ✅ **灵活配置**: 支持不同的增长策略和更新间隔

## 关键设计决策

1. **继承方式**: `AdaptiveMoEDiffusionIteration` 继承 `BaseDiffusionIteration`，重用大部分逻辑
2. **参数传递**: 通过实例属性 (`self.num_experts_per_tok`) 在 iteration 和 generate 之间共享状态
3. **更新时机**: 每 N 步更新一次，避免过于频繁的更新影响性能
4. **Prompt 处理**: Prompt 部分使用最大专家数，确保输入编码质量
