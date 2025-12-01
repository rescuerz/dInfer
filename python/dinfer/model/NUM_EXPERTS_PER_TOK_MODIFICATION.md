# modeling_fused_olmoe.py 动态专家激活修改

## 目的
支持 per-token 动态专家数，参数 `num_experts_per_tok: Optional[torch.Tensor]` 从顶层传递到底层 MoE 实现。

## 传递链路

```python
FusedOlmoeForCausalLM.forward()         # L1232: 接收参数
  ↓ num_experts_per_tok (L1288)
OlmoeModel.forward()                    # L997: 接收参数
  ↓ num_experts_per_tok (L1059)
OlmoeDecoderLayer.forward()             # L746: 接收参数 ✅ 修改点
  ↓ num_experts_per_tok (L803)
OlmoeMoE.forward()                      # L697: 接收参数
  ↓ num_experts_per_tok (L713-716)
FusedMoE.forward_impl()                 # 最终使用
```

## 维度处理 (L705-706)

```python
# OlmoeMoE.forward() 中自动展平
if num_experts_per_tok is not None and num_experts_per_tok.dim() > 1:
    num_experts_per_tok = num_experts_per_tok.view(-1)  # [B,L] → [B*L]
```
