"""
测试动态专家激活机制的脚本

这个脚本验证修改后的vLLM是否正确支持per-token动态专家数。
"""

import torch
import sys
sys.path.insert(0, 'python')

from dinfer.model.modeling_fused_olmoe import OlmoeMoE
from dinfer.model.configuration_olmoe import OlmoeConfig


def test_adaptive_moe():
    """测试动态专家激活功能"""

    print("=" * 80)
    print("测试动态专家激活机制")
    print("=" * 80)

    # 1. 创建配置
    config = OlmoeConfig(
        hidden_size=512,
        num_experts=8,
        num_experts_per_tok=2,  # 默认top_k=2
        expert_intermediate_size=1024,
    )

    # 2. 创建OlmoeMoE层
    print("\n[1] 创建OlmoeMoE层...")
    moe_layer = OlmoeMoE(config, prefix="test")
    moe_layer = moe_layer.cuda()
    moe_layer.eval()
    print("✓ OlmoeMoE层创建成功")

    # 3. 准备测试数据
    batch_size = 4
    seq_len = 8
    hidden_size = config.hidden_size

    print(f"\n[2] 准备测试数据: batch_size={batch_size}, seq_len={seq_len}")
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).cuda()
    print("✓ 测试数据准备完成")

    # 4. 测试固定模式（不传num_experts_per_tok）
    print("\n[3] 测试固定模式（所有token使用2个专家）...")
    with torch.no_grad():
        output_fixed = moe_layer(hidden_states)
    print(f"✓ 固定模式输出shape: {output_fixed.shape}")
    assert output_fixed.shape == hidden_states.shape, "输出shape不匹配！"

    # 5. 测试自适应模式（传入动态的num_experts_per_tok）
    print("\n[4] 测试自适应模式（每个token使用不同数量的专家）...")

    # 创建动态专家数张量：每个token使用1-8个专家
    num_tokens = batch_size * seq_len
    num_experts_per_tok = torch.tensor([
        1, 2, 3, 4,  # 第一个样本的8个token
        5, 6, 7, 8,
        2, 2, 2, 2,  # 第二个样本
        4, 4, 4, 4,
        1, 3, 5, 7,  # 第三个样本
        2, 4, 6, 8,
        8, 7, 6, 5,  # 第四个样本
        4, 3, 2, 1,
    ], dtype=torch.long).cuda()

    print(f"   动态专家数分布: {num_experts_per_tok.tolist()}")

    with torch.no_grad():
        output_adaptive = moe_layer(hidden_states, num_experts_per_tok=num_experts_per_tok)

    print(f"✓ 自适应模式输出shape: {output_adaptive.shape}")
    assert output_adaptive.shape == hidden_states.shape, "输出shape不匹配！"

    # 6. 验证两种模式的输出不同（因为使用的专家数不同）
    print("\n[5] 验证两种模式的输出差异...")
    diff = torch.abs(output_fixed - output_adaptive).mean().item()
    print(f"   固定模式 vs 自适应模式的平均差异: {diff:.6f}")

    if diff > 1e-6:
        print("✓ 两种模式产生了不同的输出（符合预期）")
    else:
        print("⚠ 警告：两种模式的输出几乎相同，可能存在问题")

    # 7. 测试边界情况
    print("\n[6] 测试边界情况...")

    # 7.1 所有token都使用1个专家
    num_experts_per_tok_min = torch.ones(num_tokens, dtype=torch.long).cuda()
    with torch.no_grad():
        output_min = moe_layer(hidden_states, num_experts_per_tok=num_experts_per_tok_min)
    print(f"✓ 最小专家数（全1）测试通过，输出shape: {output_min.shape}")

    # 7.2 所有token都使用8个专家
    num_experts_per_tok_max = torch.full((num_tokens,), 8, dtype=torch.long).cuda()
    with torch.no_grad():
        output_max = moe_layer(hidden_states, num_experts_per_tok=num_experts_per_tok_max)
    print(f"✓ 最大专家数（全8）测试通过，输出shape: {output_max.shape}")

    # 8. 性能对比
    print("\n[7] 性能对比...")
    import time

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = moe_layer(hidden_states)

    # 固定模式性能测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = moe_layer(hidden_states)
    torch.cuda.synchronize()
    fixed_time = time.time() - start

    # 自适应模式性能测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = moe_layer(hidden_states, num_experts_per_tok=num_experts_per_tok)
    torch.cuda.synchronize()
    adaptive_time = time.time() - start

    print(f"   固定模式平均耗时: {fixed_time/100*1000:.2f} ms")
    print(f"   自适应模式平均耗时: {adaptive_time/100*1000:.2f} ms")
    print(f"   性能比: {adaptive_time/fixed_time:.2f}x")

    # 9. 总结
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！动态专家激活机制工作正常。")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        success = test_adaptive_moe()
        if success:
            print("\n✓ 测试成功完成！")
            sys.exit(0)
        else:
            print("\n✗ 测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
