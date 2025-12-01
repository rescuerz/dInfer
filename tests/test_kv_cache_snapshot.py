"""
KV Cache 快照/回滚机制的单元测试

测试 KVCacheSnapshot 类的功能：
1. 保存和恢复 KV Cache 状态
2. 部分区域的快照
3. 边界情况处理
"""

import torch
import pytest
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from dinfer.decoding.utils import (
    KVCache, KVCacheSnapshot, DiffusionKVCacheManager, KVCacheFactory
)


class TestKVCacheSnapshot:
    """KVCacheSnapshot 单元测试"""
    
    @pytest.fixture
    def mock_kv_cache_manager(self):
        """创建模拟的 KV Cache 管理器"""
        # 创建模拟的 past_key_values
        # 形状: [num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]
        num_layers = 2
        batch_size = 1
        num_heads = 4
        seq_len = 64
        hidden_dim = 32
        
        # 创建随机 KV 数据
        kv_data = torch.randn(num_layers, 2, batch_size, num_heads, seq_len, hidden_dim)
        
        # 创建 KV Cache 管理器
        manager = DiffusionKVCacheManager(cache_type='dual', backend='vllm')
        
        # 手动设置 past_key_values
        manager.past_key_values = KVCache.__new__(KVCache)
        manager.past_key_values._data = kv_data
        
        return manager, kv_data.clone()
    
    def test_snapshot_save_restore(self, mock_kv_cache_manager):
        """测试快照保存和恢复的正确性"""
        manager, original_data = mock_kv_cache_manager
        
        # 创建快照
        block_start, block_end = 10, 30
        snapshot = KVCacheSnapshot(block_start, block_end)
        snapshot.save(manager)
        
        assert snapshot.is_saved, "快照应该已保存"
        
        # 修改 KV Cache
        manager.past_key_values._data[:, :, :, :, block_start:block_end, :] = 999.0
        
        # 验证修改生效
        modified_region = manager.past_key_values._data[:, :, :, :, block_start:block_end, :]
        assert torch.all(modified_region == 999.0), "KV Cache 应该已被修改"
        
        # 恢复快照
        snapshot.restore(manager)
        
        # 验证恢复成功
        restored_region = manager.past_key_values._data[:, :, :, :, block_start:block_end, :]
        original_region = original_data[:, :, :, :, block_start:block_end, :]
        assert torch.allclose(restored_region, original_region), "KV Cache 应该已恢复到原始状态"
    
    def test_snapshot_partial_region(self, mock_kv_cache_manager):
        """测试只保存部分区域的快照"""
        manager, original_data = mock_kv_cache_manager
        
        # 只保存一小部分区域
        block_start, block_end = 20, 25
        snapshot = KVCacheSnapshot(block_start, block_end)
        snapshot.save(manager)
        
        # 验证快照数据的形状
        expected_length = block_end - block_start
        assert snapshot.snapshot_data.shape[4] == expected_length, \
            f"快照长度应该是 {expected_length}，实际是 {snapshot.snapshot_data.shape[4]}"
    
    def test_snapshot_boundary_cases(self, mock_kv_cache_manager):
        """测试边界情况"""
        manager, original_data = mock_kv_cache_manager
        seq_len = original_data.shape[4]
        
        # 测试超出边界的情况
        snapshot = KVCacheSnapshot(seq_len - 5, seq_len + 10)
        snapshot.save(manager)
        
        # 应该只保存有效区域
        assert snapshot.is_saved, "快照应该已保存"
        assert snapshot.snapshot_data.shape[4] == 5, "应该只保存有效区域"
    
    def test_snapshot_empty_region(self, mock_kv_cache_manager):
        """测试空区域的情况"""
        manager, _ = mock_kv_cache_manager
        
        # 创建一个空区域的快照
        snapshot = KVCacheSnapshot(30, 30)  # start == end
        snapshot.save(manager)
        
        assert not snapshot.is_saved, "空区域不应该保存快照"
    
    def test_snapshot_none_manager(self):
        """测试 None 管理器的情况"""
        snapshot = KVCacheSnapshot(10, 20)
        snapshot.save(None)
        
        assert not snapshot.is_saved, "None 管理器不应该保存快照"
        
        # restore 应该不会报错
        snapshot.restore(None)
    
    def test_snapshot_uninitialized_cache(self):
        """测试未初始化的 KV Cache"""
        manager = DiffusionKVCacheManager(cache_type='dual', backend='vllm')
        # past_key_values 是 None
        
        snapshot = KVCacheSnapshot(10, 20)
        snapshot.save(manager)
        
        assert not snapshot.is_saved, "未初始化的 KV Cache 不应该保存快照"
    
    def test_multiple_snapshots(self, mock_kv_cache_manager):
        """测试多次快照和恢复"""
        manager, original_data = mock_kv_cache_manager
        
        # 第一次快照
        snapshot1 = KVCacheSnapshot(10, 20)
        snapshot1.save(manager)
        
        # 修改区域 1
        manager.past_key_values._data[:, :, :, :, 10:20, :] = 111.0
        
        # 第二次快照（不同区域）
        snapshot2 = KVCacheSnapshot(30, 40)
        snapshot2.save(manager)
        
        # 修改区域 2
        manager.past_key_values._data[:, :, :, :, 30:40, :] = 222.0
        
        # 恢复第一个快照
        snapshot1.restore(manager)
        
        # 验证区域 1 已恢复
        region1 = manager.past_key_values._data[:, :, :, :, 10:20, :]
        original_region1 = original_data[:, :, :, :, 10:20, :]
        assert torch.allclose(region1, original_region1), "区域 1 应该已恢复"
        
        # 验证区域 2 仍然是修改后的值
        region2 = manager.past_key_values._data[:, :, :, :, 30:40, :]
        assert torch.all(region2 == 222.0), "区域 2 应该保持修改后的值"
        
        # 恢复第二个快照
        snapshot2.restore(manager)
        
        # 验证区域 2 已恢复
        region2_restored = manager.past_key_values._data[:, :, :, :, 30:40, :]
        original_region2 = original_data[:, :, :, :, 30:40, :]
        assert torch.allclose(region2_restored, original_region2), "区域 2 应该已恢复"


class TestKVCacheSnapshotIntegration:
    """KVCacheSnapshot 集成测试"""
    
    def test_snapshot_with_kv_cache_factory(self):
        """测试与 KVCacheFactory 的集成"""
        # 创建 KV Cache 工厂
        factory = KVCacheFactory(cache_type='dual', backend='vllm')
        manager = factory.create()
        
        # 模拟初始化 KV Cache
        num_layers = 2
        batch_size = 1
        num_heads = 4
        seq_len = 64
        hidden_dim = 32
        
        # 创建模拟的 past_key_values 列表
        past_kv_list = []
        for _ in range(num_layers * 2):  # key 和 value
            past_kv_list.append(torch.randn(batch_size, num_heads, seq_len, hidden_dim))
        
        # 更新 KV Cache
        manager.update(past_kv_list)
        
        # 保存原始数据
        original_data = manager.past_key_values._data.clone()
        
        # 创建快照
        snapshot = KVCacheSnapshot(20, 40)
        snapshot.save(manager)
        
        # 修改 KV Cache
        manager.past_key_values._data[:, :, :, :, 20:40, :] = 0.0
        
        # 恢复快照
        snapshot.restore(manager)
        
        # 验证恢复成功
        restored_region = manager.past_key_values._data[:, :, :, :, 20:40, :]
        original_region = original_data[:, :, :, :, 20:40, :]
        assert torch.allclose(restored_region, original_region), "KV Cache 应该已恢复"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
