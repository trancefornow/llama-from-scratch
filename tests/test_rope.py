#大模型生成测试项

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.rope import RotaryEmbedding

def test_rope_pipeline():
    print("开始进行 RoPE 旋转位置编码严苛数学测试...")
    
    # 模拟 Attention 内部投影后的 Q 和 K
    # 形状: [Batch=2, Heads=4, SeqLen=10, Head_Dim=64]
    B, H, S, D_head = 2, 4, 10, 64
    q = torch.randn(B, H, S, D_head)
    k = torch.randn(B, H, S, D_head)
    
    # 初始化 RoPE
    rope = RotaryEmbedding(dim=D_head)
    q_rope, k_rope = rope(q, k, seq_len=S)
    
    # 测试 1：检查输出维度
    assert q_rope.shape == q.shape and k_rope.shape == k.shape, "【失败】RoPE 改变了张量的原始维度"
    print("测试维度不变性正确")
    
    # 测试 2：高阶数学特性验证 (相对位置不变性)
    # 抽取第 2 个位置的 Q 和 第 5 个位置的 K 计算点积（距离为 3）
    dot_pos_2_5 = torch.sum(q_rope[:, :, 2, :] * k_rope[:, :, 5, :])
    
    # 抽取第 4 个位置的 Q 和 第 7 个位置的 K 计算点积（距离同样为 3）
    dot_pos_4_7 = torch.sum(q_rope[:, :, 4, :] * k_rope[:, :, 7, :])
    
    # 理论上，由于相对距离都是 3，这两个点积结果在数学上应当完全相等！
    # 注意：由于我们要排除原始向量本身数值不同的干扰，这里我们直接用同一对向量放在不同的相对位置来测
    q_single = torch.randn(1, 1, 1, D_head)
    k_single = torch.randn(1, 1, 1, D_head)
    
    # 把同一个 q_single 放到位置 2 和位置 4，k_single 放到位置 5 和位置 7
    q_test = torch.zeros(1, 1, S, D_head)
    k_test = torch.zeros(1, 1, S, D_head)
    q_test[:, :, 2, :] = q_single
    q_test[:, :, 4, :] = q_single
    k_test[:, :, 5, :] = k_single
    k_test[:, :, 7, :] = k_single
    
    qr, kr = rope(q_test, k_test, seq_len=S)
    
    dot1 = torch.sum(qr[:, :, 2, :] * kr[:, :, 5, :]) # 相对距离 3
    dot2 = torch.sum(qr[:, :, 4, :] * kr[:, :, 7, :]) # 相对距离 3
    
    assert torch.allclose(dot1, dot2, atol=1e-4), f"[失败] RoPE失去相对位置感知特性: {dot1.item()} != {dot2.item()}"
    print("测试数学平移不变性成功")
    
    print("RoPE验证正确\n")

if __name__ == "__main__":
    test_rope_pipeline()