import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.rmsnorm import RMSNorm
from model.transformer import LlamaDecoderLayer

def test_rmsnorm_scalar_invariance():
    print("🧪 [RMSNorm] 启动标量不变性测试...")
    dim = 128
    rmsnorm = RMSNorm(dim=dim)
    x = torch.randn(2, 10, dim)
    
    out1 = rmsnorm(x)
    out2 = rmsnorm(x * 12.5) # 故意放大 12.5 倍
    
    # 检查均方误差是否接近于 0
    assert torch.allclose(out1, out2, atol=1e-5), "❌ RMSNorm 未通过标量不变性测试！请检查分母的 RMS 计算。"
    print("  -> ✅ 成功：输入缩放不影响归一化分布，标量守恒定律成立。")

def test_bias_free_extreme():
    print("🧪 [Architecture] 启动无偏置（Bias-Free）极端值测试...")
    dim = 128
    layer = LlamaDecoderLayer(dim=dim, n_heads=4)
    
    # 注入纯 0 矩阵。如果有任何一行带有 bias，输出绝对不会是全 0
    zero_input = torch.zeros(1, 8, dim)
    out = layer(zero_input)
    
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), "❌ 发现隐藏的偏置项！请检查 Linear 是否漏掉了 bias=False。"
    print("  -> ✅ 成功：输入全 0 则输出全 0，完美符合 LLaMA 无偏置规范。")

def test_causal_mask_leakage():
    print("🧪 [Attention] 启动因果掩码（Causal Mask）时空防偷看测试...")
    dim = 128
    layer = LlamaDecoderLayer(dim=dim, n_heads=4)
    layer.eval()
    
    # 构造两个序列：x1 和 x2 的前 4 个 Token 完全一样，但第 5 个 Token 截然不同
    x1 = torch.randn(1, 5, dim)
    x2 = x1.clone()
    x2[0, 4, :] += 10.0 # 剧烈修改最后一个 Token
    
    with torch.no_grad():
        out1 = layer(x1)
        out2 = layer(x2)
        
    # 判定：即使未来（第5个字）改翻天，历史（前4个字）的表征向量也必须纹丝不动
    assert torch.allclose(out1[0, :4, :], out2[0, :4, :], atol=1e-5), "❌ 发生时间穿梭！因果掩码存在未来信息泄露。"
    print("  -> ✅ 成功：修改尾部 Token 不影响前方历史状态，时间防御坚不可摧。")

if __name__ == "__main__":
    print("==================================================")
    print("🛡️ 正在激活 LLaMA 1 三重行为神智检查矩阵...")
    print("==================================================")
    try:
        test_rmsnorm_scalar_invariance()
        test_bias_free_extreme()
        test_causal_mask_leakage()
        print("\n🎉【神智全面清醒】所有魔改算子的边界行为与官方数学期望 100% 对齐！")
    except AssertionError as e:
        print(f"\n💥【检测到架构异常】")
        raise e