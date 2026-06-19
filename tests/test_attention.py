import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.attention import LlamaAttention

def test_attention_pipeline():
    print("开始进行LLaMA Causal Attention测试")
    
    B, S, D = 2, 64, 512
    n_heads = 8
    x1 = torch.randn(B, S, D)
    
    attn = LlamaAttention(dim=D, n_heads=n_heads)
    out1 = attn(x1)
    
    # 检查维度
    assert out1.shape == x1.shape, f"【失败】维度未对齐，得到 {out1.shape}"
    print("测试dim成功")
    
    # 测试mask
    # 我们克隆一份输入 x2，并疯狂篡改它【最后一个词】的特征
    x2 = x1.clone()
    x2[:, -1, :] += 99.9  
    out2 = attn(x2)
    
    # 理论上，由于因果掩码的存在，最后这个词的剧烈波动，绝对无法逆流影响到它前面的那 63 个词！
    # 也就是说，out1 和 out2 的前 S-1 个位置应该完全 100% 相等。
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5), "【失败】时间穿梭了！因果掩码失效，发生了未来信息泄露！"
    print("测试mask成功")
    
    print("LLaMA Attention成功\n")

if __name__ == "__main__":
    test_attention_pipeline()