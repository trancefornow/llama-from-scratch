import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.swiglu import FeedForward

def test_swiglu_pipeline():
    print("测试SwiGLU")

    B, S, D = 4, 128, 512
    hidden_dim = 1376
    x = torch.randn(B, S, D)

    ffn = FeedForward(dim=D, hidden_dim=hidden_dim)
    out = ffn(x)
    
    #检测维度
    assert out.shape == x.shape, f"[失败]期望维度 {x.shape}，但得到 {out.shape}"
    print("测试维度成功")
    
    # 测试反向传播梯度
    loss = out.sum()
    loss.backward()
    # 确保三个权重矩阵都顺利拿到了梯度，没有发生梯度断流
    assert ffn.w1.weight.grad is not None, "[失败]W1 梯度断流"
    assert ffn.w3.weight.grad is not None, "[失败]W3 梯度断流"
    assert ffn.w2.weight.grad is not None, "[失败]W2 梯度断流"
    print("测试反向传播全通路梯度计算正常")
    
    print("SwiGLU正确\n")

if __name__ == "__main__":
    test_swiglu_pipeline()