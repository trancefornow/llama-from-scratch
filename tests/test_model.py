import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.model import Llama

def test_full_llama():
    print("LLaMA完整模型全流程测试")
    
    # 设定超参数
    vocab_size = 2000    # 假定词表有2000个词
    dim = 512            # 隐层维度
    n_layers = 4         # 尝试4层Decoder
    n_heads = 8          # 8个注意力头
    
    # 实例化完整的LLaMA
    model = Llama(vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads)
    
    # 模拟输入：2句话，每句话包含32个TokenID（全都是0到1999之间的整数）
    B, S = 2, 32
    mock_tokens = torch.randint(0, vocab_size, (B, S))
    
    print(f"  -> 输入文本token形状: {mock_tokens.shape} (Batch={B}, SeqLen={S})")
    
    # 运行前向传播
    logits = model(mock_tokens)
    
    # 验证最终输出维度
    # 期望输出形状: [Batch, SeqLen, Vocab_Size] 
    # 代表模型为每个位置的词，都预测了词表里2000个词里每一个词出现的概率
    expected_shape = (B, S, vocab_size)
    assert logits.shape == expected_shape, f"[失败]输出维度错误！期望 {expected_shape}，实际得到 {logits.shape}"
    
    print(f"  -> 输出 Logits 形状: {logits.shape}")
    print("从Token输入到词表概率输出，整套LLaMA架构成功\n")

if __name__ == "__main__":
    test_full_llama()