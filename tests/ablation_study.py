import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from model.model import Llama

# ==========================================
# 1. 构建一个 GPT-2 风格的传统 Transformer 对照组 Layer
# ==========================================
class GPT2StyleBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, bias=True, batch_first=True)
        self.ln_2 = nn.LayerNorm(dim)
        # 传统 MLP：GeLU 激活 + 带偏置
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=True)
        )

    def forward(self, x, attn_mask):
        # 传统前置层归一化
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

class BaselineGPT2(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([GPT2StyleBlock(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=True)

    def forward(self, x):
        seq_len = x.shape[1]
        h = self.embed(x)
        # 生成标准 MultiheadAttention 所需的因果掩码
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        for block in self.blocks:
            h = block(h, attn_mask=mask)
        return self.head(self.ln_f(h))

# ==========================================
# 2. 启动双模型同台竞技竞技场
# ==========================================
def run_ablation_arena():
    print("==================================================")
    print("⚖️  正在启动 LLaMA 1 vs GPT-2 论文级消融对抗竞技场")
    print("==================================================")
    
    # 数据集准备
    text = (
        "Far over the misty mountains cold\n"
        "To dungeons deep and caverns old\n"
        "We must away, ere break of day\n"
        "To find our long-forgotten gold"
    )
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2id = {ch: i for i, ch in enumerate(chars)}
    
    data = torch.tensor([char2id[ch] for ch in text], dtype=torch.long)
    x = data[:-1].unsqueeze(0)
    y = data[1:].unsqueeze(0)

    # 实验配置对齐
    dim, n_layers, n_heads = 128, 2, 4
    
    # 初始化实验组与对照组
    llama_model = Llama(vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads)
    gpt2_model = BaselineGPT2(vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads)
    
    # 优化器对齐
    llama_opt = optim.AdamW(llama_model.parameters(), lr=1.5e-3)
    gpt2_opt = optim.AdamW(gpt2_model.parameters(), lr=1.5e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"{'Epoch':<10}{'🚀 Your LLaMA 1 Loss':<25}{'📜 Baseline GPT-2 Loss':<25}")
    print("-" * 60)
    
    # 开始监控百轮收敛速度
    for epoch in range(1, 101):
        # 训练 LLaMA
        llama_model.train()
        llama_opt.zero_grad()
        l_logits = llama_model(x)
        l_loss = criterion(l_logits.view(-1, vocab_size), y.view(-1))
        l_loss.backward()
        llama_opt.step()
        
        # 训练 GPT-2 Baseline
        gpt2_model.train()
        gpt2_opt.zero_grad()
        g_logits = gpt2_model(x)
        g_loss = criterion(g_logits.view(-1, vocab_size), y.view(-1))
        g_loss.backward()
        gpt2_opt.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:<10}{l_loss.item():<25.4f}{g_loss.item():<25.4f}")
            
    print("-" * 60)
    print("🏁 竞技场对决结束！")
    print("💡 论文结论支撑点：请观察在相同的学习率下，搭载了 RMSNorm + SwiGLU 的 LLaMA 1")
    print("   其 Loss 下降斜率是否比搭载了 LayerNorm + GeLU 的传统 Baseline 更为激进和陡峭。")

if __name__ == "__main__":
    run_ablation_arena()