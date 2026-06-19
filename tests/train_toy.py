import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from model.model import Llama

def run_toy_training():
    print("LlaMA收敛性与自回归测试")

    # Misty Mountains Cold, from The Hobbit
    text = (
        "Far over the misty mountains cold\n"
        "To dungeons deep and caverns old\n"
        "We must away, ere break of day\n"
        "To find our long-forgotten gold\n\n"
        "The pines were roaring on the height\n"
        "The winds were moaning in the night\n"
        "The fire was red, it flaming spread\n"
        "The trees like torches blazed with light"
    )
    print(f"文段加载成功，长{len(text)}")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2id = {ch: i for i, ch in enumerate(chars)}
    id2char = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([char2id[ch] for ch in text], dtype=torch.long)
    x = data[:-1].unsqueeze(0)
    y = data[1:].unsqueeze(0)

    model = Llama(vocab_size=vocab_size, dim=128, n_layers=2, n_heads=4)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(1, 201):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch{epoch: 3d} | 当前交叉熵loss:{loss.item(): .4f}")
    
    print("自回归推理验证")
    model.eval()

    start_text = "Far over"
    tokens = [char2id[ch] for ch in start_text]
    input_tensor = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        for _ in range(len(text) - len(start_text)):
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()

            tokens.append(next_token)
            input_tensor = torch.tensor([tokens], dtype=torch.long)
    
    generated_text = "".join([id2char[t] for t in tokens])
    print("-" * 60)
    print(f"输入的提示词是{start_text}")
    print(f"LLaMA生成结果:\n{generated_text}")
    print("-" * 60)

    if generated_text == text:
        print("\n输出完全相同")
    else:
        print("\n输出存在不同")
        

if __name__ == "__main__":
    run_toy_training()