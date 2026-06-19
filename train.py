import torch
from model.tokenizer import BPETokenizer

# 初始化分词器
tokenizer = BPETokenizer("gpt2")
vocab_size = tokenizer.vocab_size

#载入并编码数据
print("正在载入数据并进行 BPE 编码...")
with open('data/input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

#测试data分割规模
train_ids = tokenizer.encode(data)
print(f"文本总共被切分成了 {len(train_ids)} 个 BPE Tokens")

train_tensor = torch.tensor(train_ids, dtype=torch.long)

#超参配置
batch_size = 4
block_size = 128  #mini size

#训练集90%， 验证集10%
n = int(0.9 * len(train_tensor))
train_data = train_tensor[:n]
val_data = train_tensor[n:]

#批处理
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

# 测试数据管道
xb, yb = get_batch('train')
print("\n=== 数据管道打桩测试 ===")
print(f"输入 X 的形状 (Batch, SeqLen): {xb.shape}")
print(f"标签 Y 的形状 (Batch, SeqLen): {yb.shape}")

print("\n深入理解自回归输入输出关系（以第一个 Batch 的前3个词为例）：")
for t in range(3):
    context = xb[0, :t+1]
    target = yb[0, t]
    print(f"当输入上下文为: {tokenizer.decode(context.tolist())} ---> 模型应该预测出: {tokenizer.decode([target.item()])}")