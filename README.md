# LLaMA From Scratch

这是一个从零实现LLaMA 1核心架构的学习型项目。

项目不依赖HuggingFace Transformers等高级模型框架，而是尽量只使用 PyTorch 的基础张量算子，手动实现模拟LLaMA Decoder-only Transformer的核心模块。

## 项目目标

本项目的目标是通过手写代码理解大模型底层结构，包括：

- 自注意力机制
- 因果掩码
- RMSNorm
- SwiGLU
- RoPE旋转位置编码
- 无偏置线性层
- 静态KV Cache推理加速

## 已实现功能

- LLaMA风格Decoder-only Transformer
- RMSNorm归一化层
- SwiGLU前馈网络
- RoPE旋转位置编码
- Bias-Free线性层
- Causal Self-Attention
- 静态预分配KV Cache
- 基于`start_pos`的增量推理
- 文本生成入口 `generate.py`
- KV Cache性能基准测试
- 多个单元测试与sanity check

## 项目结构

```text
llama_from_scratch/
├── benchmarks/
│   └── benchmark_kv_cache.py  # full recompute vs static KV cache性能对比
├── data/
│   └── input.txt              # 训练/测试文本
├── model/
│   ├── attention.py           # 因果注意力与静态KV Cache
│   ├── model.py               # 顶层LLaMA模型
│   ├── rmsnorm.py             # RMSNorm实现
│   ├── rope.py                # RoPE旋转位置编码
│   ├── swiglu.py              # SwiGLU前馈网络
│   ├── tokenizer.py           # GPT-2 BPE tokenizer封装
│   └── transformer.py         # Decoder Layer
├── tests/
│   ├── sanity_check.py        # 架构行为校验
│   ├── verify_kv_cache.py     # KV Cache 等价性验证
│   └── test_*.py              # 各模块测试
├── generate.py                # 自回归文本生成入口
└── train.py                   # 数据管道与训练实验入口
```

## 核心架构

### RMSNorm

LLaMA使用RMSNorm替代传统LayerNorm。

RMSNorm不做均值中心化，只根据均方根进行归一化，因此计算更简洁。

### SwiGLU

前馈网络使用SwiGLU门控结构：

```text
FFN(x) = W2(silu(W1(x)) * W3(x))
```

相比普通GeLU FFN，SwiGLU具备更强的非线性表达能力。

### RoPE

RoPE，即Rotary Position Embedding，用于Query和Key注入位置信息。

它通过旋转向量空间的方式，让模型能够隐式建模相对位置关系。

### Bias-Free

所有线性层均使用：

```python
bias=False
```

以对齐LLaMA的无偏置设计。

## 静态 KV Cache

本项目实现了类似Meta LLaMA官方源码的静态 KV Cache。

使用方式：

```python
model.eval()
model.setup_cache(max_batch_size=batch_size, max_seq_len=max_seq_len)
model.clear_cache()

for pos in range(seq_len):
    logits = model(tokens[:, pos:pos + 1], start_pos=pos, use_cache=True)
```

该实现不会在每一步使用`torch.cat`动态拼接K/V，而是提前分配缓存空间，然后通过`start_pos`原地写入。

## 文本生成

运行随机初始化模型的生成流程：

```bash
python -X utf8 generate.py --prompt "Far over" --max-new-tokens 80 --use-cache
```

关闭KV Cache进行对照：

```bash
python -X utf8 generate.py --prompt "Far over" --max-new-tokens 80 --no-cache
```

加载checkpoint：

```bash
python -X utf8 generate.py --checkpoint checkpoint.pt --prompt "Far over" --max-new-tokens 80
```

注：如果不传入`--checkpoint`，脚本会使用随机初始化权重。此时输出只能用于验证推理链路是否跑通，不代表模型具备真实语言能力。

## KV Cache Benchmark

运行基准测试：

```bash
python -X utf8 benchmarks/benchmark_kv_cache.py
```

可自定义模型规模和上下文长度：

```bash
python -X utf8 benchmarks/benchmark_kv_cache.py \
  --device cpu \
  --prompt-len 128 \
  --new-tokens 64 \
  --dim 128 \
  --n-layers 2 \
  --n-heads 4
```

一次CPU测试结果示例：

```text
full recompute : 0.5310s | 120.53 tok/s
static KV cache: 0.2414s | 265.10 tok/s
speedup        : 2.20x
max_abs_diff   : 0.00000143
tokens_match   : True
```

其中：

- `full recompute` 表示每生成一个token都重新计算完整上下文。
- `static KV cache` 表示只计算新增 token，并复用历史 K/V。
- `max_abs_diff` 用于检查两条路径的 logits 数值误差。
- `tokens_match=True` 表示两条路径在贪心解码下生成结果完全一致。

## 测试

运行核心测试：

```bash
python -X utf8 tests/verify_kv_cache.py
python -X utf8 tests/sanity_check.py
python -X utf8 tests/test_attention.py
python -X utf8 tests/test_model.py
python -X utf8 tests/test_rope.py
python -X utf8 tests/test_rmsnorm.py
python -X utf8 tests/test_swiglu.py
python -X utf8 tests/test_pipeline.py
```

测试内容包括：

- RMSNorm标量不变性
- 无偏置结构验证
- 因果掩码防泄漏测试
- RoPE数学正确性验证
- 静态KV Cache与全量重算结果对齐
- Token-by-token推理验证
- Chunked prefill推理验证

## 环境依赖

```bash
pip install torch tiktoken
```

推荐环境：

- Python 3.10+
- PyTorch 2.x

## 当前状态

已完成：

- LLaMA 核心结构实现
- RMSNorm / SwiGLU / RoPE
- Causal Attention
- Bias-Free 架构
- 静态 KV Cache
- 文本生成入口
- KV Cache benchmark
- 基础测试矩阵

后续有计划的扩展：

- GQA / MQA
- 更完整的 tokenizer
- 训练 checkpoint 保存与加载
- 训练日志与 loss 曲线
- 更系统的推理速度benchmark
- 实验总结

## 项目定位

这是一个用于学习和理解大模型底层机制的educational project。

项目重点不是训练一个大规模可用模型，而是通过从零实现LLaMA核心组件，理解现代Transformer架构的内部工作方式。
