# LLaMA From Scratch

这是一个从零实现 LLaMA 1 核心架构的学习型项目。

项目不依赖 HuggingFace Transformers 等高级模型框架，而是尽量只使用 PyTorch 的基础张量算子，手动实现 LLaMA 风格 Decoder-only Transformer 的核心模块。

## 项目目标

本项目的目标是通过手写代码理解大模型底层结构，包括：

- 自注意力机制
- 因果掩码
- RMSNorm
- SwiGLU
- RoPE 旋转位置编码
- 无偏置线性层
- KV Cache 推理加速

## 已实现功能

- LLaMA 风格 Decoder-only Transformer
- RMSNorm 归一化层
- SwiGLU 前馈网络
- RoPE 旋转位置编码
- Bias-Free 线性层
- Causal Self-Attention
- 静态预分配 KV Cache
- 基于 `start_pos` 的增量推理
- 多个单元测试与 sanity check

## 项目结构

```text
llama_from_scratch/
├── model/
│   ├── attention.py      # 因果注意力与静态 KV Cache
│   ├── model.py          # 顶层 LLaMA 模型
│   ├── rmsnorm.py        # RMSNorm 实现
│   ├── rope.py           # RoPE 旋转位置编码
│   ├── swiglu.py         # SwiGLU 前馈网络
│   ├── tokenizer.py      # 简单 tokenizer
│   └── transformer.py    # Decoder Layer
├── tests/
│   ├── sanity_check.py       # 架构行为校验
│   ├── verify_kv_cache.py    # KV Cache 等价性验证
│   └── test_*.py             # 各模块测试
└── train.py