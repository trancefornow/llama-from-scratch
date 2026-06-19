import sys
import os
# 强行将项目根目录加入环境变量，防止跨文件夹导入报错
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.transformer import LlamaDecoderLayer

def test_decoder_pipeline():
    print("🚀 正在激活 LLaMA 单层解码器完整流水线测试...")
    
    # 模拟输入参数：Batch=2, 句子长度=64, 维度D=512
    B, S, D = 2, 64, 512
    n_heads = 8
    
    # 随机生成伪 token 的 embedding 数据
    x = torch.randn(B, S, D)
    
    # 实例化大合体构件
    try:
        layer = LlamaDecoderLayer(dim=D, n_heads=n_heads)
        out = layer(x)
        
        # 验证 1：检查输出维度是否绝对等于输入维度
        assert out.shape == x.shape, f"【失败】输出维度异常: 期望 {x.shape}，实际得到 {out.shape}"
        print("  -> 测试维度成功")
        
        # 验证 2：检查输出是否包含无效值（NaN 或 Inf）
        assert not torch.isnan(out).any(), "【失败】输出中包含 NaN 坏值，请检查 RMSNorm 或 Softmax！"
        print("  -> 测试前向传播成功")
        
        print("流水线正确\n")
        
    except Exception as e:
        print(f"\n❌【流水线崩溃】报错详情如下：")
        raise e

if __name__ == "__main__":
    test_decoder_pipeline()