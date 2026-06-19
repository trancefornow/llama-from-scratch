import sys
import os
# 根目录加入环境变量，防找不到model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.rmsnorm import RMSNorm

def test_rmsnorm_pipeline():
    print("开始测试RMSNorm")

    B, S, D = 4, 128, 512
    x = torch.randn(B, S, D)

    rms = RMSNorm(dim=512)
    out = rms(x)

    #测试输出维度
    assert out.shape == x.shape, f"[失败] 期望维度 {x.shape}，但得到 {out.shape}"
    print("输出维度正确")

    #测试RMS数学正确性
    rms_value = torch.sqrt(out.pow(2).mean(-1))
    assert torch.allclose(rms_value, torch.ones_like(rms_value), atol=1e-6), f"[失败] 归一化不符合预期"
    print("RMS测试正确")

    print("RMS功能正确")

if __name__ == "__main__":
    test_rmsnorm_pipeline()