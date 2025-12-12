import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config.config import Config
from src.model.model import LLM2CADDecoder # 假设你把上面的类加到了 model.py

# 1. 配置初始化
cfg = Config(phase="test")

# 2. 实例化模型
model = LLM2CADDecoder(cfg, llm_hidden_dim=4096)
print("模型构建成功！")

# 3. 模拟输入数据
batch_size = 4
llm_seq_len = 50 # 假设 LLM 输出了 50 个 token
llm_dim = 4096   # 假设是 LLaMA-7B
fake_llm_output = torch.randn(batch_size, llm_seq_len, llm_dim)

# 4. 前向传播
try:
    cmd_logits, args_features, angle_logits, pos_logits = model(fake_llm_output)
    print("前向传播成功！")
    print(f"Command Output Shape: {cmd_logits.shape}")        # 预期: [4, 60, 6]
    print(f"Args Features Shape: {args_features.shape}")      # 预期: [4, 60, 12, 257]
    print(f"Angle Token Logits Shape: {angle_logits.shape}")  # 预期: [4, 60, 729]
    print(f"Position Token Logits Shape: {pos_logits.shape}") # 预期: [4, 60, 46656]
except Exception as e:
    print(f"出错啦: {e}")