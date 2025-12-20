import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config.config import Config
from src.model.model import MechCAD # CHANGE: Import MechCAD
from PIL import Image # NEW: For dummy image

# 1. 配置初始化
cfg = Config(phase="test")

# 2. 实例化模型
# CHANGE: Instantiate MechCAD
model = MechCAD(cfg) # llava_model_name will default to "llava-hf/llava-1.5-7b-hf"
print("模型构建成功！")

# 3. 模拟输入数据
batch_size = 2 # Use a smaller batch size for dummy test
# NEW: Dummy image input
dummy_images = [Image.new('RGB', (224, 224), color = 'red') for _ in range(batch_size)]
# NEW: Dummy text input
dummy_texts = ["A red cube with a hole.", "A blue cylinder."]


# 4. 前向传播
try:
    # CHANGE: New inputs for MechCAD
    cmd_logits, args_features, angle_logits, pos_logits = model(dummy_images, dummy_texts)
    print("前向传播成功！")
    # CHANGE: Update expected batch size in print statements
    print(f"Command Output Shape: {cmd_logits.shape}")        # 预期: [2, 60, 7]
    print(f"Args Features Shape: {args_features.shape}")      # 预期: [2, 60, 12, 257]
    print(f"Angle Token Logits Shape: {angle_logits.shape}")  # 预期: [2, 60, 729]
    print(f"Position Token Logits Shape: {pos_logits.shape}") # 预期: [2, 60, 46656]
except Exception as e:
    print(f"出错啦: {e}")
