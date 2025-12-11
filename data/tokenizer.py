import numpy as np
import torch
import sys
import os

# 设置工作目录为项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)               

class CADtokenizer:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # --- 1. 命令词表 ---
        # 对应 DeepCAD/Drawing2CAD 的命令类型
        self.cmd_vocab = {
            "PAD": 0, "SOS": 1, "EOS": 2, 
            "Line": 3, "Arc": 4, "Circle": 5, "Extrude": 6
        }
        self.id2cmd = {v: k for k, v in self.cmd_vocab.items()}#反转字典
        self.n_commands = len(self.cmd_vocab)# 命令词表大小， 7 个命令

        # --- 2. 从 cfg 读取基础几何参数 ---
        self.n_bins = cfg.n_bins
        self.min_val = cfg.min_val
        self.max_val = cfg.max_val
        
        # --- 3. 从 cfg 读取空间Token配置 ---
        self.angle_bins = cfg.angle_bins
        self.pos_grid_size = cfg.pos_grid_size
        
        self.n_angle_tokens = cfg.n_angle_tokens
        self.n_pos_tokens = cfg.n_pos_tokens

    # ================= 核心功能：数值 -> Token ID =================
    
    def quantize_val(self, val):
        """将 [-1, 1] 的浮点数映射到 [0, 255]"""
        val = np.clip(val, self.min_val, self.max_val)#裁剪到范围内
        norm = (val - self.min_val) / (self.max_val - self.min_val)
        return int(norm * (self.n_bins - 1))

    def encode_angle(self, theta, phi, gamma):
        """
        将三个欧拉角 (0~2pi) 编码为一个 <A_n> Token ID
        实现 CAD-GPT 论文公式 (2)
        """
        # 简化：假设输入已归一化到 [0, 1]
        t_idx = int(theta * (self.angle_bins - 1))
        p_idx = int(phi * (self.angle_bins - 1))
        g_idx = int(gamma * (self.angle_bins - 1))
        # 扁平化索引: base-9 encoding
        token_id = t_idx * (self.angle_bins ** 2) + p_idx * self.angle_bins + g_idx
        return token_id

    def encode_pos(self, px, py, pz):
        """
        将 (px, py, pz) 编码为一个 <P_k> Token ID
        实现 CAD-GPT 论文公式 (3)
        """
        # 输入假设在 [-1, 1]，先归一化到 [0, 1]
        def norm(v): return (v - self.min_val) / (self.max_val - self.min_val)
        
        ix = int(norm(px) * (self.pos_grid_size - 1))
        iy = int(norm(py) * (self.pos_grid_size - 1))
        iz = int(norm(pz) * (self.pos_grid_size - 1))
        
        # 扁平化索引: z -> y -> x (参考论文顺序)
        token_id = iz * (self.pos_grid_size ** 2) + iy * self.pos_grid_size + ix
        return token_id

    # ================= 核心功能：Token ID -> 数值 =================
    
    def dequantize_val(self, token_id):
        norm = token_id / (self.n_bins - 1)
        return norm * (self.max_val - self.min_val) + self.min_val

    # (解码 angle 和 pos 的逻辑是 encode 的逆过程，此处略，需补全)

    def tokenize_sequence(self, raw_sequence):
        """
        输入: DeepCAD 原始 JSON 序列
        输出: command_ids, args_ids
        """
        # 这里需要根据 DeepCAD 的具体 JSON 结构编写解析逻辑
        # 稍后在第三步详细处理
        pass

# 测试代码
if __name__ == "__main__":
    cfg = Config("train")
    tok = CADtokenizer(cfg)
    print(f"命令词表大小: {tok.n_commands}")
    print(f"角度Token数量: {tok.n_angle_tokens}") # 应为 729
    print(f"位置Token数量: {tok.n_pos_tokens}")   # 应为 46656
    
    # 测试编码
    angle_token = tok.encode_angle(0.5, 0.5, 0.5)
    print(f"角度 (0.5,0.5,0.5) -> Token {angle_token}")