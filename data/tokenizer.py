import numpy as np
import torch

class CADtokenizer:
    def __init__(self, cfg):
        # --- 1. 命令词表 ---
        # 对应 DeepCAD/Drawing2CAD 的命令类型
        self.cmd_vocab = {
            "PAD": 0, "SOS": 1, "EOS": 2, 
            "Line": 3, "Arc": 4, "Circle": 5, "Extrude": 6
        }
        self.id2cmd = {v: k for k, v in self.cmd_vocab.items()}#反转字典
        self.n_commands = len(self.cmd_vocab)# 命令词表大小， 7 个命令

        # --- 2. 基础参数配置 ---
        self.n_bins = 256  # 2D坐标量化精度 (Drawing2CAD默认)
        self.min_val = -1.0 # 坐标最小值
        self.max_val = 1.0 # 坐标最大值
        
        # --- 3. 空间Token配置 (CAD-GPT Innovation) ---
        # 角度离散化: 3个欧拉角，每个角分 9 档 -> 729 个Token
        self.angle_bins = 9 
        self.n_angle_tokens = self.angle_bins ** 3 
        
        # 3D位置离散化: K=36 -> 46656 个Token
        self.pos_grid_size = 36 
        self.n_pos_tokens = self.pos_grid_size ** 3

    # ================= 核心功能：数值 -> Token ID =================
    
    def quantize_val(self, val):
        """将 [-1, 1] 的浮点数映射到 [0, 255]"""
        val = np.clip(val, self.min_val, self.max_val)
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
    tok = CADtokenizer(None)
    print(f"命令词表大小: {tok.n_commands}")
    print(f"角度Token数量: {tok.n_angle_tokens}") # 应为 729
    print(f"位置Token数量: {tok.n_pos_tokens}")   # 应为 32768
    
    # 测试编码
    angle_token = tok.encode_angle(0.5, 0.5, 0.5)
    print(f"角度 (0.5,0.5,0.5) -> Token {angle_token}")