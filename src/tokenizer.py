import numpy as np
import torch
import sys
import os
# 设置工作目录为项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)               
from config.macro import (
    CAD_N_ARGS, PAD_VAL, CAD_COMMANDS,
    CAD_N_ARGS_SKETCH
)
from config.config import Config               

class CADtokenizer:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # --- 1. 命令词表 ---
        # 对应 DeepCAD/Drawing2CAD 的命令类型
        from config.macro import CAD_COMMANDS

        # Fixed special tokens
        self.cmd_vocab = {"PAD": 0, "SOS": 1, "EOS": 2}
        next_idx = 3 # Start indexing commands after fixed special tokens

        # Add CAD_COMMANDS to vocab, assigning new sequential IDs
        # Make sure not to re-add 'EOS' if it exists in CAD_COMMANDS
        for cmd in CAD_COMMANDS:
            if cmd not in self.cmd_vocab: # 'EOS' is already defined, so it will be skipped if present in CAD_COMMANDS
                self.cmd_vocab[cmd] = next_idx
                next_idx += 1
        
        self.id2cmd = {v: k for k, v in self.cmd_vocab.items()}
        self.n_commands = len(self.cmd_vocab)

        # Store command IDs as instance variables for direct use in tokenize_sequence
        self.CAD_LINE_IDX = self.cmd_vocab["Line"]
        self.CAD_ARC_IDX = self.cmd_vocab["Arc"]
        self.CAD_CIRCLE_IDX = self.cmd_vocab["Circle"]
        self.CAD_EOS_IDX = self.cmd_vocab["EOS"]
        self.CAD_SOL_IDX = self.cmd_vocab["SOL"]
        self.CAD_EXT_IDX = self.cmd_vocab["Ext"] # Use "Ext" from config.macro

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
        输入: DeepCAD 原始 JSON 序列 (假设为 list of dicts)
        例如:
        [
          {"command": "Line", "parameters": [x, y, alpha, f, r]},
          {"command": "Extrude", "parameters": [theta, phi, gamma, px, py, pz, s, e1, e2, b, u]},
          // ...
        ]
        输出: command_ids (list), args_ids (list of lists)
        """
        command_ids = []
        args_ids = []

        for op in raw_sequence:
            cmd_str = op["command"]
            cmd_id = self.cmd_vocab.get(cmd_str)
            if cmd_id is None:
                raise ValueError(f"Unknown command: {cmd_str}")



            command_ids.append(cmd_id)

            current_args = [PAD_VAL] * CAD_N_ARGS # Initialize with PAD_VAL

            params = op["parameters"]

            if cmd_id == self.CAD_LINE_IDX: # Line: x, y (params 0, 1 from original DeepCAD)
                current_args[0] = self.quantize_val(params[0]) # x
                current_args[1] = self.quantize_val(params[1]) # y
            elif cmd_id == self.CAD_ARC_IDX: # Arc: x, y, alpha, f (params 0, 1, 2, 3 from original DeepCAD)
                current_args[0] = self.quantize_val(params[0]) # x
                current_args[1] = self.quantize_val(params[1]) # y
                current_args[2] = self.quantize_val(params[2]) # alpha
                current_args[3] = self.quantize_val(params[3]) # f
            elif cmd_id == self.CAD_CIRCLE_IDX: # Circle: x, y, r (params 0, 1, 4 from original DeepCAD)
                current_args[0] = self.quantize_val(params[0]) # x
                current_args[1] = self.quantize_val(params[1]) # y
                current_args[4] = self.quantize_val(params[4]) # r
            elif cmd_id == self.CAD_EXT_IDX: # Extrude

                # Original params: [theta, phi, gamma, px, py, pz, s, e1, e2, b, u] (11 params)
                # New structure: [angle_token, pos_token, s, e1, e2, b, u] (7 params)
                
                # Angle token (original params 0, 1, 2) -> new arg index 5
                theta, phi, gamma = params[0], params[1], params[2]
                angle_token = self.encode_angle(theta, phi, gamma)
                current_args[CAD_N_ARGS_SKETCH + 0] = angle_token # Slot 5

                # Position token (original params 3, 4, 5) -> new arg index 6
                px, py, pz = params[3], params[4], params[5] # This is where the error occurs
                pos_token = self.encode_pos(px, py, pz)
                current_args[CAD_N_ARGS_SKETCH + 1] = pos_token # Slot 6

                # Scale 's' (original param 6) -> new arg index 7
                current_args[CAD_N_ARGS_SKETCH + 2] = self.quantize_val(params[6]) # Slot 7

                # Extrusion parameters e1, e2, b, u (original params 7, 8, 9, 10) -> new arg indices 8, 9, 10, 11
                current_args[CAD_N_ARGS_SKETCH + 3] = self.quantize_val(params[7]) # Slot 8
                current_args[CAD_N_ARGS_SKETCH + 4] = self.quantize_val(params[8]) # Slot 9
                current_args[CAD_N_ARGS_SKETCH + 5] = self.quantize_val(params[9]) # Slot 10
                current_args[CAD_N_ARGS_SKETCH + 6] = self.quantize_val(params[10]) # Slot 11
            
            # EOS and SOL have no arguments, so current_args will remain all PAD_VALs

            args_ids.append(current_args)
        
        return command_ids, args_ids

# 测试代码
if __name__ == "__main__":
    # Mock Config class for testing purposes
    class MockConfig:
        def __init__(self):
            self.n_bins = 256
            self.min_val = -1.0
            self.max_val = 1.0
            self.angle_bins = 9
            self.pos_grid_size = 36
            self.args_dim = 256 # Added for compatibility if model later uses it
        
        @property
        def n_angle_tokens(self):
            return self.angle_bins ** 3
                
        @property
        def n_pos_tokens(self):
            return self.pos_grid_size ** 3

    cfg = MockConfig()
    tok = CADtokenizer(cfg)
    print(f"命令词表大小: {tok.n_commands}")
    print(f"角度Token数量: {tok.n_angle_tokens}") # 应为 729
    print(f"位置Token数量: {tok.n_pos_tokens}")   # 应为 46656
    
    # Test `encode_angle`
    angle_token = tok.encode_angle(0.5, 0.5, 0.5)
    print(f"角度 (0.5,0.5,0.5) -> Token {angle_token}")

    # Test `tokenize_sequence` with a sample DeepCAD JSON
    sample_raw_sequence = [
        {"command": "Line", "parameters": [0.1, 0.2, 0.0, 0.0, 0.0]}, # x, y
        {"command": "Arc", "parameters": [0.3, 0.4, 0.5, 0.6, 0.0]}, # x, y, alpha, f
        {"command": "Circle", "parameters": [0.7, 0.8, 0.0, 0.0, 0.9]}, # x, y, r
        {"command": "Ext", "parameters": [0.1, 0.2, 0.3, -0.4, -0.5, -0.6, 0.7, 0.8, 0.9, 0.1, 0.2]}, # theta, phi, gamma, px, py, pz, s, e1, e2, b, u
        {"command": "EOS", "parameters": []}
    ]

    print("\n--- Testing tokenize_sequence ---")
    command_ids, args_ids = tok.tokenize_sequence(sample_raw_sequence)
