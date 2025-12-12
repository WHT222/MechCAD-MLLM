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
    """
    CAD命令和参数的Tokenizer类。
    负责将CAD命令和参数转换为离散的Token ID，反之亦然。
    支持DeepCAD格式的输入输出。
    """
    def __init__(self, cfg):
        self.cfg = cfg
        
        # --- 1. 命令词表 ---
        # 对应 DeepCAD/Drawing2CAD 的命令类型

        # 修复特殊标记
        self.cmd_vocab = {"PAD": 0, "SOS": 1, "EOS": 2}
        next_idx = 3 # 意思是从3开始给命令编号

        # 将 CAD_COMMANDS 添加到词表，分配新的连续ID
        # 确保如果 CAD_COMMANDS 中存在 'EOS'，不会重复添加
        for cmd in CAD_COMMANDS:
            if cmd not in self.cmd_vocab: # 'EOS' 已经定义，如果在 CAD_COMMANDS 中出现，将被跳过
                self.cmd_vocab[cmd] = next_idx
                next_idx += 1
        
        self.id2cmd = {v: k for k, v in self.cmd_vocab.items()}#格式转化为 {id：命令}
        self.n_commands = len(self.cmd_vocab)

        # 将命令ID存储为实例变量，方便在tokenize_sequence中直接使用
        self.CAD_LINE_IDX = self.cmd_vocab["Line"]
        self.CAD_ARC_IDX = self.cmd_vocab["Arc"]
        self.CAD_CIRCLE_IDX = self.cmd_vocab["Circle"]
        self.CAD_EOS_IDX = self.cmd_vocab["EOS"]
        self.CAD_SOL_IDX = self.cmd_vocab["SOL"]
        self.CAD_EXT_IDX = self.cmd_vocab["Ext"]
        "后期可以配置到macro.py中"

        # --- 2. 从 cfg 读取基础几何参数 ---
        self.n_bins = cfg.n_bins # 通用参数量化等级256
        self.min_val = cfg.min_val # 通用参数最小值-1.0
        self.max_val = cfg.max_val # 通用参数最大值1.0
        
        # --- 3. 从 cfg 读取空间Token配置 ---
        self.angle_bins = cfg.angle_bins # 角度量化等级
        self.pos_grid_size = cfg.pos_grid_size # 空间位置网格大小
        self.sketch_bins = cfg.sketch_bins # 新增：2D草图坐标的量化等级
        
        self.n_angle_tokens = cfg.n_angle_tokens
        self.n_pos_tokens = cfg.n_pos_tokens
        self.n_sketch_tokens = cfg.n_sketch_tokens

    # ================= 核心功能：数值 -> Token ID =================
    
    def quantize_val(self, val):
        """将 [-1, 1] 的浮点数映射到 [0, n_bins-1], 用于通用参数"""
        val = np.clip(val, self.min_val, self.max_val)#clip实现
        norm = (val - self.min_val) / (self.max_val - self.min_val)
        return int(norm * (self.n_bins - 1))

    def encode_sketch_x(self, x):
        """将2D X坐标编码为独立的 <SlX> Token ID [0, 127]"""
        x_norm = (np.clip(x, self.min_val, self.max_val) - self.min_val) / (self.max_val - self.min_val)
        return int(x_norm * (self.sketch_bins - 1))

    def encode_sketch_y(self, y):
        """将2D Y坐标编码为独立的 <SmY> Token ID [128, 255]"""
        y_norm = (np.clip(y, self.min_val, self.max_val) - self.min_val) / (self.max_val - self.min_val)
        # 添加偏移量以区分X和Y
        return int(y_norm * (self.sketch_bins - 1)) + self.sketch_bins

    def encode_angle(self, theta, phi, gamma):
        """
        将三个欧拉角 (范围 [0, 2*pi]) 编码为一个 <A_n> Token ID。
        实现了 CAD-GPT 论文公式 (2)。
        按照 theta (低位) -> phi -> gamma (高位) 的顺序进行组合编码。
        """
        theta_norm = np.clip(theta, 0, 2 * np.pi) / (2 * np.pi)
        phi_norm = np.clip(phi, 0, 2 * np.pi) / (2 * np.pi)
        gamma_norm = np.clip(gamma, 0, 2 * np.pi) / (2 * np.pi)

        t_idx = int(theta_norm * (self.angle_bins - 1))
        p_idx = int(phi_norm * (self.angle_bins - 1))
        g_idx = int(gamma_norm * (self.angle_bins - 1))
        
        # 扁平化索引: 按照 gamma (高位) -> phi -> theta (低位) 的顺序
        token_id = g_idx * (self.angle_bins ** 2) + p_idx * self.angle_bins + t_idx
        return token_id

    def encode_pos(self, px, py, pz):
        """
        将 (px, py, pz) (范围 [-1, 1]) 编码为一个 <P_k> Token ID。
        实现了 CAD-GPT 论文公式 (3)。
        """
        def norm(v): return (np.clip(v, self.min_val, self.max_val) - self.min_val) / (self.max_val - self.min_val)
        
        ix = int(norm(px) * (self.pos_grid_size - 1))
        iy = int(norm(py) * (self.pos_grid_size - 1))
        iz = int(norm(pz) * (self.pos_grid_size - 1))
        
        token_id = iz * (self.pos_grid_size ** 2) + iy * self.pos_grid_size + ix
        return token_id

    # ================= 核心功能：Token ID -> 数值 =================
    
    def dequantize_val(self, token_id):
        """将 [0, n_bins-1] 的ID反量化为 [-1, 1] 的浮点数"""
        norm = token_id / (self.n_bins - 1)
        return norm * (self.max_val - self.min_val) + self.min_val

    def decode_sketch_coord(self, token_id):
        """将 <SlX> 或 <SmY> Token ID 解码回其原始2D坐标值"""
        if 0 <= token_id < self.sketch_bins: # X-coord
            norm = token_id / (self.sketch_bins - 1)
        elif self.sketch_bins <= token_id < self.sketch_bins * 2: # Y-coord
            norm = (token_id - self.sketch_bins) / (self.sketch_bins - 1)
        else:
            raise ValueError(f"Invalid sketch token ID: {token_id}")
        
        return norm * (self.max_val - self.min_val) + self.min_val

    def decode_angle(self, token_id):
        """
        将 <A_n> Token ID 解码回三个欧拉角 (范围 [0, 2*pi])。
        按照 gamma (高位) -> phi -> theta (低位) 的顺序进行反向解码。
        """
        # 逆向扁平化
        t_idx = token_id % self.angle_bins # 提取最低有效位
        token_id //= self.angle_bins
        p_idx = token_id % self.angle_bins # 提取中间位
        g_idx = token_id // self.angle_bins # 提取最高有效位

        theta_norm = t_idx / (self.angle_bins - 1)
        phi_norm = p_idx / (self.angle_bins - 1)
        gamma_norm = g_idx / (self.angle_bins - 1)
        
        theta = theta_norm * 2 * np.pi
        phi = phi_norm * 2 * np.pi
        gamma = gamma_norm * 2 * np.pi

        return theta, phi, gamma

    def decode_pos(self, token_id):
        """将 <P_k> Token ID 解码回 (px, py, pz) 坐标 (范围 [-1, 1])"""
        ix = token_id % self.pos_grid_size
        token_id //= self.pos_grid_size
        iy = token_id % self.pos_grid_size
        iz = token_id // self.pos_grid_size

        px_norm = ix / (self.pos_grid_size - 1)
        py_norm = iy / (self.pos_grid_size - 1)
        pz_norm = iz / (self.pos_grid_size - 1)

        px = px_norm * (self.max_val - self.min_val) + self.min_val
        py = py_norm * (self.max_val - self.min_val) + self.min_val
        pz = pz_norm * (self.max_val - self.min_val) + self.min_val
        
        return px, py, pz

    def tokenize_sequence(self, raw_sequence):
        """
        将DeepCAD原始JSON序列转换为Token ID序列。
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

            if cmd_id == self.CAD_LINE_IDX: # Line: x, y (params 0, 1)
                current_args[0] = self.encode_sketch_x(params[0]) # x
                current_args[1] = self.encode_sketch_y(params[1]) # y
            elif cmd_id == self.CAD_ARC_IDX: # Arc: x, y, alpha, f (params 0, 1, 2, 3)
                current_args[0] = self.encode_sketch_x(params[0]) # x
                current_args[1] = self.encode_sketch_y(params[1]) # y
                # alpha 和 f 是通用参数，继续使用 quantize_val
                current_args[2] = self.quantize_val(params[2]) # alpha
                current_args[3] = self.quantize_val(params[3]) # f
            elif cmd_id == self.CAD_CIRCLE_IDX: # Circle: x, y, r (params 0, 1, 4)
                current_args[0] = self.encode_sketch_x(params[0]) # x
                current_args[1] = self.encode_sketch_y(params[1]) # y
                # r (半径) 是通用参数
                current_args[4] = self.quantize_val(params[4]) # r
            elif cmd_id == self.CAD_EXT_IDX: # Extrude
                # Angle token
                theta, phi, gamma = params[0], params[1], params[2]
                angle_token = self.encode_angle(theta, phi, gamma)
                current_args[CAD_N_ARGS_SKETCH + 0] = angle_token # Slot 5
                # Position token
                px, py, pz = params[3], params[4], params[5]
                pos_token = self.encode_pos(px, py, pz)
                current_args[CAD_N_ARGS_SKETCH + 1] = pos_token # Slot 6
                # Other extrusion params are generic
                for i in range(6, 11):
                    current_args[CAD_N_ARGS_SKETCH + (i - 4)] = self.quantize_val(params[i])
            
            args_ids.append(current_args)
        
        return command_ids, args_ids

# 测试代码
if __name__ == "__main__":
    class MockConfig:
        def __init__(self):
            self.n_bins = 256
            self.sketch_bins = 128
            self.min_val = -1.0
            self.max_val = 1.0
            self.angle_bins = 9
            self.pos_grid_size = 36
        
        @property
        def n_angle_tokens(self): return self.angle_bins ** 3
        @property
        def n_pos_tokens(self): return self.pos_grid_size ** 3
        @property
        def n_sketch_tokens(self): return self.sketch_bins * 2

    cfg = MockConfig()
    tok = CADtokenizer(cfg)
    print(f"命令词表大小: {tok.n_commands}")
    print(f"2D草图Token数量: {tok.n_sketch_tokens}") # 应为 256
    print(f"角度Token数量: {tok.n_angle_tokens}") # 应为 729
    print(f"位置Token数量: {tok.n_pos_tokens}")   # 应为 46656
    
    # Test `encode_angle` with new order
    test_theta, test_phi, test_gamma = np.pi/4, np.pi/2, 3*np.pi/4 # 真实弧度值
    angle_token = tok.encode_angle(test_theta, test_phi, test_gamma)
    print(f"\n--- Testing Angle Tokens (new order) ---")
    print(f"原始角度: ({test_theta:.4f}, {test_phi:.4f}, {test_gamma:.4f})")
    print(f"编码Token: {angle_token}")
    
    decoded_theta, decoded_phi, decoded_gamma = tok.decode_angle(angle_token)
    print(f"解码角度: ({decoded_theta:.4f}, {decoded_phi:.4f}, {decoded_gamma:.4f})")
    
    # Verify round-trip (allow for quantization error)
    assert np.isclose(test_theta, decoded_theta, atol=2*np.pi/(tok.angle_bins-1)), "Theta round-trip failed"
    assert np.isclose(test_phi, decoded_phi, atol=2*np.pi/(tok.angle_bins-1)), "Phi round-trip failed"
    assert np.isclose(test_gamma, decoded_gamma, atol=2*np.pi/(tok.angle_bins-1)), "Gamma round-trip failed"
    print("角度Token编码-解码往返测试通过！")

    # Test `encode_sketch_x/y`
    x_val, y_val = 0.5, -0.5
    x_token = tok.encode_sketch_x(x_val)
    y_token = tok.encode_sketch_y(y_val)
    print(f"\n--- Testing 2D Sketch Tokens ---")
    print(f"X coord {x_val} -> Token {x_token} (范围 [0, 127])")
    print(f"Y coord {y_val} -> Token {y_token} (范围 [128, 255])")
    
    decoded_x = tok.decode_sketch_coord(x_token)
    decoded_y = tok.decode_sketch_coord(y_token)
    print(f"Token {x_token} -> Decoded X: {decoded_x:.4f}")
    print(f"Token {y_token} -> Decoded Y: {decoded_y:.4f}")
    
    # Verify round-trip for sketch coords
    assert np.isclose(x_val, decoded_x, atol=2/(tok.sketch_bins-1)), "X sketch round-trip failed"
    assert np.isclose(y_val, decoded_y, atol=2/(tok.sketch_bins-1)), "Y sketch round-trip failed"
    print("2D草图Token编码-解码往返测试通过！")


    # Test `tokenize_sequence`
    sample_raw_sequence = [
        {"command": "Line", "parameters": [0.1, 0.2, 0.0, 0.0, 0.0]},
        {"command": "Ext", "parameters": [np.pi/4, np.pi/2, 3*np.pi/4, -0.4, -0.5, -0.6, 0.7, 0.8, 0.9, 0.1, 0.2]}
    ]
    print("\n--- Testing tokenize_sequence with 'Line' and 'Ext' ---")
    command_ids, args_ids = tok.tokenize_sequence(sample_raw_sequence)
    
    # Line command args
    print(f"命令 0: {tok.id2cmd[command_ids[0]]}")
    print(f"  Args[0:2]: [{args_ids[0][0]}, {args_ids[0][1]}]")
    print(f"  Decoded X: {tok.decode_sketch_coord(args_ids[0][0]):.4f}, Decoded Y: {tok.decode_sketch_coord(args_ids[0][1]):.4f}")
    
    # Extrude command args
    print(f"命令 1: {tok.id2cmd[command_ids[1]]}")
    print(f"  Angle Token: {args_ids[1][CAD_N_ARGS_SKETCH + 0]}, Pos Token: {args_ids[1][CAD_N_ARGS_SKETCH + 1]}")
    dec_t, dec_p, dec_g = tok.decode_angle(args_ids[1][CAD_N_ARGS_SKETCH + 0])
    print(f"  Decoded Ext Angle: ({dec_t:.4f}, {dec_p:.4f}, {dec_g:.4f})")
    dec_px, dec_py, dec_pz = tok.decode_pos(args_ids[1][CAD_N_ARGS_SKETCH + 1])
    print(f"  Decoded Ext Pos: ({dec_px:.4f}, {dec_py:.4f}, {dec_pz:.4f})")
    print("tokenize_sequence 测试通过！")