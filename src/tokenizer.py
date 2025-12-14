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
    """
    def __init__(self, cfg):
        self.cfg = cfg
        
        # --- 1. 命令词表 ---
        # 遵循DeepCAD/vec data的编码顺序，先从CAD_COMMANDS列表构建
        self.cmd_vocab = {cmd: i for i, cmd in enumerate(CAD_COMMANDS)}
        
        # 添加不在CAD_COMMANDS中的特殊符号
        if 'PAD' not in self.cmd_vocab:
            self.cmd_vocab['PAD'] = len(self.cmd_vocab)
        if 'SOS' not in self.cmd_vocab:
            self.cmd_vocab['SOS'] = len(self.cmd_vocab)
        
        self.id2cmd = {v: k for k, v in self.cmd_vocab.items()}
        self.n_commands = len(self.cmd_vocab)

        # 将命令ID存储为实例变量
        self.CAD_LINE_IDX = self.cmd_vocab.get("Line")
        self.CAD_ARC_IDX = self.cmd_vocab.get("Arc")
        self.CAD_CIRCLE_IDX = self.cmd_vocab.get("Circle")
        self.CAD_EOS_IDX = self.cmd_vocab.get("EOS")
        self.CAD_SOL_IDX = self.cmd_vocab.get("SOL")
        self.CAD_EXT_IDX = self.cmd_vocab.get("Ext")

        # --- 2. 从 cfg 读取参数 ---
        self.pad_val = cfg.pad_val
        self.angle_bins = cfg.angle_bins
        self.pos_grid_size = cfg.pos_grid_size
        self.cad_n_args = cfg.cad_n_args
        self.cad_n_args_sketch = cfg.cad_n_args_sketch
        
    def tokenize_from_vec(self, vec_data):
        """
        将来自HDF5文件的、已经量化但未组合的 (N, 17) 矩阵进行最终的Token化。
        """
        command_ids = []
        args_ids = []
        
        for row in vec_data:
            cmd_id = int(row[0])
            quantized_params = row[1:] # 16个已量化的整数参数
            
            command_ids.append(cmd_id)
            
            # 初始化最终的12维参数向量
            final_args = [self.pad_val] * self.cad_n_args

            if cmd_id in [self.CAD_LINE_IDX, self.CAD_ARC_IDX, self.CAD_CIRCLE_IDX]:
                # 对于草图命令，HDF5中的前5个参数就是最终参数
                # 直接复制这些已经量化好的整数
                for i in range(self.cad_n_args_sketch):
                    final_args[i] = int(quantized_params[i])

            elif cmd_id == self.CAD_EXT_IDX:
                # 对于拉伸命令，我们需要将多个量化整数组合成空间Token
                
                # --- 组合角度Token ---
                # HDF5中拉伸参数从第5个索引开始，对应theta, phi, gamma的量化值
                t_idx_quant = int(quantized_params[5])
                p_idx_quant = int(quantized_params[6])
                g_idx_quant = int(quantized_params[7])
                
                # 仅在所有索引都有效时才组合
                if all(idx != self.pad_val for idx in [t_idx_quant, p_idx_quant, g_idx_quant]):
                    angle_token = g_idx_quant * (self.angle_bins ** 2) + p_idx_quant * self.angle_bins + t_idx_quant
                    final_args[5] = angle_token # 存入12维向量的第5个槽位

                # --- 组合位置Token ---
                # HDF5中位置参数对应 px, py, pz的量化值
                ix_quant = int(quantized_params[8])
                iy_quant = int(quantized_params[9])
                iz_quant = int(quantized_params[10])

                if all(idx != self.pad_val for idx in [ix_quant, iy_quant, iz_quant]):
                    pos_token = iz_quant * (self.pos_grid_size ** 2) + iy_quant * self.pos_grid_size + ix_quant
                    final_args[6] = pos_token # 存入12维向量的第6个槽位
                
                # --- 其他拉伸参数 ---
                # s, e1, e2, b, u 对应 HDF5 中索引 11 到 15
                for i in range(5):
                    if quantized_params[11 + i] != self.pad_val:
                        final_args[7 + i] = int(quantized_params[11 + i])

            # 对于 SOL 和 EOS, final_args 保持为 PAD
            
            args_ids.append(final_args)
            
        return command_ids, args_ids

# --- 用于将浮点数转换为Token的辅助函数 (参考) ---
# 注意：这些函数在处理H5数据时不再直接使用，但保留作为逻辑参考和未来使用
    
    def quantize_val(self, val):
        """将 [-1, 1] 的浮点数映射到 [0, n_bins-1], 用于通用参数"""
        val = np.clip(val, self.cfg.min_val, self.cfg.max_val)
        norm = (val - self.cfg.min_val) / (self.cfg.max_val - self.cfg.min_val)
        return int(norm * (self.cfg.n_bins - 1))

    def encode_angle(self, theta, phi, gamma):
        """
        将三个欧拉角 (范围 [0, 2*pi]) 编码为一个 <A_n> Token ID。
        """
        theta_norm = np.clip(theta, 0, 2 * np.pi) / (2 * np.pi)
        phi_norm = np.clip(phi, 0, 2 * np.pi) / (2 * np.pi)
        gamma_norm = np.clip(gamma, 0, 2 * np.pi) / (2 * np.pi)

        t_idx = int(theta_norm * (self.angle_bins - 1))
        p_idx = int(phi_norm * (self.angle_bins - 1))
        g_idx = int(gamma_norm * (self.angle_bins - 1))
        
        token_id = g_idx * (self.angle_bins ** 2) + p_idx * self.angle_bins + t_idx
        return token_id

# 测试代码
if __name__ == '__main__':
    class MockConfig:
        def __init__(self):
            # 属性以匹配CADtokenizer的__init__
            self.pad_val = -1
            self.cad_n_args = 12
            self.cad_n_args_sketch = 5
            self.angle_bins = 9
            self.pos_grid_size = 36
            # 以下属性在tokenize_from_vec中不直接使用，但为保持兼容性而提供
            self.n_bins = 256
            self.sketch_bins = 128
            self.min_val = -1.0
            self.max_val = 1.0
        
        # 这些属性的getter在当前测试中不是必需的，但为了完整性而保留
        @property
        def n_angle_tokens(self): return self.angle_bins ** 3
        @property
        def n_pos_tokens(self): return self.pos_grid_size ** 3
        @property
        def n_sketch_tokens(self): return self.sketch_bins * 2

    cfg = MockConfig()
    tok = CADtokenizer(cfg)
    
    # --- 测试 tokenize_from_vec ---
    print("\n--- Testing tokenize_from_vec ---")
    # 模拟从HDF5文件读取的数据, 使用正确的命令ID
    # CAD_COMMANDS = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
    # ID ->           0,     1,      2,       3,     4,     5
    mock_h5_data = np.array([
        [  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], # SOL
        [  2, 176, 128, -1, -1, 48, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], # Circle
        [  5, -1, -1, -1, -1, -1, 6, 4, 2, 18, 18, 18, 30, -1, -1, -1, -1],   # Extrude, t=6, p=4, g=2, x=18, y=18, z=18, s=30
        [  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  # EOS
    ], dtype=np.int64)

    command_ids, args_ids = tok.tokenize_from_vec(mock_h5_data)

    print("原始H5数据 (部分):")
    print(mock_h5_data[:4, :])
    
    print("\n转换后的 Command IDs:")
    print(command_ids)

    print("\n转换后的 Args IDs (12维):")
    for i, args in enumerate(args_ids):
        cmd_name = tok.id2cmd[command_ids[i]]
        print(f"  {cmd_name}: {args}")

    # 验证Extrude的组合Token
    ext_args = args_ids[2]
    expected_angle_token = 2 * (9**2) + 4 * 9 + 6 # g*81 + p*9 + t
    expected_pos_token = 18 * (36**2) + 18 * 36 + 18 # z*1296 + y*36 + x
    print(f"\n验证 'Ext' 的组合Token:")
    print(f"  - Angle Token: {ext_args[5]} (预期: {expected_angle_token})")
    print(f"  - Pos Token: {ext_args[6]} (预期: {expected_pos_token})")
    assert ext_args[5] == expected_angle_token
    assert ext_args[6] == expected_pos_token
    print("  组合Token测试通过！")

    # 验证Circle的参数
    circle_args = args_ids[1]
    assert circle_args[0] == 176
    assert circle_args[1] == 128
    assert circle_args[4] == 48
    print("\n验证 'Circle' 的参数: 通过！")
    
    # 验证SOL和EOS
    assert args_ids[0] == [-1] * 12
    assert args_ids[3] == [-1] * 12
    print("验证 'SOL' 和 'EOS' 的参数: 通过！")
