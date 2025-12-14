# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
import sys
import os
import numpy as np

# 假设这个脚本位于 src/ 目录下, 为了导入同级目录的 tokenizer
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.tokenizer import CADtokenizer
# 假设主配置文件可以被导入，如果不能，则使用下面的MockConfig
# from config.config import Config 

class CADDataset(Dataset):
    """
    用于加载和处理 CAD .h5 向量数据的 PyTorch 数据集。
    """
    def __init__(self, data_dir, config, max_len=60):
        """
        初始化数据集。

        Args:
            data_dir (str): 包含 .h5 文件的根目录 (例如 'data/raw/cad_vec')。
            config: 包含所有配置的对象，用于初始化 CADtokenizer。
            max_len (int): 序列的最大长度，用于填充。
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.file_paths = sorted(list(self.data_dir.rglob('*.h5')))
        
        if not self.file_paths:
            print(f"警告：在目录 '{data_dir}' 中没有找到任何 .h5 文件。")

        self.tokenizer = CADtokenizer(config)
        self.max_len = max_len
        self.pad_cmd_id = self.tokenizer.cmd_vocab['PAD']
        self.pad_args = [self.tokenizer.pad_val] * self.tokenizer.cad_n_args

    def __len__(self):
        """返回数据集中文件的总数。"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        获取并处理索引为 idx 的数据样本。
        """
        file_path = self.file_paths[idx]
        
        with h5py.File(file_path, 'r') as f:
            vec_data = f['vec'][:] # type: ignore
        
        # 使用tokenizer将H5的中间格式数据转换为最终的token序列
        command_ids, args_ids = self.tokenizer.tokenize_from_vec(vec_data)
        
        # --- 填充/截断序列到最大长度 ---
        seq_len = len(command_ids)
        
        # 1. 填充/截断命令
        if seq_len < self.max_len:
            padding_len = self.max_len - seq_len
            padded_command_ids = command_ids + [self.pad_cmd_id] * padding_len
        else:
            padded_command_ids = command_ids[:self.max_len]
        
        # 2. 填充/截断参数
        if seq_len < self.max_len:
            padding_len = self.max_len - seq_len
            padded_args_ids = args_ids + [self.pad_args] * padding_len
        else:
            padded_args_ids = args_ids[:self.max_len]
        
        # 转换为 PyTorch 张量
        command_tensor = torch.LongTensor(padded_command_ids)
        args_tensor = torch.LongTensor(padded_args_ids)
        
        # 对于自回归模型，通常输入是 target[:-1], 目标是 target[1:]
        # 在这里，我们返回完整的序列，具体的切分在训练循环中处理
        return command_tensor, args_tensor

# --- 用于测试的示例代码 ---
if __name__ == '__main__':
    # 模拟一个与简化后tokenizer相匹配的Config类
    class MockConfig:
        def __init__(self):
            # 关键属性，必须与tokenizer的__init__匹配
            self.pad_val = -1
            self.cad_n_args = 12
            self.cad_n_args_sketch = 5
            self.angle_bins = 9
            self.pos_grid_size = 36
            # 以下属性在新的tokenizer中不直接使用，但为保持兼容性而提供
            self.n_bins = 256
            self.sketch_bins = 128
            self.min_val = -1.0
            self.max_val = 1.0

        # 以下属性的getter也不再是测试的关键
        @property
        def n_angle_tokens(self): return self.angle_bins ** 3
        @property
        def n_pos_tokens(self): return self.pos_grid_size ** 3
        @property
        def n_sketch_tokens(self): return self.sketch_bins * 2

    mock_cfg = MockConfig()
    
    # 为了测试，先手动创建一个假的h5文件
    test_dir = 'data/raw/cad_vec_test'
    if not os.path.exists(f'{test_dir}/0000'):
        os.makedirs(f'{test_dir}/0000')
    
    # 使用与tokenizer词汇表一致的正确命令ID
    # CAD_COMMANDS = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
    # ID ->           0,     1,      2,       3,     4,     5
    mock_h5_data = np.array([
        [  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], # SOL
        [  2, 176, 128, -1, -1, 48, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], # Circle
        [  5, -1, -1, -1, -1, -1, 6, 4, 2, 18, 18, 18, 30, -1, -1, -1, -1],   # Extrude
        [  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  # EOS
    ], dtype=np.int64)

    test_h5_path = f'{test_dir}/0000/test.h5'
    with h5py.File(test_h5_path, 'w') as f:
        f.create_dataset('vec', data=mock_h5_data)

    print(f"创建了一个假的 HDF5 文件用于测试: {test_h5_path}")

    # 创建 Dataset 实例
    dataset = CADDataset(data_dir=test_dir, config=mock_cfg, max_len=10)
    
    print(f"\n找到 {len(dataset)} 个数据文件。\n")
    
    # 获取第一个样本
    if len(dataset) > 0:
        commands, args = dataset[0]
        
        print("获取的第一个样本 (已填充到长度10):")
        print("命令张量 (shape):", commands.shape)
        print("命令张量 (内容):", commands)
        
        print("\n参数张量 (shape):", args.shape)
        print("参数张量 (内容):")
        print(args)

        # 验证Extrude的组合Token是否正确
        # 原始数据中Extrude在第3行(索引2)，命令ID为5
        # 编码后它在最终张量的第3行(索引2)
        ext_args_tensor = args[2]
        expected_angle_token = 2 * (9**2) + 4 * 9 + 6 # g*81 + p*9 + t = 204
        expected_pos_token = 18 * (36**2) + 18 * 36 + 18 # z*1296 + y*36 + x = 23994
        
        print(f"\n验证 'Ext' 命令的参数 (第3行):")
        print(f"  - 角度Token: {ext_args_tensor[5]} (预期: {expected_angle_token})")
        print(f"  - 位置Token: {ext_args_tensor[6]} (预期: {expected_pos_token})")
        assert ext_args_tensor[5] == expected_angle_token
        assert ext_args_tensor[6] == expected_pos_token
        print("  验证通过！")

    else:
        print("没有找到数据，无法测试。")
