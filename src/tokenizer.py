import json
import math
import numpy as np
from scipy.spatial.transform import Rotation
import sys
import os

# 将项目根目录添加到sys.path，确保可以导入config.macro
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import macro # 导入 macro.py

class CADTokenizer:
    """
    一个分词器，用于将CAD模型的连续几何参数转换为离散的Token ID。
    它主要负责参数值的量化，命令的索引直接来自 config/macro.py。
    """
    def __init__(self, vocab_path='config/arg_vocab.json'):
        """
        初始化分词器，加载参数值的词汇表。
        """
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        self.token_to_id = {}
        # 1. 加载特殊 token (<EMPTY>)
        self.token_to_id.update(vocab_data['special_tokens'])
        
        # 2. 动态生成并加载量化值 token (P, A, V, BOOL, EXTENT, SX, SY)
        self.value_ranges = vocab_data['value_ranges']
        for key, value in self.value_ranges.items():
            # Add a check for "SX_tokens" and "SY_tokens" specifically to build the token strings
            if key == "SX_tokens":
                # For SX_tokens, generate tokens like <SX_0>, <SX_1>, ...
                for i in range(value['num_bins']):
                    self.token_to_id[f"<SX_{i}>"] = value['start_id'] + i
            elif key == "SY_tokens":
                # For SY_tokens, generate tokens like <SY_0>, <SY_1>, ...
                for i in range(value['num_bins']):
                    self.token_to_id[f"<SY_{i}>"] = value['start_id'] + i
            else:
                # For other tokens (V, A, P, BOOL, EXTENT), use the general prefix_N format
                prefix = key.split('_')[0]
                start_id = value['start_id']
                num_bins = value['num_bins']
                for i in range(num_bins):
                    self.token_to_id[f"<{prefix}_{i}>"] = start_id + i

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # 常用 Token ID
        self.empty_token_id = self.token_to_id.get('<EMPTY>') # 0号位，用于参数向量中的空槽位
    
    def get_vocab_size(self):
        """获取参数值 Token 词汇表的大小"""
        return len(self.token_to_id)

    def _normalize_value(self, value, min_val, max_val):
        """将一个值从 [min_val, max_val] 归一化到 [0, 1]"""
        if max_val <= min_val: # 避免除以零或范围为零的情况
            return 0.5 # 返回中间值
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)

    def _quantize_value(self, normalized_value, num_bins):
        """将一个 [0, 1] 范围内的归一化值量化到 [0, num_bins-1] 的整数 bin"""
        return math.floor(normalized_value * (num_bins - 1e-6))

    def _get_bbox_max_dim(self, bbox):
        """计算边界框的最大维度，用于归一化"""
        min_pt = bbox['min_point']
        max_pt = bbox['max_point']
        max_dim = max(
            max_pt['x'] - min_pt['x'],
            max_pt['y'] - min_pt['y'],
            max_pt['z'] - min_pt['z']
        )
        return max_dim if max_dim > 1e-6 else 1.0

    def tokenize_scalar_param(self, raw_value, bbox):
        """将通用标量参数（如radius, distance）量化为一个 V_token ID (256 bins)"""
        max_dim = self._get_bbox_max_dim(bbox)
        # 假设参数值范围大致在 [-max_dim/2, max_dim/2]
        normalized_val = self._normalize_value(raw_value, -max_dim / 2, max_dim / 2)
        
        v_bin = self._quantize_value(normalized_val, self.value_ranges['V_tokens']['num_bins'])
        return self.value_ranges['V_tokens']['start_id'] + v_bin

    def tokenize_sketch_2d_param(self, raw_value, bbox, axis_type):
        """将2D草图坐标（X或Y）量化为一个 SX_token 或 SY_token ID (128 bins)"""
        if axis_type not in ['X', 'Y']:
            raise ValueError("axis_type must be 'X' or 'Y'")

        token_key = f"S{axis_type}_tokens"
        num_bins = self.value_ranges[token_key]['num_bins'] # Should be 128
        
        max_dim = self._get_bbox_max_dim(bbox)
        # For 2D sketch coordinates, assuming they are within [-max_dim/2, max_dim/2] range
        normalized_val = self._normalize_value(raw_value, -max_dim / 2, max_dim / 2)

        s_bin = self._quantize_value(normalized_val, num_bins)
        return self.value_ranges[token_key]['start_id'] + s_bin

    def tokenize_3d_point_param(self, point_dict, bbox):
        """将 3D 点 (例如草图平面原点) 量化为一个 P_token ID"""
        k = int(round(self.value_ranges['P_tokens']['num_bins']**(1/3))) # 应该是36
        max_dim = self._get_bbox_max_dim(bbox)
        min_pt_x = bbox['min_point']['x']
        min_pt_y = bbox['min_point']['y']
        min_pt_z = bbox['min_point']['z']

        # 将点坐标归一化到 [0, 1]
        norm_x = self._normalize_value(point_dict['x'], min_pt_x, min_pt_x + max_dim)
        norm_y = self._normalize_value(point_dict['y'], min_pt_y, min_pt_y + max_dim)
        norm_z = self._normalize_value(point_dict['z'], min_pt_z, min_pt_z + max_dim)

        ix = self._quantize_value(norm_x, k)
        iy = self._quantize_value(norm_y, k)
        iz = self._quantize_value(norm_z, k)

        # 索引计算 (z -> y -> x 顺序)
        p_index = iz * (k**2) + iy * k + ix
        return self.value_ranges['P_tokens']['start_id'] + p_index

    def tokenize_axes_param(self, x_axis, y_axis, z_axis):
        """将轴向量定义的旋转矩阵转换为欧拉角，再量化为一个 A_token ID"""
        try:
            rot_matrix = np.array([
                [x_axis['x'], y_axis['x'], z_axis['x']],
                [x_axis['y'], y_axis['y'], z_axis['y']],
                [x_axis['z'], y_axis['z'], z_axis['z']]
            ])
            r = Rotation.from_matrix(rot_matrix)
            euler_angles = r.as_euler('xyz', degrees=False)
            theta, phi, gamma = euler_angles[0], euler_angles[1], euler_angles[2]
        except Exception:
            theta, phi, gamma = 0.0, 0.0, 0.0
        
        n_bins = int(round(self.value_ranges['A_tokens']['num_bins']**(1/3))) # 9

        # 将角度从 [-pi, pi] 归一化到 [0, 1]
        norm_theta = (theta + np.pi) / (2 * np.pi)
        norm_phi = (phi + np.pi) / (2 * np.pi)
        norm_gamma = (gamma + np.pi) / (2 * np.pi)
        
        i_theta = self._quantize_value(norm_theta, n_bins)
        i_phi = self._quantize_value(norm_phi, n_bins)
        i_gamma = self._quantize_value(norm_gamma, n_bins)
        
        a_index = i_theta * (n_bins**2) + i_phi * n_bins + i_gamma
        return self.value_ranges['A_tokens']['start_id'] + a_index
        
    def tokenize_boolean_op_param(self, operation_str):
        """将拉伸操作字符串量化为一个 BOOL_token ID"""
        mapping = self.value_ranges['BOOL_tokens']['mapping']
        # 反转 mapping
        str_to_idx = {v: int(k) for k, v in mapping.items()}
        op_idx = str_to_idx.get(operation_str, 1) # 默认为 Join
        return self.value_ranges['BOOL_tokens']['start_id'] + op_idx

    def tokenize_extent_type_param(self, extent_type_str):
        """将挤出范围类型字符串量化为一个 EXTENT_token ID"""
        mapping = self.value_ranges['EXTENT_tokens']['mapping']
        str_to_idx = {v: int(k) for k, v in mapping.items()}
        type_idx = str_to_idx.get(extent_type_str, 0) # 默认为 OneSide
        return self.value_ranges['EXTENT_tokens']['start_id'] + type_idx

    def create_argument_vector(self, command_type_str, raw_params, bbox):
        """
        根据命令类型和原始参数，构建一个 12 维的参数 Token ID 向量。
        """
        arg_vector = [self.empty_token_id] * macro.CAD_N_ARGS
        
        try:
            cmd_idx = macro.CAD_COMMANDS.index(command_type_str)
            arg_mask = macro.CAD_CMD_ARGS_MASK[cmd_idx]
        except ValueError:
            if command_type_str in ['SOL', 'EOS']:
                return arg_vector
            print(f"警告: 未知的命令类型 '{command_type_str}'，返回全空参数向量.\n")
            return arg_vector 

        if command_type_str == 'Line':
            if arg_mask[0]: arg_vector[0] = self.tokenize_sketch_2d_param(raw_params.get('end_point', {}).get('x', 0.0), bbox, 'X')
            if arg_mask[1]: arg_vector[1] = self.tokenize_sketch_2d_param(raw_params.get('end_point', {}).get('y', 0.0), bbox, 'Y')
        
        elif command_type_str == 'Arc':
            if arg_mask[0]: arg_vector[0] = self.tokenize_sketch_2d_param(raw_params.get('center_point', {}).get('x', 0.0), bbox, 'X')
            if arg_mask[1]: arg_vector[1] = self.tokenize_sketch_2d_param(raw_params.get('center_point', {}).get('y', 0.0), bbox, 'Y')
            if arg_mask[2]: arg_vector[2] = self.tokenize_scalar_param(raw_params.get('start_angle', 0.0), bbox)
            if arg_mask[3]: arg_vector[3] = self.tokenize_scalar_param(raw_params.get('end_angle', 0.0), bbox)

        elif command_type_str == 'Circle':
            if arg_mask[0]: arg_vector[0] = self.tokenize_sketch_2d_param(raw_params.get('center_point', {}).get('x', 0.0), bbox, 'X')
            if arg_mask[1]: arg_vector[1] = self.tokenize_sketch_2d_param(raw_params.get('center_point', {}).get('y', 0.0), bbox, 'Y')
            if arg_mask[4]: arg_vector[4] = self.tokenize_scalar_param(raw_params.get('radius', 0.0), bbox)
        
        elif command_type_str == 'Ext':
            ext_args_start_idx = macro.CAD_N_ARGS_SKETCH
            transform = raw_params.get('transform', {})
            x_axis = transform.get('x_axis')
            y_axis = transform.get('y_axis')
            z_axis = transform.get('z_axis')
            origin_pt = transform.get('origin')
            extent_one_dist = raw_params.get('extent_one', {}).get('distance', {}).get('value')
            extent_two_dist = raw_params.get('extent_two', {}).get('distance', {}).get('value', 0.0)
            operation_type = raw_params.get('operation')
            extent_type = raw_params.get('extent_type')
            
            if arg_mask[ext_args_start_idx + 0]: 
                if x_axis and y_axis and z_axis and all(isinstance(v, dict) for v in [x_axis, y_axis, z_axis]):
                    arg_vector[ext_args_start_idx + 0] = self.tokenize_axes_param(x_axis, y_axis, z_axis)
                else:
                    arg_vector[ext_args_start_idx + 0] = self.tokenize_axes_param({'x':1.,'y':0.,'z':0.}, {'x':0.,'y':1.,'z':0.}, {'x':0.,'y':0.,'z':1.})
            
            if arg_mask[ext_args_start_idx + 1] and origin_pt and all(k in origin_pt for k in ['x','y','z']):
                arg_vector[ext_args_start_idx + 1] = self.tokenize_3d_point_param(origin_pt, bbox)
            
            if arg_mask[ext_args_start_idx + 2]:
                 arg_vector[ext_args_start_idx + 2] = self.tokenize_scalar_param(self._get_bbox_max_dim(bbox), bbox) 

            if arg_mask[ext_args_start_idx + 3] and extent_one_dist is not None:
                arg_vector[ext_args_start_idx + 3] = self.tokenize_scalar_param(extent_one_dist, bbox)
            
            if arg_mask[ext_args_start_idx + 4]:
                arg_vector[ext_args_start_idx + 4] = self.tokenize_scalar_param(extent_two_dist, bbox)

            if arg_mask[ext_args_start_idx + 5] and operation_type:
                arg_vector[ext_args_start_idx + 5] = self.tokenize_boolean_op_param(operation_type)
            
            if arg_mask[ext_args_start_idx + 6] and extent_type:
                arg_vector[ext_args_start_idx + 6] = self.tokenize_extent_type_param(extent_type)
        
        return arg_vector

    def decode_token(self, token_id):
        """将单个 token ID 解码回可读的 token 字符串"""
        return self.id_to_token.get(token_id, "<unk>")
        
    def decode_sequence(self, token_ids):
        """将一系列 token ID 解码回可读的 token 字符串列表"""
        return [self.decode_token(token_id) for token_id in token_ids]

if __name__ == '__main__':
    print("正在初始化CADTokenizer...")
    tokenizer = CADTokenizer(vocab_path='config/arg_vocab.json')
    print(f"参数值和特殊Token词汇表大小: {tokenizer.get_vocab_size()}")
    print(f"<empty> token ID: {tokenizer.empty_token_id}")

    mock_bbox = {'min_point': {'x': -10.0, 'y': -10.0, 'z': -10.0}, 'max_point': {'x': 10.0, 'y': 10.0, 'z': 10.0}}
    print(f"\n使用模拟边界框: {mock_bbox}")

    op_str = "CutFeatureOperation"
    bool_token_id = tokenizer.tokenize_boolean_op_param(op_str)
    print(f"布尔操作 '{op_str}' -> Token ID: {bool_token_id} -> 解码: {tokenizer.decode_token(bool_token_id)}")
    
    extent_str = "TwoSidesFeatureExtentType"
    extent_token_id = tokenizer.tokenize_extent_type_param(extent_str)
    print(f"挤出范围类型 '{extent_str}' -> Token ID: {extent_token_id} -> 解码: {tokenizer.decode_token(extent_token_id)}")

    ext_raw_params = {
        'transform': {'origin': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'x_axis': {'x': 1, 'y': 0, 'z': 0}, 'y_axis': {'x': 0, 'y': 1, 'z': 0}, 'z_axis': {'x': 0, 'y': 0, 'z': 1}},
        'extent_one': {'distance': {'value': 15.0}},
        'extent_two': {'distance': {'value': -5.0}},
        'operation': 'CutFeatureOperation',
        'extent_type': 'SymmetricFeatureExtentType'
    }
    ext_arg_vec = tokenizer.create_argument_vector('Ext', ext_raw_params, mock_bbox)
    print(f"\n解码 Extrude 参数向量 (槽位 5-11):")
    print(tokenizer.decode_sequence(ext_arg_vec[5:]))
    
    # NEW: Test sketch 2D tokens
    print(f"\n测试 Line 命令的 2D 草图坐标 Token (X=5.0, Y=3.0):")
    line_raw_params = {
        'end_point': {'x': 5.0, 'y': 3.0}
    }
    line_arg_vec = tokenizer.create_argument_vector('Line', line_raw_params, mock_bbox)
    print(f"X Token ID: {line_arg_vec[0]} -> 解码: {tokenizer.decode_token(line_arg_vec[0])}")
    print(f"Y Token ID: {line_arg_vec[1]} -> 解码: {tokenizer.decode_token(line_arg_vec[1])}")
