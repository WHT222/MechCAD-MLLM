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
    def __init__(self, vocab_path='config/vocab.json'):
        """
        初始化分词器，加载参数值的词汇表。
        """
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        self.token_to_id = {}
        # 1. 加载特殊 token
        self.token_to_id.update(vocab_data['special_tokens'])
        
        # 2. 动态生成并加载量化值 token (P, A, V, BOOL)
        self.value_ranges = vocab_data['value_token_ranges']
        for key, value in self.value_ranges.items():
            prefix = key.split('_')[0]
            start_id = value['start_id']
            num_bins = value['num_bins']
            for i in range(num_bins):
                self.token_to_id[f"<{prefix}_{i}>"] = start_id + i

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # 常用 Token ID
        self.sos_token_id = self.token_to_id.get('<sos>')
        self.eos_token_id = self.token_to_id.get('<eos>')
        self.pad_token_id = self.token_to_id.get('<pad>')
        self.unk_token_id = self.token_to_id.get('<unk>')
        self.empty_token_id = self.token_to_id.get('<empty>') # 用于参数向量中的空槽位
    
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
        """将标量参数（如x, y, radius, distance）量化为一个 V_token ID"""
        # 归一化策略：所有坐标和距离都相对于整个模型 bbox 的最大尺寸进行归一化
        max_dim = self._get_bbox_max_dim(bbox)
        # 假设参数值可能在 [-max_dim/2, max_dim/2] 之间，归一化到 [0, max_dim]
        # 这里的 min_val 和 max_val 需要根据实际数据范围调整
        normalized_val = self._normalize_value(raw_value, -max_dim/2, max_dim/2)
        
        v_bin = self._quantize_value(normalized_val, self.value_ranges['V_tokens']['num_bins'])
        return self.value_ranges['V_tokens']['start_id'] + v_bin

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
            # 构建旋转矩阵
            rot_matrix = np.array([
                [x_axis['x'], y_axis['x'], z_axis['x']],
                [x_axis['y'], y_axis['y'], z_axis['y']],
                [x_axis['z'], y_axis['z'], z_axis['z']]
            ])
            # 从旋转矩阵获取欧拉角 (弧度)，使用 'xyz' 约定
            r = Rotation.from_matrix(rot_matrix)
            euler_angles = r.as_euler('xyz', degrees=False)
            theta, phi, gamma = euler_angles[0], euler_angles[1], euler_angles[2]
        except Exception:
            # 如果转换失败（例如矩阵不正交），使用默认值 (0,0,0)
            theta, phi, gamma = 0.0, 0.0, 0.0
        
        n_bins = int(round(self.value_ranges['A_tokens']['num_bins']**(1/3))) # 应该是9

        # 将角度从 [-pi, pi] 归一化到 [0, 1]
        norm_theta = (theta + np.pi) / (2 * np.pi)
        norm_phi = (phi + np.pi) / (2 * np.pi)
        norm_gamma = (gamma + np.pi) / (2 * np.pi)
        
        i_theta = self._quantize_value(norm_theta, n_bins)
        i_phi = self._quantize_value(norm_phi, n_bins)
        i_gamma = self._quantize_value(norm_gamma, n_bins)
        
        # 索引计算
        a_index = i_theta * (n_bins**2) + i_phi * n_bins + i_gamma
        return self.value_ranges['A_tokens']['start_id'] + a_index
        
    def tokenize_boolean_op_param(self, operation_str):
        """将拉伸操作字符串（如 NewBody, Cut）量化为一个 BOOL_token ID"""
        op_map = {
            "NewBodyFeatureOperation": 0,
            "JoinFeatureOperation": 0,
            "CutFeatureOperation": 1,
            "IntersectFeatureOperation": 2,
        }
        op_idx = op_map.get(operation_str, 0) # 默认为 NewBody (索引 0)
        return self.value_ranges['BOOL_tokens']['start_id'] + op_idx

    def tokenize_extent_type_param(self, extent_type_str):
        """将挤出范围类型字符串（如 OneSide, Symmetric）量化为一个 BOOL_token ID"""
        type_map = {
            "OneSideFeatureExtentType": 0,
            "SymmetricFeatureExtentType": 1,
            "TwoSidesFeatureExtentType": 2,
        }
        type_idx = type_map.get(extent_type_str, 0) # 默认为 OneSide (索引 0)
        return self.value_ranges['BOOL_tokens']['start_id'] + type_idx

    def create_argument_vector(self, command_type_str, raw_params, bbox):
        """
        根据命令类型和原始参数，构建一个 12 维的参数 Token ID 向量。
        参数槽位和掩码来自 config.macro。

        Args:
            command_type_str (str): 命令类型字符串 (e.g., 'Line', 'Ext').
            raw_params (dict): 包含此命令所有原始几何参数的字典.
            bbox (dict): 整个 CAD 模型的边界框，用于归一化。

        Returns:
            list: 包含 12 个参数 Token ID 的列表。
        """
        arg_vector = [self.empty_token_id] * macro.CAD_N_ARGS
        
        # 获取命令索引和对应的参数掩码
        try:
            cmd_idx = macro.CAD_COMMANDS.index(command_type_str)
            arg_mask = macro.CAD_CMD_ARGS_MASK[cmd_idx]
        except ValueError:
            # 对于不在 CAD_COMMANDS 中的命令，返回全 empty 的向量
            # 例如 'SOL', 'EOL', 'SOS', 'EOS' 等命令在 macro.py 中，但其参数掩码是全0
            # 这里应确保这些命令有对应的索引，且掩码处理正确
            if command_type_str in ['SOL', 'EOS']: # 这些是 macro.py 中定义的，但其参数掩码是全0
                 return arg_vector
            print(f"警告: 未知的命令类型 '{command_type_str}'，返回全空参数向量。 সন")
            return arg_vector 

        # 填充参数槽位 (根据 macro.CAD_CMD_ARGS_MASK 索引)
        # CAD_N_ARGS_SKETCH = 5 # sketch parameters: x, y, alpha, f, r
        # CAD_N_ARGS_EXT = 7 # Extrude parameters start after sketch args
        
        # Line 命令: 占用 sketch args 的前两个 (end_x, end_y)
        if command_type_str == 'Line':
            # 假设 raw_params 包含 'end_x', 'end_y'
            # 检查掩码，确保对应槽位是活跃的
            if arg_mask[0]: # 槽位0: end_x
                arg_vector[0] = self.tokenize_scalar_param(raw_params.get('end_point', {}).get('x', 0.0), bbox)
            if arg_mask[1]: # 槽位1: end_y
                arg_vector[1] = self.tokenize_scalar_param(raw_params.get('end_point', {}).get('y', 0.0), bbox)
        
        # Arc 命令: 占用 sketch args 的前四个 (center_x, center_y, start_angle, end_angle)
        elif command_type_str == 'Arc':
            # 假设 raw_params 包含 'center_x', 'center_y', 'start_angle', 'end_angle' (或 sweep_angle, radius等)
            # 注意：OmniCAD 的 Arc 可能有 center_point, start_point, end_point, radius, start_angle, end_angle
            # 这里我们根据 macro.py 的参数掩码来填
            if arg_mask[0]: # 槽位0: center_x
                arg_vector[0] = self.tokenize_scalar_param(raw_params.get('center_point', {}).get('x', 0.0), bbox)
            if arg_mask[1]: # 槽位1: center_y
                arg_vector[1] = self.tokenize_scalar_param(raw_params.get('center_point', {}).get('y', 0.0), bbox)
            if arg_mask[2]: # 槽位2: start_angle
                arg_vector[2] = self.tokenize_scalar_param(raw_params.get('start_angle', 0.0), bbox)
            if arg_mask[3]: # 槽位3: end_angle
                arg_vector[3] = self.tokenize_scalar_param(raw_params.get('end_angle', 0.0), bbox)

        # Circle 命令: 占用 sketch args 的 x, y, radius (根据CAD_CMD_ARGS_MASK，radius是第5个参数，即索引4)
        elif command_type_str == 'Circle':
            # 假设 raw_params 包含 'center_x', 'center_y', 'radius'
            if arg_mask[0]: # 槽位0: center_x
                arg_vector[0] = self.tokenize_scalar_param(raw_params.get('center_point', {}).get('x', 0.0), bbox)
            if arg_mask[1]: # 槽位1: center_y
                arg_vector[1] = self.tokenize_scalar_param(raw_params.get('center_point', {}).get('y', 0.0), bbox)
            if arg_mask[4]: # 槽位4: radius
                arg_vector[4] = self.tokenize_scalar_param(raw_params.get('radius', 0.0), bbox)
        
        # Ext 命令: 占用 Extrude 的7个参数槽位 (从索引 macro.CAD_N_ARGS_SKETCH = 5 开始)
        elif command_type_str == 'Ext':
            ext_args_start_idx = macro.CAD_N_ARGS_SKETCH # 5
            
            # 1. Plane Orientation (theta, phi, gamma) -> 1个 A_token
            # 槽位5 (对应 CAD_N_ARGS_PLANE, 第一个 Ext 参数)
            transform = raw_params.get('transform', {})
            x_axis = transform.get('x_axis')
            y_axis = transform.get('y_axis')
            z_axis = transform.get('z_axis')
            if arg_mask[ext_args_start_idx + 0]: 
                # 检查轴向量是否存在
                if x_axis and y_axis and z_axis and all(isinstance(v, dict) for v in [x_axis, y_axis, z_axis]):
                    arg_vector[ext_args_start_idx + 0] = self.tokenize_axes_param(x_axis, y_axis, z_axis)
                else: # 默认情况，例如没有提供transform，或者数据格式不正确
                    # 如果没有 z_axis，可以尝试从 x, y 轴交叉积生成，但这里简化为默认 Identity
                    arg_vector[ext_args_start_idx + 0] = self.tokenize_axes_param(
                        {'x':1.0,'y':0.0,'z':0.0}, {'x':0.0,'y':1.0,'z':0.0}, {'x':0.0,'y':0.0,'z':1.0}, # 默认 Identity 变换
                    )

            # 2. Plane Origin (px, py, pz) -> 1个 P_token
            # 槽位6 (对应 CAD_N_ARGS_TRANS, 第二个 Ext 参数)
            origin_pt = transform.get('origin')
            if arg_mask[ext_args_start_idx + 1]: 
                if origin_pt and all(k in origin_pt for k in ['x','y','z']):
                    arg_vector[ext_args_start_idx + 1] = self.tokenize_3d_point_param(origin_pt, bbox)
            
            # 3. Scale 's' (模型缩放因子，OmniCAD中不直接体现，用最大尺寸代表) -> 1个 V_token
            # 槽位7 (对应 CAD_N_ARGS_TRANS, 第三个 Ext 参数)
            if arg_mask[ext_args_start_idx + 2]:
                 arg_vector[ext_args_start_idx + 2] = self.tokenize_scalar_param(self._get_bbox_max_dim(bbox), bbox) 

            # 4. Extrusion Distance e1 -> 1个 V_token
            # 槽位8 (对应 CAD_N_ARGS_EXT_PARAM, 第四个 Ext 参数)
            extent_one_dist = raw_params.get('extent_one', {}).get('distance', {}).get('value')
            if arg_mask[ext_args_start_idx + 3]:
                if extent_one_dist is not None:
                    arg_vector[ext_args_start_idx + 3] = self.tokenize_scalar_param(extent_one_dist, bbox)
            
            # 5. Extrusion Distance e2 (如果存在 TwoSides) -> 1个 V_token
            # 槽位9 (对应 CAD_N_ARGS_EXT_PARAM, 第五个 Ext 参数)
            if arg_mask[ext_args_start_idx + 4]:
                extent_two_dist = raw_params.get('extent_two', {}).get('distance', {}).get('value', 0.0) # 默认为0
                arg_vector[ext_args_start_idx + 4] = self.tokenize_scalar_param(extent_two_dist, bbox)

            # 6. Boolean Operation 'b' -> 1个 BOOL_token
            # 槽位10 (对应 CAD_N_ARGS_EXT_PARAM, 第六个 Ext 参数)
            operation_type = raw_params.get('operation')
            if arg_mask[ext_args_start_idx + 5]:
                if operation_type:
                    arg_vector[ext_args_start_idx + 5] = self.tokenize_boolean_op_param(operation_type)
            
            # 7. Extent Type 'u' -> 1个 BOOL_token
            # 槽位11 (对应 CAD_N_ARGS_EXT_PARAM, 第七个 Ext 参数)
            extent_type = raw_params.get('extent_type')
            if arg_mask[ext_args_start_idx + 6]:
                if extent_type:
                    arg_vector[ext_args_start_idx + 6] = self.tokenize_extent_type_param(extent_type)
        
        # 对于 SOL/EOL/EOS 等命令，参数槽位都为空 (会保持为 self.empty_token_id)
        # 这些命令在 macro.py 的 CAD_CMD_ARGS_MASK 中应该都是全 0

        return arg_vector


    def decode_token(self, token_id):
        """将单个 token ID 解码回可读的 token 字符串"""
        return self.id_to_token.get(token_id, "<unk>")
        
    def decode_sequence(self, token_ids):
        """将一系列 token ID 解码回可读的 token 字符串列表"""
        return [self.decode_token(token_id) for token_id in token_ids]

if __name__ == '__main__':
    # 简单的测试
    # 在运行前请确保 `pip install scipy numpy`
    print("正在初始化CADTokenizer...")
    tokenizer = CADTokenizer(vocab_path='config/vocab.json')
    print(f"参数值和特殊Token词汇表大小: {tokenizer.get_vocab_size()}")
    print(f"<empty> token ID: {tokenizer.empty_token_id}")

    # 模拟一个边界框
    mock_bbox = {'min_point': {'x': -10.0, 'y': -10.0, 'z': -10.0}, 'max_point': {'x': 10.0, 'y': 10.0, 'z': 10.0}}
    print(f"\n使用模拟边界框: {mock_bbox}")

    # 1. 测试一个标量 (例如 Line 的 end_x)
    val_x = 5.0
    token_id_x = tokenizer.tokenize_scalar_param(val_x, mock_bbox)
    print(f"标量值 {val_x} -> Token ID: {token_id_x} -> 解码: {tokenizer.decode_token(token_id_x)}")

    # 2. 测试一个3D点 (例如 Extrude 的 origin)
    point_3d = {'x': 2.0, 'y': -3.0, 'z': 4.0}
    token_id_3d = tokenizer.tokenize_3d_point_param(point_3d, mock_bbox)
    print(f"3D点 {point_3d} -> Token ID: {token_id_3d} -> 解码: {tokenizer.decode_token(token_id_3d)}")

    # 3. 测试一组轴 (例如 Extrude 的 transform)
    x_ax = {'x': 1, 'y': 0, 'z': 0}
    y_ax = {'x': 0, 'y': 1, 'z': 0}
    z_ax = {'x': 0, 'y': 0, 'z': 1} # 默认正向
    axes_token_id = tokenizer.tokenize_axes_param(x_ax, y_ax, z_ax)
    print(f"一组轴向量 (Identity) -> Token ID: {axes_token_id} -> 解码: {tokenizer.decode_token(axes_token_id)}")

    # 4. 测试布尔操作
    op_str = "CutFeatureOperation"
    bool_token_id = tokenizer.tokenize_boolean_op_param(op_str)
    print(f"布尔操作 '{op_str}' -> Token ID: {bool_token_id} -> 解码: {tokenizer.decode_token(bool_token_id)}")
    
    # 5. 测试挤出范围类型
    extent_str = "TwoSidesFeatureExtentType"
    extent_token_id = tokenizer.tokenize_extent_type_param(extent_str)
    print(f"挤出范围类型 '{extent_str}' -> Token ID: {extent_token_id} -> 解码: {tokenizer.decode_token(extent_token_id)}")


    # --- 测试 create_argument_vector ---
    print("\n--- 测试 create_argument_vector ---")
    
    # 模拟 Line 命令参数
    line_raw_params = {'end_point': {'x': 5.0, 'y': 3.0}} # OmniCAD 的 Line 有 start_point 和 end_point
    line_arg_vec = tokenizer.create_argument_vector('Line', line_raw_params, mock_bbox)
    print(f"Line 参数向量: {line_arg_vec[:2]}... (前2个参数)")
    print(f"解码 Line 参数向量: {tokenizer.decode_sequence(line_arg_vec)}")

    # 模拟 Arc 命令参数
    arc_raw_params = {
        'center_point': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'start_angle': 0.0,
        'end_angle': np.pi / 2
    }
    arc_arg_vec = tokenizer.create_argument_vector('Arc', arc_raw_params, mock_bbox)
    print(f"Arc 参数向量: {arc_arg_vec[:4]}... (前4个参数)")
    print(f"解码 Arc 参数向量: {tokenizer.decode_sequence(arc_arg_vec)}")

    # 模拟 Circle 命令参数
    circle_raw_params = {
        'center_point': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'radius': 7.5
    }
    circle_arg_vec = tokenizer.create_argument_vector('Circle', circle_raw_params, mock_bbox)
    print(f"Circle 参数向量: {circle_arg_vec[0]}, {circle_arg_vec[1]}, {circle_arg_vec[4]}... (x, y, r)")
    print(f"解码 Circle 参数向量: {tokenizer.decode_sequence(circle_arg_vec)}")


    # 模拟 Extrude 命令参数
    ext_raw_params = {
        'transform': {'origin': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 
                      'x_axis': {'x': 1, 'y': 0, 'z': 0},
                      'y_axis': {'x': 0, 'y': 1, 'z': 0},
                      'z_axis': {'x': 0, 'y': 0, 'z': 1}},
        'extent_one': {'distance': {'value': 15.0}},
        'extent_two': {'distance': {'value': -5.0}},
        'operation': 'CutFeatureOperation',
        'extent_type': 'SymmetricFeatureExtentType'
    }
    ext_arg_vec = tokenizer.create_argument_vector('Ext', ext_raw_params, mock_bbox)
    print(f"Ext 参数向量: {ext_arg_vec[5:]} (后7个参数)") # Ext参数从索引5开始
    print(f"解码 Ext 参数向量: {tokenizer.decode_sequence(ext_arg_vec)}")
