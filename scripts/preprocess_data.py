import os
import json
import numpy as np
from tqdm import tqdm
import argparse
import sys

# 将项目根目录添加到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.tokenizer import CADTokenizer
from config import macro

def process_single_cad_file(json_path, tokenizer):
    """
    处理单个 Omni-CAD JSON 文件，将其转换为 (N, 13) 的指令矩阵。
    其中 N 是序列长度, 13 = 1 (命令索引) + 12 (参数Token ID)。

    Args:
        json_path (str): Omni-CAD json 文件的路径。
        tokenizer (CADTokenizer): 分词器实例。

    Returns:
        np.ndarray: 形状为 (N, 13) 的 numpy 数组，如果处理失败则返回 None。
    """
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: 无法从 {json_path} 解码 JSON")
            return None

    bbox = data.get('properties', {}).get('bounding_box')
    if not bbox:
        print(f"警告: 在 {json_path} 中未找到边界框")
        return None

    sequence_vectors = []

    # 遍历顶层操作序列 (Sketch, ExtrudeFeature, etc.)
    for op_entry in data.get('sequence', []):
        op_type = op_entry.get('type')
        entity_id = op_entry.get('entity')
        if not op_type or not entity_id:
            continue

        entity = data.get('entities', {}).get(entity_id)
        if not entity:
            continue

        # 1. 分解 Sketch 操作
        if op_type == 'Sketch':
            # 对于每个Sketch，我们将其分解为 SOL -> [Line/Arc/Circle...] -> EOS
            for profile in entity.get('profiles', {}).values():
                for loop in profile.get('loops', []):
                    # --- 添加 SOL (Start of Loop) 命令 ---
                    sol_cmd_idx = macro.CAD_COMMANDS.index('SOL')
                    # SOL 命令没有参数，其参数向量是全空的
                    sol_arg_vec = [tokenizer.empty_token_id] * macro.CAD_N_ARGS
                    sequence_vectors.append([sol_cmd_idx] + sol_arg_vec)
                    
                    # --- 添加几何图元命令 ---
                    for curve in loop.get('profile_curves', []):
                        curve_type = curve.get('type')
                        cmd_str = None
                        
                        # 直接将 curve 字典作为 raw_params 传递
                        raw_params = curve

                        if curve_type == 'Line3D':
                            cmd_str = 'Line'
                        elif curve_type == 'Circle3D':
                            cmd_str = 'Circle'
                        elif curve_type == 'Arc3D':
                            cmd_str = 'Arc'

                        if cmd_str:
                            cmd_idx = macro.CAD_COMMANDS.index(cmd_str)
                            arg_vec = tokenizer.create_argument_vector(cmd_str, raw_params, bbox)
                            sequence_vectors.append([cmd_idx] + arg_vec)
                    
                    # --- 添加 EOS (End of Sketch/Loop) 命令 ---
                    eos_cmd_idx = macro.CAD_COMMANDS.index('EOS')
                    eos_arg_vec = [tokenizer.empty_token_id] * macro.CAD_N_ARGS
                    sequence_vectors.append([eos_cmd_idx] + eos_arg_vec)
        
        # 2. 处理 ExtrudeFeature 操作
        elif op_type == 'ExtrudeFeature':
            cmd_str = 'Ext'
            cmd_idx = macro.CAD_COMMANDS.index(cmd_str)
            # ExtrudeFeature 的所有信息都在 entity 字典中，直接作为 raw_params 传入
            # 我们需要将它关联的 Sketch 的 transform 信息也传入，因为 Extrude 命令需要它
            profiles = entity.get('profiles', [])
            if profiles:
                # (注意: 这是一个简化，假设 Extrude 总是使用其关联的第一个 profile 的 sketch 的 transform)
                associated_sketch_id = profiles[0].get('sketch')
                if associated_sketch_id:
                    sketch_entity = data.get('entities', {}).get(associated_sketch_id, {})
                    entity['transform'] = sketch_entity.get('transform', {})

            arg_vec = tokenizer.create_argument_vector(cmd_str, entity, bbox)
            sequence_vectors.append([cmd_idx] + arg_vec)

    if not sequence_vectors:
        return None

    # 将列表转换为 numpy 数组
    final_sequence = np.array(sequence_vectors, dtype=np.int32)
    
    # 截断或填充到最大长度
    num_steps = final_sequence.shape[0]
    if num_steps > macro.CAD_MAX_TOTAL_LEN:
        # 截断
        final_sequence = final_sequence[:macro.CAD_MAX_TOTAL_LEN, :]
    else:
        # 填充
        padding_len = macro.CAD_MAX_TOTAL_LEN - num_steps
        # 使用 PAD_VAL (-1) 进行填充
        # 注意：这里的 PAD_VAL 是-1，不是 tokenizer 中的 pad_token_id
        # 模型在处理时需要能识别这种填充
        padding = np.full((padding_len, 1 + macro.CAD_N_ARGS), macro.PAD_VAL, dtype=np.int32)
        # 将命令列的填充值设为 EOS_IDX，参数列设为 PAD_VAL (-1)
        padding[:, 0] = macro.CAD_EOS_IDX 
        final_sequence = np.vstack([final_sequence, padding])
    
    return final_sequence


def main(args):
    # 1. 初始化 Tokenizer
    # Tokenizer 负责将浮点数参数量化为整数ID
    try:
        tokenizer = CADTokenizer(vocab_path=args.vocab_path)
    except FileNotFoundError:
        print(f"错误: 词汇表文件未找到于 '{args.vocab_path}'。请确保文件存在。")
        sys.exit(1)
    except ImportError as e:
        if 'scipy' in str(e):
            print("错误: 缺少 'scipy' 库，请运行 'pip install scipy' 进行安装。")
            sys.exit(1)
        raise

    # 2. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 查找所有 json 文件
    json_files = []
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    print(f"找到 {len(json_files)} 个 JSON 文件进行处理。")

    # 4. 遍历并处理每个文件
    for json_path in tqdm(json_files, desc="预处理CAD文件"):
        processed_sequence = process_single_cad_file(json_path, tokenizer)
        
        if processed_sequence is not None:
            # 确定输出路径，保持原始目录结构
            relative_path = os.path.relpath(json_path, args.data_dir)
            output_path = os.path.join(args.output_dir, os.path.splitext(relative_path)[0] + '.npz')
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 以压缩格式保存 numpy 数组
            np.savez_compressed(
                output_path,
                cad_sequence=processed_sequence
            )

    print(f"预处理完成。处理后的文件保存在 {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将 Omni-CAD JSON 数据预处理为 (N, 13) 的指令矩阵。")
    parser.add_argument('--data_dir', type=str, default='data/raw/json', help="Omni-CAD 数据集的 JSON 文件根目录")
    parser.add_argument('--output_dir', type=str, default='data/processed', help="保存处理后的 .npz 文件的目录。")
    parser.add_argument('--vocab_path', type=str, default='config/arg_vocab.json', help="参数值Token词汇表的路径。")
    
    args = parser.parse_args()
    main(args)