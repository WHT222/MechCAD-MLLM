import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import random # 用于随机采样视图
from tqdm import tqdm
import glob # 用于查找多个视图文件

# 将项目根目录添加到sys.path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import macro # 导入 macro.py

class CADDataset(Dataset):
    def __init__(self, processed_data_dir, raw_text_dir=None, raw_image_dir=None, split='train', sample_limit=None, num_views_to_sample=1):
        """
        初始化 CAD 数据集。

        Args:
            processed_data_dir (str): 预处理后的 .npz 文件根目录 (例如 'data/processed').
            raw_text_dir (str, optional): 原始文本描述文件 (例如 'data/raw/text/0000.json') 的路径.
            raw_image_dir (str, optional): 原始图像文件的根目录 (例如 'data/raw/img').
            split (str, optional): 数据集划分 ('train', 'val', 'test').
            sample_limit (int, optional): 限制加载的样本数量，主要用于测试。默认为 None (加载所有)。
            num_views_to_sample (int): 每次为每个样本随机采样的视图数量。默认为 1。
        """
        self.processed_data_dir = processed_data_dir
        self.raw_text_dir = raw_text_dir
        self.raw_image_dir = raw_image_dir
        self.split = split # TODO: 实际的 split 逻辑需要文件列表或索引
        self.sample_limit = sample_limit
        self.num_views_to_sample = num_views_to_sample

        self.samples = []
        self._load_samples()

        self.text_captions = self._load_text_captions()

        # 图像预处理
        self.image_transform = Compose([
            Resize((224, 224)), # MLLM通常期望224x224
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 标准
        ])

    def _load_samples(self):
        """
        遍历预处理目录，收集所有 .npz 文件的路径。
        """
        print(f"正在收集 '{self.processed_data_dir}' 中的预处理样本...")
        collected_count = 0
        
        # 确保目录存在
        if not os.path.isdir(self.processed_data_dir):
            print(f"警告: 预处理目录不存在: {self.processed_data_dir}")
            return

        for root, _, files in os.walk(self.processed_data_dir):
            # 如果达到了样本限制，我们可以提前退出外层循环
            if self.sample_limit is not None and collected_count >= self.sample_limit:
                break
                
            for file in files:
                if file.endswith('.npz'):
                    # 获取文件相对于 self.processed_data_dir 的子目录路径
                    subdir = os.path.relpath(root, self.processed_data_dir)
                    # 获取文件名（不含后缀）
                    filestem = os.path.splitext(file)[0]
                    
                    # 构造一个总是一致的 sample_id，格式为 'subdir/filestem'
                    # 即使 subdir 是 '.' (表示在根目录)，os.path.join 也能正确处理
                    # 为了跨平台兼容，统一使用 '/' 作为分隔符
                    if subdir == '.':
                        sample_id = filestem
                    else:
                        sample_id = os.path.join(subdir, filestem).replace(os.path.sep, '/')

                    self.samples.append({
                        'id': sample_id,
                        'npz_path': os.path.join(root, file)
                    })
                    collected_count += 1
                    if self.sample_limit is not None and collected_count >= self.sample_limit:
                        print(f"达到样本限制 {self.sample_limit}，停止收集。")
                        break # 退出内层循环
        print(f"收集到 {len(self.samples)} 个样本。")

    def _load_text_captions(self):
        """
        加载所有文本描述到内存中。
        """
        if not self.raw_text_dir or not os.path.exists(self.raw_text_dir):
            return {}
        
        captions = {}
        # 假设 text_dir 是一个包含多个JSON文件的目录，或者是一个单一的JSON文件
        if os.path.isdir(self.raw_text_dir):
            for filename in os.listdir(self.raw_text_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.raw_text_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            captions[item['id']] = item['text caption']
        elif os.path.isfile(self.raw_text_dir) and self.raw_text_dir.endswith('.json'):
             with open(self.raw_text_dir, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    captions[item['id']] = item['text caption']
        
        print(f"加载了 {len(captions)} 条文本描述。")
        return captions

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引加载、处理并返回一个样本。
        """
        sample_info = self.samples[idx]
        sample_id = sample_info['id']
        npz_path = sample_info['npz_path']

        # 1. 加载预处理后的 CAD 序列
        cad_sequence_data = np.load(npz_path)
        cad_sequence = cad_sequence_data['cad_sequence'] # 形状 (macro.CAD_MAX_TOTAL_LEN, 13) 
        cad_sequence = torch.from_numpy(cad_sequence).long() # 转换为 LongTensor

        # 分离命令和参数，如果模型需要
        command_tokens = cad_sequence[:, 0]
        arg_tokens = cad_sequence[:, 1:]

        # 2. 加载文本描述
        text_caption = self.text_captions.get(sample_id, "无描述") # 如果没有找到，提供一个默认值

        # 3. 加载图像 (多视图处理)
        # 默认返回一个堆叠的黑色图像tensor
        image_tensors = torch.zeros(self.num_views_to_sample, 3, 224, 224) 

        if self.raw_image_dir:
            img_id_parts = sample_id.split('/') # 例如 ['0000', '00000068_00001']
            img_subdir = img_id_parts[0]
            img_base_name = img_id_parts[1] # 例如 '00000068_00001'

            # 查找所有可能的视图文件
            all_view_paths = []
            # 假设视图后缀是 _XXX 格式，查找 pattern: raw_image_dir/subdir/basename_*.png/jpg
            view_pattern = os.path.join(self.raw_image_dir, img_subdir, img_base_name + "_*")
            
            # 查找所有可能的图片文件（png, jpg, jpeg）
            for ext in ['.png', '.jpg', '.jpeg']:
                all_view_paths.extend(glob.glob(view_pattern + ext))
            
            # 如果没有找到带后缀的视图，尝试查找不带后缀的通用图片名
            if not all_view_paths:
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_generic_path = os.path.join(self.raw_image_dir, img_subdir, img_base_name + ext)
                    if os.path.exists(potential_generic_path):
                        all_view_paths.append(potential_generic_path)
                        break # 找到通用图片就够了

            # 从所有找到的视图中随机采样 num_views_to_sample 张
            if all_view_paths:
                selected_view_paths = random.sample(all_view_paths, min(len(all_view_paths), self.num_views_to_sample))
                
                loaded_images = []
                for img_path in selected_view_paths:
                    try:
                        image = Image.open(img_path).convert('RGB')
                        loaded_images.append(self.image_transform(image))
                    except Exception as e:
                        print(f"警告: 无法加载或处理图片 {img_path}: {e}")
                
                if loaded_images:
                    image_tensors = torch.stack(loaded_images) # 堆叠成 (k, C, H, W)
                else:
                    print(f"警告: 尽管找到图片路径，但未能加载 {sample_id} 的任何图片。")
            else:
                pass # print(f"警告: 未找到 {sample_id} 对应的任何视图图片。") # 频繁打印可能很吵

        return {
            'id': sample_id,
            'cad_sequence': cad_sequence, # 整个 (60, 13) 矩阵
            'command_tokens': command_tokens, # 1D命令序列
            'arg_tokens': arg_tokens,       # 2D参数序列
            'text_caption': text_caption,
            'image': image_tensors # 现在是 (k, 3, 224, 224)
        }

# 简单的测试 (在 __main__ 块中运行)
if __name__ == '__main__':
    # 假设你已经运行了 preprocess_data.py，并且输出在 data/processed
    mock_processed_dir = 'data/processed'
    mock_raw_text_file = 'data/raw/text/0000.json' # 假设文本在一个文件中
    mock_raw_image_dir = 'data/raw/step_img' # 假设图片在一个目录中

    # 创建一个 dummy 图片目录，以避免FileNotFoundError
    # 确保每个 ID 有多个视图，例如 '_000', '_001'
    os.makedirs(os.path.join(mock_raw_image_dir, '0000'), exist_ok=True)
    
    dummy_img_base = os.path.join(mock_raw_image_dir, '0000', '00000068_00001')
    if not os.path.exists(dummy_img_base + '_000.png'):
        Image.new('RGB', (100, 100), color = 'red').save(dummy_img_base + '_000.png')
    if not os.path.exists(dummy_img_base + '_001.png'):
        Image.new('RGB', (100, 100), color = 'blue').save(dummy_img_base + '_001.png')
    if not os.path.exists(dummy_img_base + '.png'): # 通用名也创建一个
        Image.new('RGB', (100, 100), color = 'green').save(dummy_img_base + '.png')


    print("正在初始化 CADDataset (采样1个视图)...")
    dataset_single_view = CADDataset(
        processed_data_dir=mock_processed_dir,
        raw_text_dir=mock_raw_text_file,
        raw_image_dir=mock_raw_image_dir,
        sample_limit=2, # 限制加载2个样本进行测试
        num_views_to_sample=1
    )
    print(f"数据集大小 (单视图): {len(dataset_single_view)} 个样本")

    if len(dataset_single_view) > 0:
        sample = dataset_single_view[0]
        print(f"\n第一个样本的 ID: {sample['id']}")
        print(f"CAD 序列形状: {sample['cad_sequence'].shape}")
        print(f"命令 Token 形状: {sample['command_tokens'].shape}")
        print(f"参数 Token 形状: {sample['arg_tokens'].shape}")
        print(f"文本描述: {sample['text_caption']}")
        print(f"图像 Tensor 形状 (单视图): {sample['image'].shape}") # 期望 (1, 3, 224, 224)
        print(f"CAD 序列前2行:\n{sample['cad_sequence'][:2]}")
    else:
        print("数据集中没有可用的样本。" )
    
    print("\n----------------------------------------------------")
    print("正在初始化 CADDataset (采样2个视图)...")
    dataset_multi_view = CADDataset(
        processed_data_dir=mock_processed_dir,
        raw_text_dir=mock_raw_text_file,
        raw_image_dir=mock_raw_image_dir,
        sample_limit=2, # 限制加载2个样本进行测试
        num_views_to_sample=2 # 采样2个视图
    )
    print(f"数据集大小 (多视图): {len(dataset_multi_view)} 个样本")
    if len(dataset_multi_view) > 0:
        sample_multi = dataset_multi_view[0]
        print(f"图像 Tensor 形状 (多视图): {sample_multi['image'].shape}") # 期望 (2, 3, 224, 224))