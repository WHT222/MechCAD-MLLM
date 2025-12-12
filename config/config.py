import os
import argparse
import json
import shutil
from .file_utils import ensure_dirs
from .macro import *

class Config(object):#配置类
    """
    实验的配置类
    该类初始化超参数，解析命令行参数，并设置日志记录和模型保存的路径。
    """
    def __init__(self, phase):
        self.is_train = phase == "train"

        # 初始化超参数并从命令行解析
        parser, args = self.parse()

        self.set_configuration()

        # 显式声明关键属性，便于类型检查
        self.exp_name: str = args.exp_name
        self.proj_dir: str = args.proj_dir

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():#实现将命令行参数打印出来
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)#将命令行参数设置为配置类的属性

        # 路径处理逻辑
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if phase == "train" and args.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # save this configuration
        if self.is_train:
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def set_configuration(self):
        self.args_dim = ARGS_DIM # 256
        self.cad_n_args = CAD_N_ARGS
        self.svg_n_args = SVG_N_ARGS
        self.cad_n_commands = len(CAD_COMMANDS)  # Line, Arc, Circle, EOS, SOL, Ext
        self.svg_n_commands = len(SVG_COMMANDS)  # SOS, EOS, L, C

        self.n_layers = 4                # Number of Encoder blocks
        self.n_layers_decode = 4         # Number of Decoder blocks
        self.n_heads = 8                 # Transformer config: number of heads
        self.dim_feedforward = 512      # Transformer config: FF dimensionality，注意力机制中前馈神经网络的维度
        self.d_model = 256               # Transformer config: model dimensionality，即嵌入向量的维度
        self.dropout = 0.1               # Dropout rate used in basic layers and Transformers
        self.dim_z = 256                 # Latent vector dimensionality，即潜在向量的维度

        self.cad_max_n_ext = CAD_MAX_N_EXT # cad最大挤出数
        self.cad_max_n_loops = CAD_MAX_N_LOOPS # cad最大环数
        self.cad_max_n_curves = CAD_MAX_N_CURVES # cad最大曲线数

        self.cad_max_total_len = CAD_MAX_TOTAL_LEN # cad最大序列长度
        self.svg_max_total_len = SVG_MAX_TOTAL_LEN # svg最大序列长度

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0
        }

        # --- [新增] 适配 LLM 的参数 ---
        self.llm_hidden_dim = 4096  # LLaVA-7B/Vicuna-7B 的输出维度
        
        # --- [新增] 适配 CAD-GPT 空间机制的参数 ---
        # 2D Sketch 参数量化 
        self.n_bins = 256 
        self.sketch_bins = 128  # 2D草图坐标单坐标轴量化等级
        self.min_val = -1.0
        self.max_val = 1.0
        "n_bins与n_sketch_bins的区别在于，n_bins是通用参数量化等级，n_sketch_bins专门用于2D草图坐标的量化,实则没有必要区分，可以统一为256"
        
        # 3D Extrude 空间/角度 Token (CAD-GPT 相关)
        self.angle_bins = 9         # 欧拉角离散化粒度 (9档)
        self.pos_grid_size = 36     # 3D空间体素化粒度 (36x36x36)
        
    # 计算衍生参数 (方便调用)
    @property
    def n_angle_tokens(self):
        return self.angle_bins ** 3
            
    @property
    def n_pos_tokens(self):
        return self.pos_grid_size ** 3
    
    @property
    def m_sketch_bins(self):
        return self.sketch_bins * 2
        
    def parse(self):
        """
        初始化参数解析器。定义默认超参数并从命令行参数收集。
        返回:
            parser: 定义参数的ArgumentParser对象。
            args: 解析后的参数作为Namespace对象。
        """
        parser = argparse.ArgumentParser()

        parser.add_argument('--proj_dir', type=str, default="proj_log", help="path to project folder where models and logs will be saved")
        parser.add_argument('--data_root', type=str, default="data", help="path to source data folder")
        parser.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        parser.add_argument('-g', '--gpu_ids', type=str, default='0', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")        
        
        parser.add_argument('--batch_size', type=int, default=256, help="batch size")
        parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

        parser.add_argument('--nr_epochs', type=int, default=200, help="total number of epochs to train")
        parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
        parser.add_argument('--grad_clip', type=float, default=1.0, help="initial learning rate")
        parser.add_argument('--warmup_step', type=int, default=2000, help="step size for learning rate warm up")
        parser.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
        parser.add_argument('--save_frequency', type=int, default=100, help="save models every x epochs")
        parser.add_argument('--val_frequency', type=int, default=50, help="run validation every x iterations")
        parser.add_argument('--vis_frequency', type=int, default=2000, help="visualize output every x iterations")

        parser.add_argument('--input_option', type=str, default="3x", help="number of input views (1x, 3x, 4x)")
        
        args = parser.parse_args()
        return parser, args