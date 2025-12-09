import torch
from config.config import Config
from model.model import LLM2CADDecoder # 假设你把上面的类加到了 model.py

# 1. 模拟配置 (Mock Configuration)
# 这里手动设置一些 Drawing2CAD 需要的参数，避免加载整个配置文件
class MockConfig:
    d_model = 256 # 嵌入维度
    dim_z = 256 # 全局特征维度
    n_heads = 8 # 注意力头数
    dim_feedforward = 512 # 前馈网络维度
    dropout = 0.1 # dropout 比例
    cad_n_commands = 6  # Line, Arc, Circle, EOS, SOL, Ext
    cad_n_args = 16     # CAD 参数数量
    args_dim = 256      # 参数离散化的 bin 数量
    n_layers_decode = 4 # 解码器层数
    cad_max_total_len = 60 # 序列最大长度

cfg = MockConfig()

# 2. 实例化模型
model = LLM2CADDecoder(cfg, llm_hidden_dim=4096)
print("模型构建成功！")

# 3. 模拟输入数据
batch_size = 4
llm_seq_len = 50 # 假设 LLM 输出了 50 个 token
llm_dim = 4096   # 假设是 LLaMA-7B
fake_llm_output = torch.randn(batch_size, llm_seq_len, llm_dim)

# 4. 前向传播
try:
    cmd_logits, args_logits = model(fake_llm_output)
    print("前向传播成功！")
    print(f"Command Output Shape: {cmd_logits.shape}") # 预期: [4, 60, 6]
    print(f"Args Output Shape: {args_logits.shape}")   # 预期: [4, 60, 16, 257]
except Exception as e:
    print(f"出错啦: {e}")