import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration # Assuming transformers will be installed

from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask_svg, _get_key_padding_mask_svg


class ConstEmbedding(nn.Module):
    """
    learned constant embedding
    学习常数嵌入,
    """
    def __init__(self, cfg):
        super().__init__()

        self.d_model = cfg.d_model
        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=cfg.cad_max_total_len)#这是位置编码器
        self.seq_len = cfg.cad_max_total_len

    def forward(self, z):
        # z: [Seq_Len_LLM, Batch, Dim]
        N = z.size(1)
        # Create a zero tensor of shape [Seq_Len_CAD, Batch, Dim] and add PE
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src


class CommandFCN(nn.Module):
    def __init__(self, d_model, n_commands):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_commands)
        )
    def forward(self, out):
        command_logits = self.mlp(out)  # Shape [S, N, n_commands]

        return command_logits

class ArgsFCN(nn.Module):
    def __init__(self, d_model, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, n_args * args_dim)
        )
    def forward(self, out):
        S, N, _ = out.shape

        args_logits = self.mlp(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return args_logits

class CommandDecoder(nn.Module):
    def __init__(self, cfg):
        super(CommandDecoder, self).__init__()

        self.embedding = ConstEmbedding(cfg)

        # Use TransformerDecoderLayerImproved which supports Multihead Attention for memory
        # This allows attending to the full LLM sequence
        decoder_layer = TransformerDecoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        self.fcn = CommandFCN(cfg.d_model, cfg.cad_n_commands)

    def forward(self, z, memory_key_padding_mask=None):
        # z (memory) is [Seq_Len_LLM, Batch, d_model]
        src = self.embedding(z) # [Seq_Len_CAD, Batch, d_model]
        
        # Decoder attends to z (LLM features)
        out = self.decoder(src, z, tgt_mask=None, memory_mask=None, 
                           tgt_key_padding_mask=None, memory_key_padding_mask=memory_key_padding_mask)

        command_logits = self.fcn(out)

        # guidance
        return command_logits, out

class ArgsDecoder(nn.Module):
    def __init__(self, cfg):
        super(ArgsDecoder, self).__init__()

        self.embedding = ConstEmbedding(cfg)

        decoder_layer = TransformerDecoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        # 基础全连接网络，输出尺寸为 args_dim + 1 
        args_dim_features = cfg.args_dim + 1 
        self.fcn = ArgsFCN(cfg.d_model, cfg.cad_n_args, args_dim_features)

        # 独立的头部用于离散角度和位置Token
        self.cad_n_args = cfg.cad_n_args
        self.n_angle_tokens = cfg.n_angle_tokens
        self.n_pos_tokens = cfg.n_pos_tokens

        # These indices refer to the argument slots for angle and position tokens #
        # CAD_N_ARGS_SKETCH is 5, so angle_token is at index 5, pos_token at index 6
        from config.macro import CAD_N_ARGS_SKETCH
        self.angle_token_idx = CAD_N_ARGS_SKETCH + 0
        self.pos_token_idx = CAD_N_ARGS_SKETCH + 1

        self.angle_head = nn.Linear(args_dim_features, self.n_angle_tokens)
        self.pos_head = nn.Linear(args_dim_features, self.n_pos_tokens)


    def forward(self, z, guidance, memory_key_padding_mask=None):
        # z (memory) is [Seq_Len_LLM, Batch, d_model]
        # guidance is [Seq_Len_CAD, Batch, d_model] from CommandDecoder
        
        src = self.embedding(z) # [Seq_Len_CAD, Batch, d_model]
        
        # Inject guidance into the query
        # This helps the ArgsDecoder know which command it is generating args for
        src = src + guidance
        
        out = self.decoder(src, z, tgt_mask=None, memory_mask=None, 
                           tgt_key_padding_mask=None, memory_key_padding_mask=memory_key_padding_mask)

        # args_features: [S, N, cad_n_args, args_dim_features]
        args_features = self.fcn(out)

        # Extract features for discrete tokens and pass through their heads，额外的两个头部分别预测角度Token和位置Token
        angle_token_logits = self.angle_head(args_features[:, :, self.angle_token_idx, :])
        pos_token_logits = self.pos_head(args_features[:, :, self.pos_token_idx, :])

        # Return all as a tuple. The original args_features can be used for continuous args，
        # The calling LLM2CADDecoder will need to correctly interpret these.
        return args_features, angle_token_logits, pos_token_logits


    
# 进行修改，借用双解码器结构
class LLM2CADDecoder(nn.Module):
    def __init__(self, cfg, llm_hidden_dim=4096):
        """
        cfg: 沿用 Drawing2CAD 的配置对象
        llm_hidden_dim: 使用的 LLM 的输出维度 (例如 LLaMA-2-7B 是 4096)，可修改以适配不同的模型
        该解码器模块将 LLM 的输出特征转换为 CAD 命令和参数
        1. 通过线性层将 LLM 的高维特征适配到 CAD 解码器的维度
        2. 使用 Drawing2CAD 的双解码器结构，先预测命令，再预测参数
        """
        super().__init__()
        self.d_model = cfg.d_model # Drawing2CAD 默认是 256

        # 1. 适配层：将 LLM 的高维特征压缩到 CAD 解码器的维度
        self.adapter = nn.Linear(llm_hidden_dim, self.d_model)
        
        # 2. 复用 Drawing2CAD 的双解码器结构
        self.command_decoder = CommandDecoder(cfg)
        self.args_decoder = ArgsDecoder(cfg)

    def forward(self, llm_features, memory_key_padding_mask=None):
        """
        llm_features: [Batch_Size, LLM_Seq_Len, LLM_Hidden_Dim] 
                      这是 MLLM (如 LLaVA) 输出的语义向量
        memory_key_padding_mask: [Batch_Size, LLM_Seq_Len] (BoolTensor)
                                True indicates the position should be ignored.
        """
        # 步骤 A: 维度适配
        # z 的形状变为 [Batch_Size, LLM_Seq_Len, 256]
        z = self.adapter(llm_features)
        
        # 步骤 B: 维度重排以适配 Transformer [Seq_Len, Batch_Size, Dim]
        z = z.permute(1, 0, 2)

        # 步骤 C: 双解码生成
        # 1. 先预测命令，并输出 guidance (指导信号)
        # command_logits: [Seq_Len, Batch, n_commands]
        # guidance: [Seq_Len, Batch, d_model] -> 包含了解码器对“当前是什么命令”的理解
        # 这里 z 保持序列结构，解码器通过 Attention 机制关注 LLM 的特征
        command_logits, guidance = self.command_decoder(z, memory_key_padding_mask=memory_key_padding_mask)

        # 2. 再预测参数，将 guidance 加进去
        args_features_for_continuous, angle_token_logits, pos_token_logits = \
            self.args_decoder(z, guidance, memory_key_padding_mask=memory_key_padding_mask)

        # 调整输出维度为 [Batch, Seq_Len, ...] 以便计算 Loss
        command_logits = command_logits.permute(1, 0, 2)# [N, S, n_commands]
        args_features_for_continuous = args_features_for_continuous.permute(1, 0, 2, 3)# [N, S, n_args, args_dim]
        angle_token_logits = angle_token_logits.permute(1, 0, 2) # [N, S, n_angle_tokens]
        pos_token_logits = pos_token_logits.permute(1, 0, 2) # [N, S, n_pos_tokens]

        return command_logits, args_features_for_continuous, angle_token_logits, pos_token_logits


class LLaVAEncoder(nn.Module):
    def __init__(self, cfg, model_name="llava-hf/llava-1.5-7b-hf"):
        super().__init__()
        # Load LLaVA model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(model_name)
        self.llava_model.eval() # Set to eval mode, typically we freeze the MLLM encoder

        # Ensure that the hidden_size of LLaVA matches what LLM2CADDecoder expects
        # LLaVA-1.5-7B has a hidden size of 4096 for its language model outputs
        assert self.llava_model.config.hidden_size == cfg.llm_hidden_dim, \
            f"LLaVA model hidden size ({self.llava_model.config.hidden_size}) does not match cfg.llm_hidden_dim ({cfg.llm_hidden_dim})"

    def forward(self, images, texts):
        # Prepare inputs for LLaVA
        # images: List of PIL Image objects
        # texts: List of text strings (prompts)
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        # Move inputs to device (e.g., GPU)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.llava_model.device)

        with torch.no_grad(): # Don't compute gradients for the frozen LLaVA model
            # Get hidden states from LLaVA's language model part
            # output_hidden_states=True gives access to the encoder/decoder outputs
            outputs = self.llava_model(**inputs, output_hidden_states=True)
            
            # Usually, the last_hidden_state of the language model is used
            # It's typically [Batch, Sequence_Length, Hidden_Size]
            llm_features = outputs.language_model_outputs.last_hidden_state
            
            # The attention mask from LLaVA can be directly used for padding in our decoder
            memory_key_padding_mask = inputs.attention_mask.bool() # True means pad, False means not pad

        return llm_features, memory_key_padding_mask


class MechCAD(nn.Module):
    def __init__(self, cfg, llava_model_name="llava-hf/llava-1.5-7b-hf"):
        super().__init__()
        self.llava_encoder = LLaVAEncoder(cfg, model_name=llava_model_name)
        self.llm2cad_decoder = LLM2CADDecoder(cfg, llm_hidden_dim=cfg.llm_hidden_dim)

    def forward(self, images, texts):
        # Encode images and texts using LLaVA
        llm_features, memory_key_padding_mask = self.llava_encoder(images, texts)

        # Decode CAD commands and arguments
        command_logits, args_features_for_continuous, angle_token_logits, pos_token_logits = \
            self.llm2cad_decoder(llm_features, memory_key_padding_mask)

        return command_logits, args_features_for_continuous, angle_token_logits, pos_token_logits