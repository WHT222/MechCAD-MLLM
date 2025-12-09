from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask_svg, _get_key_padding_mask_svg


class SVGEmbedding(nn.Module):#处理SVG输入的embedding
    """Embedding: view embed + command embed + parameter embed + positional embed
    嵌入方式：视图嵌入+命令嵌入+参数嵌入+位置嵌入
    1x: 命令嵌入 + 参数嵌入 + 位置嵌入
    3x or 4x: 视图嵌入 + 命令嵌入 + 参数嵌入 + 位置嵌入
    """
    def __init__(self, cfg, seq_len):
        super().__init__()

        """concatenation-based"""
        # 3x or 4x
        if cfg.input_option == "3x" or cfg.input_option == "4x":#处理3倍或4倍输入选项
            self.view_embed = nn.Embedding(4, 4)#视图嵌入维度为4
            self.command_embed = nn.Embedding(cfg.svg_n_commands, 8)#命令嵌入维度为8
        # 1x: keep dimension constant with other input option#1x：保持与其他输入选项的维度恒定
        if cfg.input_option == "1x":
            self.command_embed = nn.Embedding(cfg.svg_n_commands, 12)

        args_dim = cfg.args_dim + 1
        self.args_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.args_mlp = nn.Linear(64 * cfg.svg_n_args, 128)
        self.mlp = nn.Linear(4 + 8 + 128, cfg.d_model)
        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)
        
    
    def forward(self, view, command, args):
        assert command.shape == view.shape
        S, N = command.shape

        command_embedding = self.command_embed(command.long())
        args_embedding = self.args_mlp(self.args_embed((args + 1).long()).view(S, N, -1))

        """concatenation-based"""
        # 1x
        if S == 100:
            src = torch.cat([command_embedding, args_embedding], dim=-1)
        # 3x or 4x
        if S > 100:
            view_embedding = self.view_embed(view.long())
            src = torch.cat([view_embedding, command_embedding, args_embedding], dim=-1)
        
        src = self.mlp(src)
        src = self.pos_encoding(src)

        return src
    
class ConstEmbedding(nn.Module):
    """
    learned constant embedding
    学习常数嵌入
    """
    def __init__(self, cfg):
        super().__init__()

        self.d_model = cfg.d_model
        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=cfg.cad_max_total_len)
        self.seq_len = cfg.cad_max_total_len

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        view_num = int(cfg.input_option[0])
        seq_len = view_num * cfg.svg_max_total_len
        self.embedding = SVGEmbedding(cfg, seq_len)

        encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)
    
    def forward(self, view, command, args):
        assert command.shape == view.shape
        padding_mask, key_padding_mask = _get_padding_mask_svg(command, seq_dim=0), _get_key_padding_mask_svg(command, seq_dim=0)
    
        src = self.embedding(view, command, args)

        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask)

        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z)
        return z

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

class CommandDecoder(nn.Module):#命令嵌入，继承nn.Module（这是用于定义神经网络模块的基类）
    def __init__(self, cfg):
        super(CommandDecoder, self).__init__()

        self.embedding = ConstEmbedding(cfg)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        self.fcn = CommandFCN(cfg.d_model, cfg.cad_n_commands)

    def forward(self, z):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)

        command_logits = self.fcn(out)

        # guidance
        return command_logits, out

class ArgsDecoder(nn.Module):
    def __init__(self, cfg):
        super(ArgsDecoder, self).__init__()

        self.embedding = ConstEmbedding(cfg)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        args_dim = cfg.args_dim + 1
        self.fcn = ArgsFCN(cfg.d_model, cfg.cad_n_args, args_dim)

    def forward(self, z, guidance):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)

        # guidance
        out = out + guidance

        args_logits = self.fcn(out)

        return args_logits

class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model // 2),
                                        nn.GELU(),
                                        nn.Linear(cfg.d_model // 2, cfg.d_model))

    def forward(self, z):
        return z + self.bottleneck(z)

class SVG2CADTransformer(nn.Module):
    def __init__(self, cfg):
        super(SVG2CADTransformer, self).__init__()

        self.args_dim = cfg.args_dim + 1

        self.encoder = Encoder(cfg)

        self.bottleneck = Bottleneck(cfg)

        self.command_decoder = CommandDecoder(cfg)
        self.args_decoder = ArgsDecoder(cfg)

    def forward(self, views_enc, commands_enc, args_enc):
        views_enc_, commands_enc_, args_enc_ = _make_seq_first(views_enc, commands_enc, args_enc)  # Possibly None, None, None

        z = self.encoder(views_enc_, commands_enc_, args_enc_)
        z = self.bottleneck(z)
        
        """command-guided generation"""
        command_logits, guidance = self.command_decoder(z)
        command_logits = _make_batch_first(command_logits)
        args_logits = self.args_decoder(z, guidance)
        args_logits = _make_batch_first(args_logits)

        res = {
            "command_logits": command_logits,
            "args_logits": args_logits
        }

        return res