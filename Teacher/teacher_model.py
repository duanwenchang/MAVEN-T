from __future__ import division
import torch
import torch.nn as nn
from utils import outputActivation
import math
from einops import rearrange, repeat
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from config import device
import numpy as np

BS = 128


class MambaBlock(nn.Module):
    """
    Mamba状态空间模型块，用于高效的时序建模
    实现线性时间复杂度O(T)的序列处理
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 简化的状态空间参数，避免复杂的selective scan
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.activation = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        return: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # 输入投影和门控
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split(split_size=self.d_inner, dim=-1)

        # 1D卷积
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d l -> b l d')

        # 激活函数
        x = self.activation(x)

        # 简化的门控机制
        x = x * self.activation(res)
        output = self.out_proj(x)

        return output


class MoETransformerBlock(nn.Module):
    """
    专家混合Transformer块
    实现计算效率和表示能力的平衡
    """

    def __init__(self, d_model, n_heads, d_ff, n_experts=8, k=2, dropout=0.1):
        super(MoETransformerBlock, self).__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k  # top-k专家选择

        # 多头注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_experts)
        ])

        # 门控网络
        self.gate = nn.Linear(d_model, n_experts)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        """
        x: (batch_size, seq_len, d_model)
        """
        # 多头自注意力
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + attn_output

        # 专家混合前馈网络
        residual = x
        x = self.norm2(x)

        # 门控计算
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)

        gate_logits = self.gate(x_flat)  # (batch_size * seq_len, n_experts)
        gate_scores = F.softmax(gate_logits, dim=-1)

        # Top-k专家选择
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)
        topk_scores = F.softmax(topk_scores, dim=-1)  # 重新归一化

        # 专家计算
        expert_outputs = torch.zeros_like(x_flat)
        for i in range(self.k):
            expert_idx = topk_indices[:, i]
            expert_mask = F.one_hot(expert_idx, num_classes=self.n_experts)

            for expert_id in range(self.n_experts):
                mask = expert_mask[:, expert_id].bool()
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    expert_outputs[mask] += topk_scores[:, i:i + 1][mask] * expert_output

        x = expert_outputs.view(batch_size, seq_len, d_model)
        x = residual + x

        return x


class EnhancedGraphConvolution(nn.Module):
    """
    增强的图卷积网络，基于原始代码进行MAVEN-T风格的改进
    """

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, heads=8):
        super(EnhancedGraphConvolution, self).__init__()

        self.activation = nn.ELU()
        self.gru = torch.nn.GRU(16, hidden_size=1, batch_first=True)
        # 使用GATv2替代原始的GAT
        self.gat1 = GATv2Conv((39 * 2), hidden_channels, heads=heads, dropout=0.1)
        self.gat2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, dropout=0.1)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.Conv2d(8, 16, kernel_size, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout(0.3, inplace=False)
        )

    def forward(self, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, mask_view_batch,
                graph_matrix):
        edge_index_batch = edge_index_batch.to(device)
        mask_view_batch = mask_view_batch.to(device)
        man_matrix_batch = man_matrix_batch.to(device)
        ac_matrix_batch = ac_matrix_batch.to(device)
        ve_matrix_batch = ve_matrix_batch.to(device)

        # NaN值处理
        has_nan = torch.isnan(man_matrix_batch)
        man_matrix_batch = torch.where(has_nan, torch.tensor(0.0, device=device), man_matrix_batch)
        has_nan = torch.isnan(ac_matrix_batch)
        ac_matrix_batch = torch.where(has_nan, torch.tensor(0.0, device=device), ac_matrix_batch)
        has_nan = torch.isnan(ve_matrix_batch)
        ve_matrix_batch = torch.where(has_nan, torch.tensor(0.0, device=device), ve_matrix_batch)

        man_matrix_batch1 = torch.unsqueeze(man_matrix_batch, dim=1)
        ac_matrix_batch1 = torch.unsqueeze(ac_matrix_batch, dim=1)
        ve_matrix_batch1 = torch.unsqueeze(ve_matrix_batch, dim=1)
        conv_matrix = torch.cat((man_matrix_batch1, ac_matrix_batch1, ve_matrix_batch1), dim=1)
        conv_matrix = self.conv(conv_matrix)

        outputs = []
        for i in range(conv_matrix.size(3)):
            part = conv_matrix[:, :, :, i]
            part = part.permute(0, 2, 1)
            out, _ = self.gru(part)
            outputs.append(out)
        conv_enc1 = torch.cat(outputs, dim=-1)

        mask_view_batch = torch.flatten(mask_view_batch, start_dim=1, end_dim=2)
        mask_view_batch = mask_view_batch.unsqueeze(1)
        conv_enc2 = conv_enc1 * mask_view_batch
        man_matrix_batch2 = man_matrix_batch * mask_view_batch
        graph_matrix = graph_matrix.to(device)
        graph_matrix = torch.cat((man_matrix_batch2, conv_enc2), dim=1)
        graph_matrix = graph_matrix.permute(0, 2, 1)

        x = graph_matrix.reshape(-1, (39 * 2))
        edge_index = edge_index_batch.view(2, -1)
        h = self.gat1(x, edge_index.long())
        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gat2(h, edge_index.long())
        h = F.dropout(h, p=0.2, training=self.training)
        output = h.view(BS, 39, 64)
        return output


class HybridAttentionInformer(nn.Module):
    """
    混合注意力Informer，集成Mamba和传统Transformer
    """

    def __init__(self):
        super(HybridAttentionInformer, self).__init__()
        # 保留原始Transformer但添加Mamba处理
        self.transformer = nn.Transformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
        self.mamba_temporal = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2)
        self.dropout = nn.Dropout(0)

    def forward(self, space_details, time_details):
        space_details = space_details.view(BS, 39, 64)
        time_details = time_details.view(BS, 30, 64)

        # 先通过Mamba进行时序处理
        time_enhanced = self.mamba_temporal(time_details)

        # 再通过Transformer进行全局建模
        transformer_output = self.transformer(space_details.permute(1, 0, 2), time_enhanced.permute(1, 0, 2))
        transformer_output = self.dropout(transformer_output)

        return transformer_output.permute(1, 0, 2)


class highwayNet(nn.Module):
    """
    MAVEN-T教师网络 - 基于原始代码进行增强
    """

    def __init__(self, args):
        super(highwayNet, self).__init__()

        self.args = args
        self.use_cuda = args['use_cuda']
        self.use_maneuvers = args['use_maneuvers']
        self.train_flag = args['train_flag']
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size = (((args['grid_size'][0] - 4) + 1) // 2) * self.conv_3x1_depth
        self.in_channels = args['in_channels']
        self.out_channels = args['out_channels']
        self.kernel_size = args['kernel_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.dropout = args['dropout']
        self.nbr_max = args['nbr_max']
        self.hidden_channels = args['hidden_channels']
        self.lat_length = 3
        self.lon_length = 3

        # 保留原始组件但进行增强
        self.Decoder = Decoder(args=args)
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)
        self.up_emb = torch.nn.Linear(1, self.input_embedding_size)
        self.linear1 = nn.Linear(6, 32)
        self.linear2 = nn.Linear(6, 32)
        self.activation = nn.ELU()
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)
        self.gru = torch.nn.GRU(self.input_embedding_size, self.encoder_size, 2, batch_first=True)
        self.lstm = nn.LSTM(self.input_embedding_size, self.encoder_size)
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)

        # 增强的图卷积
        self.gcn = EnhancedGraphConvolution(self.in_channels, self.hidden_channels, self.out_channels, self.kernel_size)

        # 混合注意力Informer
        self.informer = HybridAttentionInformer()

        # 保留原始注意力组件
        self.qt = nn.Linear(self.encoder_size, self.n_head * self.att_out)
        self.kt = nn.Linear(self.encoder_size, self.n_head * self.att_out)
        self.vt = nn.Linear(self.encoder_size, self.n_head * self.att_out)
        self.addAndNorm = AddAndNorm(self.encoder_size)
        self.first_glu = GLU(
            input_size=self.n_head * self.att_out,
            hidden_layer_size=self.encoder_size,
            dropout_rate=self.dropout)
        self.second_glu = GLU(
            input_size=self.encoder_size,
            hidden_layer_size=self.encoder_size,
            dropout_rate=self.dropout)
        self.normalize = nn.LayerNorm(self.encoder_size)
        self.mu_fc1 = nn.Linear(self.encoder_size, self.n_head * self.att_out)
        self.mu_fc = nn.Linear(self.n_head * self.att_out, self.encoder_size)

        # 可学习映射矩阵
        self.mapping = torch.nn.Parameter(
            torch.Tensor(self.in_length, self.out_length, self.lat_length + self.lon_length))

        # 原始卷积层
        self.soc_conv = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3, 1))
        self.soc_maxpool = torch.nn.MaxPool2d((2, 1), padding=(1, 0))

        # 解码器
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(61, self.decoder_size)
        else:
            self.dec_lstm = torch.nn.LSTM(39 + 16 + 3 + 3, self.decoder_size)

        # 输出层
        self.op = torch.nn.Linear(self.decoder_size, 5)
        self.op_lat = torch.nn.Linear(self.encoder_size, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(self.encoder_size, self.num_lon_classes)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        # 添加MoE解码器
        self.moe_decoder = nn.ModuleList([
            MoETransformerBlock(
                d_model=self.encoder_size,
                n_heads=4,
                d_ff=self.encoder_size * 2,
                n_experts=4,
                k=2,
                dropout=0.1
            ) for _ in range(2)
        ])

        # 序列长度转换
        self.tea_exchange = torch.nn.Linear(31, 30)

    def forward(self, hist, nbrs, masks, lat_enc, lon_enc, lane, nbrslane, cls, nbrscls, va, nbrsva, edge_index_batch,
                ve_matrix_batch, ac_matrix_batch, man_matrix_batch, mask_view_batch, graph_matrix):

        # 环境感知图编码
        space_details = self.gcn(edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, mask_view_batch,
                                 graph_matrix)

        # 历史轨迹编码
        hist1 = torch.cat((hist, cls, lane, va), -1)
        nbrs1 = torch.cat((nbrs, nbrscls, nbrslane, nbrsva), -1)
        hist_enc = self.activation(self.linear1(hist1))
        hist_hidden_enc, (_, _) = self.lstm(hist_enc)
        time_self_enc = hist_hidden_enc.permute(1, 0, 2)
        time_self_enc1 = hist_hidden_enc.permute(1, 2, 0)
        time_self_enc1 = self.tea_exchange(time_self_enc1)
        time_self_enc = time_self_enc1.permute(0, 2, 1)

        # 邻居车辆编码
        nbrs_enc = self.activation(self.linear1(nbrs1))
        nbrs_hidden_enc, (_, _) = self.lstm(nbrs_enc)
        mask = masks.view(masks.size(0), masks.size(1) * masks.size(2), masks.size(3))
        mask = repeat(mask, 'b g s -> t b g s', t=self.in_length)
        soc_enc = torch.zeros_like(mask).float()
        time_nbrs_enc = soc_enc.masked_scatter_(mask, nbrs_hidden_enc)

        # 混合注意力机制
        query = self.qt(time_self_enc)
        _, _, embed_size = query.shape
        query = torch.cat(torch.split(torch.unsqueeze(query, 2), int(embed_size / self.n_head), -1), 1)
        keys = torch.cat(torch.split(self.kt(time_nbrs_enc), int(embed_size / self.n_head), -1), 0).permute(1, 0, 3, 2)
        values = torch.cat(torch.split(self.vt(time_nbrs_enc), int(embed_size / self.n_head), -1), 0).permute(1, 0, 2,
                                                                                                              3)

        a = torch.matmul(query, keys)
        a /= math.sqrt(self.encoder_size)
        a = torch.softmax(a, -1)
        values = torch.matmul(a, values)
        values = torch.cat(torch.split(values, int(hist.shape[0] - 1), 1), -1)
        values = values.squeeze(2)
        time_values, _ = self.first_glu(values)
        time_detiles = self.addAndNorm(time_self_enc, time_values)

        # 混合Informer处理
        result = self.informer(space_details, time_detiles)

        # MoE解码器处理
        moe_input = result.permute(1, 0, 2)  # 转换为 (batch, seq, feature)
        for moe_block in self.moe_decoder:
            moe_input = moe_block(moe_input)

        enc, _ = self.second_glu(moe_input.permute(1, 0, 2))  # 转换回 (seq, batch, feature)
        enc1 = enc[:, -1, :]

        if self.use_maneuvers:
            maneuver_state = self.activation(self.mu_fc1(enc1))
            maneuver_state = self.activation(self.normalize(self.mu_fc(maneuver_state)))
            lat_pred = self.softmax(self.op_lat(maneuver_state))
            lon_pred = self.softmax(self.op_lon(maneuver_state))

            if self.train_flag:
                lat_man = torch.argmax(lat_pred, dim=-1).detach().unsqueeze(1)
                lon_man = torch.argmax(lon_pred, dim=-1).detach().unsqueeze(1)
                lat_enc_tmp = torch.zeros_like(lat_pred)
                lon_enc_tmp = torch.zeros_like(lon_pred)
                lat_man = lat_enc_tmp.scatter_(1, lat_man, 1)
                lon_man = lon_enc_tmp.scatter_(1, lon_man, 1)
                index = torch.cat((lat_man, lon_man), dim=-1).permute(-1, 0)
                mapping = F.softmax(torch.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
                dec = torch.matmul(mapping, enc).permute(1, 0, 2)
                fut_pred = self.Decoder(dec, lat_man, lon_man)
                return fut_pred, lat_pred, lon_pred
            else:
                out = []
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        index = torch.cat((lat_enc_tmp, lon_enc_tmp), dim=-1).permute(-1, 0)
                        mapping = F.softmax(torch.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
                        dec = torch.matmul(mapping, enc).permute(1, 0, 2)
                        fut_pred = self.Decoder(dec, lat_enc_tmp, lon_enc_tmp)
                        out.append(fut_pred)
                return out, lat_pred, lon_pred
        else:
            fut_pred = self.Decoder(enc)
            return fut_pred


class Decoder(nn.Module):
    """保持原始解码器不变"""

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.use_cuda = args['use_cuda']
        self.use_maneuvers = args['use_maneuvers']
        self.train_flag = args['train_flag']
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size = (((args['grid_size'][0] - 4) + 1) // 2) * self.conv_3x1_depth
        self.in_channels = args['in_channels']
        self.out_channels = args['out_channels']
        self.kernel_size = args['kernel_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.dropout = args['dropout']
        self.nbr_max = args['nbr_max']
        self.hidden_channels = args['hidden_channels']
        self.lat_length = 3
        self.lon_length = 3
        if self.use_maneuvers:
            self.mu_f = 16
        else:
            self.mu_f = 0
        self.activation = nn.ELU()

        self.lstm = nn.LSTM(self.encoder_size, self.encoder_size)
        self.linear1 = nn.Linear(self.encoder_size, 5)
        self.lat_linear = nn.Linear(self.lat_length, 8)
        self.lon_linear = nn.Linear(self.lon_length, 8)

        self.dec_linear = nn.Linear(self.encoder_size + self.lat_length + self.lon_length, self.encoder_size)

    def forward(self, dec, lat_enc, lon_enc):
        if self.use_maneuvers or hasattr(self, 'cat_pred'):
            lat_enc = lat_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            lon_enc = lon_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            dec = torch.cat((dec, lat_enc, lon_enc), -1)
            dec = self.dec_linear(dec)
        h_dec, _ = self.lstm(dec)
        fut_pred = self.linear1(h_dec)
        return outputActivation(fut_pred)


class GLU(nn.Module):
    """保持原始GLU不变"""

    def __init__(self, input_size, hidden_layer_size, dropout_rate):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return torch.mul(activation, gated), gated


class AddAndNorm(nn.Module):
    """保持原始AddAndNorm不变"""

    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()
        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = torch.add(torch.add(x1, x2), x3)
        else:
            x = torch.add(x1, x2)
        return self.normalize(x)