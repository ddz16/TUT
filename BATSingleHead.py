"""
Boundary-Aware Transformer
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import math
from math import sqrt
import numpy as np
from collections import Counter
from loguru import logger
from eval import segment_bars_with_confidence
from utils import extract_dis_from_attention, class2boundary, KL_loss, create_distribution_from_cls
from einops import rearrange


def postprocess(batch_output, batch_length, batch_chunk):
    """
    only used when test and predict, batch size = 1
    :param batch_output: (1, L)
    :param batch_length: list with only one elements
    :param batch_chunk: list with only one elements
    :return: (1, batch_length[0])
    """
    if batch_chunk[0] != 1:
        # batch_output = F.interpolate(batch_output.unsqueeze(1), scale_factor=batch_chunk[0])
        # batch_output = batch_output[:, 0, :batch_length[0]]
        batch_output = F.interpolate(batch_output.unsqueeze(1), size=batch_length[0])
    return batch_output


def create_window_mask(l_seg, band_width):
    """
    :param l_seg: query segment length
    :param band_width: an odd, represents window size
    :return: a mask matrix with shape of (l_seg, l_seg + 2 * b//2), the positions of elements which participate in calculation are 1
    """
    mask = torch.ones(l_seg, l_seg + 2 * (band_width // 2))
    mask = torch.tril(mask, diagonal=band_width-1)
    mask = torch.triu(mask, diagonal=0)
    return mask.bool()


def scalar_dot_attn(queries, keys, values, window_size, mask=None, pe=None, attn_dropout=None, BALoss=False):
    """
    :param queries: (B, H, D, l)
    :param keys:    (B, H, D, l_k)
    :param values:  (B, H, D, l_k)
    :param mask:    (B, l, l_k)
    :param pe:      (l, l_k)
    :return:        out: (B, H, D, l)  scores: (B, l, window_size)
    """
    B, H, D_q, L_q = queries.shape
    _, _, D_k, L_k = keys.shape
    _, _, D_v, L_v = values.shape
    assert L_k == L_v
    assert D_q == D_k

    scores = torch.einsum("bhdl,bhdn->bhln", queries, keys) / sqrt(D_q)
    if pe is not None:
        scores = scores + pe
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1), -1e9)  # Fill elements of the tensor with -inf where mask is True
    scores_norm = F.softmax(scores, dim=-1)  # (B, H, l, l_k)
    if attn_dropout is not None:
        scores_norm = attn_dropout(scores_norm)
    V = torch.einsum("bhdn,bhln->bhdl", values, scores_norm)  # (B, H, D, l)
    V = V.reshape(B, -1, L_q)

    each_seg_socre = []
    if BALoss:
        for i in range(B):
            for j in range(H):
                each_seg_socre.append(extract_dis_from_attention(scores_norm[i, j, ...], window_size))
        all_seg_scores = torch.stack(each_seg_socre, dim=0)  # (B*H, L, window_size)
    else:
        all_seg_scores = None

    return V, all_seg_scores


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1,L,D)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(2), :].transpose(1, 2)  # 后面一项: (1,D,l)
        return x


class RelativePositionEmbedding(nn.Module):
    def __init__(self, window_size, h):
        super(RelativePositionEmbedding, self).__init__()
        self.h = h
        self.band_width = window_size
        self.embeddings = nn.Embedding(window_size, h)

    def compute_bias(self, l_q, l_k):
        """
        return [1 h l_q l_k]
        """
        query_position = torch.arange(l_q, dtype=torch.long, device=self.embeddings.weight.device)[:, None]
        key_position = torch.arange(l_k, dtype=torch.long, device=self.embeddings.weight.device)[None, :]

        relative_position = query_position - key_position
        rpe = torch.clamp(relative_position, -(self.band_width // 2), self.band_width // 2) + self.band_width // 2

        bias = self.embeddings(rpe)
        bias = rearrange(bias, 'm n h -> 1 h m n')

        return bias


class LocalAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, d_model, h, l_seg, window_size, attention_dropout, rpe=None, BALoss=False):
        super(LocalAttention, self).__init__()
        """
        d_q, d_k, d_v : dimensions of input q,k,v
        d_model: dimension of hidden state in attention calculation
        l_seg, window_size: hyper parameters for attention calculation
        """
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_model = d_model
        self.l_seg = l_seg
        self.band_width = window_size
        self.rpe = rpe
        self.baloss = BALoss

        self.W_q = nn.Conv1d(d_q, d_model, 1)
        self.W_k = nn.Conv1d(d_k, d_model, 1)
        self.W_v = nn.Conv1d(d_v, d_model, 1)
        self.W_out = nn.Conv1d(d_model, d_v, 1)
        self.dropout = nn.Dropout(attention_dropout)
        self.attn_dropout = None  # nn.Dropout(attention_dropout)  # nn.Dropout(0.2)
        if rpe is not None:
            self.rpe_embedding = rpe

    def forward(self, q, k, v, input_mask):
        """
        q: (B, d_q, L)
        k: (B, d_k, L)
        v: (B, d_v, L)
        input_mask: (B, L), float, 0 or 1
        return: (B, d_v, L)
        """

        B, _, L = q.size()

        q = self.W_q(q)  # (B, d_model, L)
        k = self.W_k(k)  # (B, d_model, L)
        v = self.W_v(v)  # (B, d_model, L)

        q = q.reshape(B, self.h, -1, L)
        k = k.reshape(B, self.h, -1, L)
        v = v.reshape(B, self.h, -1, L)

        # window_mask = create_window_mask(L, self.band_width)
        # begin_index = self.band_width // 2
        # window_mask = window_mask[:, begin_index:-begin_index]  # (L, L)

        input_mask = input_mask[:, 0:1, :].bool()  # (B, 1, L)

        total_mask = ~ input_mask  # (B, L, L)

        if self.rpe:
            rpe_bias = self.rpe_embedding.compute_bias(L, L)
            out, attn = scalar_dot_attn(q, k, v, self.band_width, total_mask, pe=rpe_bias, attn_dropout=self.attn_dropout, BALoss=self.baloss)
        else:
            out, attn = scalar_dot_attn(q, k, v, self.band_width, total_mask, attn_dropout=self.attn_dropout, BALoss=self.baloss)

        if attn is not None:
            attn = attn.reshape(B, self.h, L, self.band_width)  # (B, H, L, window_size)

        out = self.W_out(self.dropout(out))  # * input_mask[:, 0:1, :]

        return out, attn


class EncoderLayer(nn.Module):
    def __init__(self, l_seg, window_size, d_model, h, d_ff, activation, attention_dropout, ffn_dropout, rpe=None, pre_layernorm=False, BALoss=False):
        super(EncoderLayer, self).__init__()
        self.attention = LocalAttention(d_model, d_model, d_model, d_model, h, l_seg, window_size, attention_dropout, rpe=rpe, BALoss=BALoss)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        # self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        # self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm1 = nn.InstanceNorm1d(d_model, track_running_stats=False)
        self.norm2 = nn.InstanceNorm1d(d_model, track_running_stats=False)
        self.dropout = nn.Dropout(ffn_dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.pre_layernorm = pre_layernorm

    def forward(self, x, mask):
        # x: (B, d_model, L)
        # out: (B, d_model, L)
        residual = x
        if self.pre_layernorm:
            # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.norm1(x)

        x, attn = self.attention(x, x, x, mask)
        # x = self.dropout(x)
        x = residual + x
        if not self.pre_layernorm:
            # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.norm1(x)

        residual = x
        if self.pre_layernorm:
            # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.norm2(x)

        x = self.conv2(self.dropout(self.activation(self.conv1(x))))
        # x = self.dropout(x)
        x = residual + x
        if not self.pre_layernorm:
            # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.norm2(x)

        return x, attn


class DecoderLayer(nn.Module):
    def __init__(self, l_seg, window_size, d_model, h, d_ff, activation, attention_dropout, ffn_dropout, rpe=None, pre_layernorm=False, BALoss=False):
        super(DecoderLayer, self).__init__()
        self.attention = LocalAttention(d_model, d_model, d_model, d_model, h, l_seg, window_size, attention_dropout, rpe=rpe, BALoss=BALoss)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        # self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        # self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm1 = nn.InstanceNorm1d(d_model, track_running_stats=False)
        self.norm2 = nn.InstanceNorm1d(d_model, track_running_stats=False)
        self.dropout = nn.Dropout(ffn_dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.pre_layernorm = pre_layernorm

    def forward(self, x, cross, mask):
        # x: (B, d_model, L)
        # out: (B, d_model, L)
        residual = x
        if self.pre_layernorm:
            # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.norm1(x)

        x, attn = self.attention(x, x, cross, mask)
        # x = self.dropout(x)
        x = residual + x
        if not self.pre_layernorm:
            # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.norm1(x)

        residual = x
        if self.pre_layernorm:
            # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.norm2(x)

        x = self.conv2(self.dropout(self.activation(self.conv1(x))))
        # x = self.dropout(x)
        x = residual + x
        if not self.pre_layernorm:
            # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.norm2(x)

        return x, attn


class Prediction_Generation(nn.Module):
    def __init__(self, configs, rpes):
        super(Prediction_Generation, self).__init__()
        # self.last_num = configs.last_num
        # self.pos = PositionalEncoding(configs.d_model_PG)
        self.conv_1x1_in = nn.Conv1d(configs.input_dim, configs.d_model_PG, 1)
        self.encoder_layers = nn.ModuleList([
                EncoderLayer(
                    l_seg=configs.l_seg,
                    window_size=configs.window_size,
                    d_model=configs.d_model_PG,
                    h=configs.n_heads_PG,
                    d_ff=configs.d_ffn_PG,
                    activation=configs.activation,
                    attention_dropout=configs.attention_dropout,
                    ffn_dropout=configs.ffn_dropout,
                    rpe=rpes[i] if rpes is not None else None,  # self.layerwise_rpe,
                    pre_layernorm=configs.pre_norm,
                    BALoss=configs.BALoss,
                ) for i in range(configs.pg_layers)
            ])
        self.decoder_layers = nn.ModuleList([
                DecoderLayer(
                    l_seg=configs.l_seg,
                    window_size=configs.window_size,
                    d_model=configs.d_model_PG,
                    h=configs.n_heads_PG,
                    d_ff=configs.d_ffn_PG,
                    activation=configs.activation,
                    attention_dropout=configs.attention_dropout,
                    ffn_dropout=configs.ffn_dropout,
                    rpe=None,
                    pre_layernorm=configs.pre_norm,
                    BALoss=configs.BALoss,
                ) for i in range(configs.pg_layers)
            ])
        # self.conv_out_layer = nn.Conv1d(3*configs.d_model_PG, configs.num_classes, 1)
        self.dropout2d = nn.Dropout2d(p=configs.input_dropout)
        self.conv_out_layer = nn.Conv1d(configs.d_model_PG, configs.num_classes, 1)
        # self.mid_layer = nn.Conv1d(configs.d_model_PG, configs.d_model_PG, 1)
        # self.conv_out_1 = nn.Conv1d(configs.d_model_PG, configs.d_model_PG, 1)
        # self.conv_out_2 = nn.Conv1d(configs.d_model_PG, configs.d_model_PG, 1)

    def forward(self, x, mask):
        # x [B, D, L]
        _, _, L = x.shape
        x = x.unsqueeze(2)
        x = self.dropout2d(x)
        x = x.squeeze(2)
        x = self.conv_1x1_in(x)
        # x = self.pos(x)
        encoder_layer_num = len(self.encoder_layers)
        decoder_layer_num = len(self.decoder_layers)
        attns = []
        outs = []
        len_list = [L // (2 ** i) for i in range(encoder_layer_num+1)]  # [L, L/2, L//4, ...]
        x_down = [x.clone()]  # down sample
        # mask_down = [mask.clone()]

        for i in range(encoder_layer_num):
            # print(x.shape)
            # x = nn.MaxPool1d(2)(x)
            # x = F.interpolate(x, size=len_list[i+1], mode='nearest')  # (B, D, L) -> (B, D, L//2)
            # mask = F.interpolate(mask, size=len_list[i + 1], mode='nearest')
            x, attn = self.encoder_layers[i](x, mask)
            attns.append(attn)
            x_down.append(x.clone())
            # mask_down.append(mask.clone())

        # x = self.mid_layer(x)

        for i in range(decoder_layer_num):
            # x = F.interpolate(x, size=len_list[decoder_layer_num-i-1], mode='nearest')
            x, attn = self.decoder_layers[i](x, x_down[decoder_layer_num-i-1], mask)
            attns.append(attn)
            # outs.append(F.interpolate(x, size=L, mode='nearest'))
            # if i == decoder_layer_num-3:
            #     outs.append(F.interpolate(self.conv_out_1(x), size=L, mode='nearest'))
            # elif i == decoder_layer_num-2:
            #     outs.append(F.interpolate(self.conv_out_2(x), size=L, mode='nearest'))
            # elif i == decoder_layer_num-1:
            #     outs.append(x)

        # out = self.conv_out_layer(torch.cat(outs[-3:], dim=1)) * mask_down[0][:, 0:1, :]
        # out = self.conv_out_layer(outs[-1]+outs[-2]+outs[-3]) * mask_down[0][:, 0:1, :]
        out = self.conv_out_layer(x) * mask[:, 0:1, :]

        return out, attns


class Refinement(nn.Module):
    def __init__(self, configs, rpes):
        super(Refinement, self).__init__()
        # self.last_num = configs.last_num
        # self.pos = PositionalEncoding(configs.d_model_R)
        self.conv_1x1_in = nn.Conv1d(configs.num_classes, configs.d_model_R, 1)
        self.encoder_layers = nn.ModuleList([
                EncoderLayer(
                    l_seg=configs.l_seg,
                    window_size=configs.window_size,
                    d_model=configs.d_model_R,
                    h=configs.n_heads_R,
                    d_ff=configs.d_ffn_R,
                    activation=configs.activation,
                    attention_dropout=configs.attention_dropout,
                    ffn_dropout=configs.ffn_dropout,
                    rpe=rpes[i] if rpes is not None else None,
                    pre_layernorm=configs.pre_norm,
                    BALoss=configs.BALoss,
                ) for i in range(configs.r_layers)
            ])
        self.decoder_layers = nn.ModuleList([
                DecoderLayer(
                    l_seg=configs.l_seg,
                    window_size=configs.window_size,
                    d_model=configs.d_model_R,
                    h=configs.n_heads_R,
                    d_ff=configs.d_ffn_R,
                    activation=configs.activation,
                    attention_dropout=configs.attention_dropout,
                    ffn_dropout=configs.ffn_dropout,
                    rpe=None,
                    pre_layernorm=configs.pre_norm,
                    BALoss=configs.BALoss,
                ) for i in range(configs.r_layers)
            ])
        # self.conv_out_layer = nn.Conv1d(3*configs.d_model_R, configs.num_classes, 1)
        self.conv_out_layer = nn.Conv1d(configs.d_model_R, configs.num_classes, 1)
        # self.mid_layer = nn.Conv1d(configs.d_model_R, configs.d_model_R, 1)
        # self.conv_out_1 = nn.Conv1d(configs.d_model_R, configs.d_model_R, 1)
        # self.conv_out_2 = nn.Conv1d(configs.d_model_R, configs.d_model_R, 1)

    def forward(self, x, mask):
        # x [B, D, L]
        _, _, L = x.shape
        x = self.conv_1x1_in(x)
        # x = self.pos(x)
        encoder_layer_num = len(self.encoder_layers)
        decoder_layer_num = len(self.decoder_layers)
        attns = []
        outs = []
        len_list = [L // (2 ** i) for i in range(encoder_layer_num+1)]  # [L, L/2, L//4, ...]
        x_down = [x.clone()]  # down sample
        # mask_down = [mask.clone()]

        for i in range(encoder_layer_num):
            # print(x.shape)
            # x = nn.MaxPool1d(2)(x)
            # x = F.interpolate(x, size=len_list[i+1], mode='nearest')  # (B, D, L) -> (B, D, L//2)
            # mask = F.interpolate(mask, size=len_list[i+1], mode='nearest')
            x, attn = self.encoder_layers[i](x, mask)
            attns.append(attn)
            x_down.append(x.clone())
            # mask_down.append(mask.clone())

        # x = self.mid_layer(x)

        conv_out_index = 0
        for i in range(decoder_layer_num):
            # x = F.interpolate(x, size=len_list[decoder_layer_num-i-1], mode='nearest')
            x, attn = self.decoder_layers[i](x, x_down[decoder_layer_num-i-1], mask)
            attns.append(attn)
            # outs.append(F.interpolate(x, size=L, mode='nearest'))
            # if i == decoder_layer_num-3:
            #     outs.append(F.interpolate(self.conv_out_1(x), size=L, mode='nearest'))
            # elif i == decoder_layer_num-2:
            #     outs.append(F.interpolate(self.conv_out_2(x), size=L, mode='nearest'))
            # elif i == decoder_layer_num-1:
            #     outs.append(x)

        # out = self.conv_out_layer(torch.cat(outs[-3:], dim=1)) * mask_down[0][:, 0:1, :]  # torch.mean(torch.stack(outs, dim=0), dim=0)
        # out = self.conv_out_layer(outs[-1]+outs[-2]+outs[-3]) * mask_down[0][:, 0:1, :]
        out = self.conv_out_layer(x) * mask[:, 0:1, :]

        return out, attns


class MultiStageModel(nn.Module):
    def __init__(self, configs):
        super(MultiStageModel, self).__init__()
        # print(configs.rpe_share)
        assert configs.pg_layers == configs.r_layers

        if configs.rpe_use:
            if configs.rpe_share:
                assert configs.n_heads_R == configs.n_heads_PG
                self.rpe_modules = nn.ModuleList([
                    RelativePositionEmbedding(configs.window_size, h=configs.n_heads_PG) for _ in range(configs.pg_layers)
                ])
                self.PG = Prediction_Generation(configs, self.rpe_modules)
                self.Rs = nn.ModuleList(
                    [
                        copy.deepcopy(Refinement(configs, self.rpe_modules))
                        for _ in range(configs.num_R)
                    ]
                )
            else:
                self.PG_rpe_modules = nn.ModuleList([
                    RelativePositionEmbedding(configs.window_size, h=configs.n_heads_PG) for _ in range(configs.pg_layers)
                ])
                self.R_rpe_modules = nn.ModuleList([
                    RelativePositionEmbedding(configs.window_size, h=configs.n_heads_R) for _ in range(configs.r_layers)
                ])
                self.PG = Prediction_Generation(configs, self.PG_rpe_modules)
                self.Rs = nn.ModuleList(
                    [
                        copy.deepcopy(Refinement(configs, self.R_rpe_modules))
                        for _ in range(configs.num_R)
                    ]
                )
        else:
            self.PG = Prediction_Generation(configs, None)
            self.Rs = nn.ModuleList(
                [
                    copy.deepcopy(Refinement(configs, None))
                    for _ in range(configs.num_R)
                ]
            )

        # self.init_emb()

    def forward(self, x, mask):
        all_attns = []
        out, attn_PG = self.PG(x, mask)
        all_attns.append(attn_PG)
        outputs = out.unsqueeze(0)  # (1,B,num_classes,L)
        for R in self.Rs:
            out, attn_R = R(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)  # (B,num_classes,L)
            all_attns.append(attn_R)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)  # (n,B,num_classes,L)
        return outputs, all_attns  # (n,B,num_classes,L)   there are n modules' output need to be computed loss

    def init_emb(self):
        """ Initialize weights of Linear layers """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.model = MultiStageModel(configs=configs)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = configs.num_classes

        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        # logger.add('logs/' + args.dataset + "_" + args.split + "_{time}.log")
        # logger.add(sys.stdout, colorize=True, format="{message}")

    def train(self, batch_gen, device, batch_gen_tst=None):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.configs.lr, betas=(0.9, 0.98), weight_decay=self.configs.weight_decay)   #or 1e-5
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        for epoch in range(self.configs.num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, _, _, _, _ = batch_gen.next_batch(self.configs.bz)
                B, _, L = batch_input.shape
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions, all_attns = self.model(batch_input, mask)

                loss = torch.tensor(0.0).to(device)
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += self.configs.gamma * torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p[:, :, 1:], dim=1),
                                F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask[:, :, 1:])
                    # loss += self.configs.gamma * torch.sum(
                    #     torch.clamp(
                    #         self.mse(
                    #             F.log_softmax(p[:, :, 1:], dim=1),
                    #             F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                    #         min=0, max=16) * mask[:, :, 1:]) / torch.sum(mask[:, :, 1:])

                if self.configs.BALoss:
                    assert B == 1
                    loss_layer_num = 1
                    #    (1,L)      (begin_length)   (end_length)
                    # extract from all layers (different resolution) to get begin_index and end_index
                    _, begin_index, end_index = class2boundary(batch_target)
                    down_target = batch_target
                    begin_index_list = [begin_index]
                    end_index_list = [end_index]
                    len_list = [L // (2 ** i) for i in range(loss_layer_num + 1)]  # [L, L/2, L//4, ...]
                    for i in range(loss_layer_num):
                        down_target = F.interpolate(down_target.float().unsqueeze(0), size=len_list[i+1]).squeeze(0).long()
                        _, begin_index, end_index = class2boundary(down_target)
                        begin_index_list.append(begin_index)
                        end_index_list.append(end_index)

                    for attn in all_attns:  # each attn is each stage list
                        # attn: a list of (1, H, L, window_size)
                        for i in range(loss_layer_num):
                            attn_begin = torch.index_select(attn[i], dim=2, index=begin_index_list[i+1].to(device))  # (1,H,l,window_size), encoder layer attn begin
                            attn_end = torch.index_select(attn[i], dim=2, index=end_index_list[i+1].to(device))  # (1,H,l,window_size), encoder layer attn end
                            loss += self.configs.beta * KL_loss(attn_begin, create_distribution_from_cls(0, self.configs.window_size).to(device))
                            loss += self.configs.beta * KL_loss(attn_end, create_distribution_from_cls(2, self.configs.window_size).to(device))

                            attn_begin = torch.index_select(attn[-i-1], dim=2, index=begin_index_list[i].to(device))  # (1,H,l,window_size), decoder layer attn begin
                            attn_end = torch.index_select(attn[-i-1], dim=2, index=end_index_list[i].to(device))  # (1,H,l,window_size), decoder layer attn begin
                            loss += self.configs.beta * KL_loss(attn_begin, create_distribution_from_cls(0, self.configs.window_size).to(device))
                            loss += self.configs.beta * KL_loss(attn_end, create_distribution_from_cls(2, self.configs.window_size).to(device))

                # predictions = self.model(batch_input, mask[:, 0, :])
                # print(batch_input.shape)  # B, D, L
                # print(batch_target.shape)  # B, L
                # print(mask.shape)  # B, Class_num, L
                # print(mask[:, 0, :].shape)  # B, L
                # print(prediction.shape)  # B, Class_num, L

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions.data[-1], 1)
                # print(predicted.shape)
                # print(batch_target.shape)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            scheduler.step(epoch_loss)
            batch_gen.reset()
            torch.save(self.model.state_dict(), self.configs.model_dir + "/epoch-" + str(epoch + 1) + ".model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                                     float(correct)/total))
            if batch_gen_tst is not None:
                self.test(batch_gen_tst, epoch, device)

    def test(self, batch_gen_tst, epoch, device):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, _, batch_length, batch_chunk, batch_target_source = batch_gen_tst.next_batch(1)
                batch_input, batch_target, mask, batch_target_source = batch_input.to(device), batch_target.to(device), mask.to(device), torch.from_numpy(batch_target_source[0]).to(device)
                predictions, _ = self.model(batch_input, mask)
                _, predicted = torch.max(predictions.data[-1], 1)
                predicted = postprocess(predicted.float(), batch_length, batch_chunk)
                predicted = predicted.int()
                correct += (predicted == batch_target_source).sum().item()
                total += batch_length[0]

        acc = float(correct) / total
        print("---[epoch %d]---: test acc = %f" % (epoch + 1, acc))

        self.model.train()
        batch_gen_tst.reset()

    def predict(self, batch_gen_tst, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(self.configs.model_dir + "/epoch-" + str(self.configs.num_epochs) + ".model"))
            batch_gen_tst.reset()
            while batch_gen_tst.has_next():
                batch_input, _, mask, vids, batch_length, batch_chunk, batch_target_source = batch_gen_tst.next_batch(1)
                batch_input, mask, batch_target_source = batch_input.to(device), mask.to(device), torch.from_numpy(batch_target_source[0]).to(device)
                vid = vids[0]
                length = batch_length[0]
                chunk_num = batch_chunk[0]
                print("predict video id {}, [length: {}, chunk num: {}]".format(vid, length, chunk_num))
                # features = np.load(self.configs.features_path + vid.split('.')[0] + '.npy')
                # features = features[:, ::sample_rate]
                #
                # input_x = torch.tensor(features, dtype=torch.float)
                # input_x.unsqueeze_(0)
                # input_x = input_x.to(device)
                predictions, all_attns = self.model(batch_input, mask)

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    predicted = postprocess(predicted.float(), batch_length, batch_chunk)  # (1, L)
                    predicted = predicted.int()
                    confidence = postprocess(confidence, batch_length, batch_chunk)  # (1, L)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()  # (L)

                    batch_target = batch_target_source.squeeze()  # (L)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()

                    segment_bars_with_confidence(self.configs.results_dir + '/{}_stage{}.png'.format(vid, i),
                                                 confidence.tolist(),
                                                 batch_target.tolist(), predicted.tolist())

                # for i in range(len(attns)):
                #     if i % 2 == 0:
                #         plot_attention_map(attns[i, 0, ...].cpu().numpy(), self.configs.attn_dir + '/{}_stage{}_encoder.png'.format(vid, i//2))
                #     else:
                #         plot_attention_map(attns[i, 0, ...].cpu().numpy(), self.configs.attn_dir + '/{}_stage{}_decoder.png'.format(vid, i//2))

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))

                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(self.configs.results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()


if __name__ == '__main__':
    pass