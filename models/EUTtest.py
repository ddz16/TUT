import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MyLayers import LearnableAPE, SinusoidalAPE, RPE, LocalAttention


class EncoderLayer(nn.Module):
    def __init__(self, l_seg, window_size, d_model, h, d_ff, activation, attention_dropout, ffn_dropout, rpe=None, pre_layernorm=False, return_attn=False):
        super(EncoderLayer, self).__init__()
        self.l_seg = l_seg
        self.window_size = window_size
        self.attention = LocalAttention(d_model, d_model, d_model, d_model, h, attention_dropout, rpe)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.InstanceNorm1d(d_model, track_running_stats=False)  # self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.InstanceNorm1d(d_model, track_running_stats=False)  # self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(ffn_dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.pre_layernorm = pre_layernorm
        self.return_attn = return_attn

    def forward(self, x, mask):
        # x: (B, d_model, L)
        # out: (B, d_model, L)

        residual = x
        if self.pre_layernorm:
            x = self.norm1(x)  # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, attn = self.attention(x, x, x, mask, self.l_seg, self.window_size, self.return_attn)
        x = residual + x
        if not self.pre_layernorm:
            x = self.norm1(x)  # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
            
        residual = x
        if self.pre_layernorm:
            x = self.norm2(x)  # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv2(self.dropout(self.activation(self.conv1(x))))
        x = residual + x
        if not self.pre_layernorm:
            x = self.norm2(x)  # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x, attn


class DecoderLayer(nn.Module):
    def __init__(self, l_seg, window_size, d_model, h, d_ff, activation, attention_dropout, ffn_dropout, rpe=None, pre_layernorm=False, return_attn=False):
        super(DecoderLayer, self).__init__()
        self.l_seg = l_seg
        self.window_size = window_size
        self.attention = LocalAttention(d_model, d_model, d_model, d_model, h, attention_dropout, rpe)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.InstanceNorm1d(d_model, track_running_stats=False)  # self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.InstanceNorm1d(d_model, track_running_stats=False)  # self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(ffn_dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.pre_layernorm = pre_layernorm
        self.return_attn = return_attn

    def forward(self, x, cross, mask):
        # x: (B, d_model, L)
        # cross: (B, d_model, L)
        # out: (B, d_model, L)

        residual = x
        if self.pre_layernorm:
            x = self.norm1(x)  # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, attn = self.attention(x, x, cross, mask, self.l_seg, self.window_size, self.return_attn)
        x = residual + x
        if not self.pre_layernorm:
            x = self.norm1(x)  # x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        residual = x
        if self.pre_layernorm:
            x = self.norm2(x)  # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv2(self.dropout(self.activation(self.conv1(x))))
        x = residual + x
        if not self.pre_layernorm:
            x = self.norm2(x)  # x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x, attn


class EUTSingleStage(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers, l_seg, window_size, d_model, n_heads, d_ff, activation, attention_dropout, ffn_dropout, input_dropout, rpes, pre_norm, baloss):
        super(EUTSingleStage, self).__init__()

        # self.pos = LearnableAPE(d_model)
        # self.pos = SinusoidalAPE(d_model)
        self.conv_1x1_in = nn.Conv1d(input_dim, d_model, 1)
        self.encoder_layers = nn.ModuleList([
                EncoderLayer(
                    l_seg=l_seg,
                    window_size=window_size,
                    d_model=d_model,
                    h=n_heads,
                    d_ff=d_ff,
                    activation=activation,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    rpe=rpes[i] if rpes is not None else None,
                    pre_layernorm=pre_norm,
                    return_attn=baloss,
                ) for i in range(num_layers)
            ])
        self.decoder_layers = nn.ModuleList([
                DecoderLayer(
                    l_seg=l_seg,
                    window_size=window_size,
                    d_model=d_model,
                    h=n_heads,
                    d_ff=d_ff,
                    activation=activation,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    rpe=None,
                    pre_layernorm=pre_norm,
                    return_attn=baloss,
                ) for _ in range(num_layers)
            ])
        
        self.num_layers = num_layers
        self.input_dropout = input_dropout
        self.dropout2d = nn.Dropout2d(p=input_dropout)
        self.conv_out_layer = nn.Conv1d(d_model, num_classes, 1)

    def forward(self, x, mask):
        # x: [B, D, L]
        # mask: [B, L]
        _, _, L = x.shape

        if self.input_dropout > 0:  # only for Prediction Generation Stage
            x = x.unsqueeze(2)
            x = self.dropout2d(x)
            x = x.squeeze(2)

        x = self.conv_1x1_in(x)
        # x = self.pos(x)

        len_list = [L // (2 ** i) for i in range(self.num_layers+1)]  # [L, L//2, L//4, ...]
        x_down = [x]  # the list of all the encoders' outputs, corresponding to different resolutions
        mask_down = [mask]  # the list of all the masks, corresponding to different resolutions
        attns = []  # the list of all the attention maps


        # encoder
        for i in range(self.num_layers):
            # print(x.shape)
            # down sample
            x = F.interpolate(x, size=len_list[i+1], mode='nearest')  # (B, D, L) -> (B, D, L//2)  ,  x = nn.MaxPool1d(2)(x)
            mask = F.interpolate(mask.unsqueeze(1), size=len_list[i + 1], mode='nearest').squeeze(1)
            x, attn = self.encoder_layers[i](x, mask)
            attns.append(attn)
            x_down.append(x)
            mask_down.append(mask)
        
        # x = self.mid_layer(x)

        # decoder
        for i in range(self.num_layers):
            # print(x.shape)
            # up sample
            x = F.interpolate(x, size=len_list[self.num_layers-i-1], mode='nearest')
            x, attn = self.decoder_layers[i](x, x_down[self.num_layers-i-1], mask_down[self.num_layers-i-1])
            attns.append(attn)

        out = self.conv_out_layer(x) * mask_down[0].unsqueeze(1)

        return out, attns


class EUT(nn.Module):
    def __init__(self, configs):
        super(EUT, self).__init__()
        # print(configs.rpe_share)

        if configs.rpe_use:
            if configs.rpe_share:  # scale-shared rpe
                assert configs.n_heads_PG == configs.n_heads_R
                assert configs.num_layers_PG == configs.num_layers_R
                self.rpe_modules = nn.ModuleList([
                    RPE(configs.window_size, h=configs.n_heads_PG) for _ in range(configs.num_layers_PG)
                ])
            else: # without share
                self.rpe_modules = [nn.ModuleList([RPE(configs.window_size, h=configs.n_heads_PG) for _ in range(configs.num_layers_PG)])] + [
                    nn.ModuleList([RPE(configs.window_size, h=configs.n_heads_R) for _ in range(configs.num_layers_R)]) for _ in range(configs.num_R)
                ]
        else:  # without rpe
            self.rpe_modules = None

        self.PG = EUTSingleStage(
            input_dim=configs.input_dim,
            num_classes=configs.num_classes,
            num_layers=configs.num_layers_PG,
            l_seg=configs.l_seg,
            window_size=configs.window_size,
            d_model=configs.d_model_PG,
            n_heads=configs.n_heads_PG,
            d_ff=configs.d_ffn_PG,
            activation=configs.activation,
            attention_dropout=configs.attention_dropout,
            ffn_dropout=configs.ffn_dropout,
            input_dropout=configs.input_dropout,
            rpes=self.rpe_modules if not isinstance(self.rpe_modules, list) else self.rpe_modules[0],
            pre_norm=configs.pre_norm,
            baloss=configs.baloss
        )

        self.Rs = nn.ModuleList([EUTSingleStage(
            input_dim=configs.num_classes,
            num_classes=configs.num_classes,
            num_layers=configs.num_layers_R,
            l_seg=configs.l_seg,
            window_size=configs.window_size,
            d_model=configs.d_model_R,
            n_heads=configs.n_heads_R,
            d_ff=configs.d_ffn_R,
            activation=configs.activation,
            attention_dropout=configs.attention_dropout,
            ffn_dropout=configs.ffn_dropout,
            input_dropout=0,
            rpes=self.rpe_modules if not isinstance(self.rpe_modules, list) else self.rpe_modules[i+1],
            pre_norm=configs.pre_norm,
            baloss=configs.baloss
        )   for i in range(configs.num_R)
        ])

        # self.init_emb()

    def forward(self, x, mask):
        all_attns = []
        out, attn_PG = self.PG(x, mask)
        all_attns.append(attn_PG)
        outputs = out.unsqueeze(0)  # (1, B, num_classes, L)
        for R in self.Rs:
            out, attn_R = R(F.softmax(out, dim=1) * mask.unsqueeze(1), mask)  # (B, num_classes, L)
            all_attns.append(attn_R)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)  # (n, B, num_classes, L)
        return outputs, all_attns  # (n,B,num_classes,L)   there are n modules' output need to be computed loss

    def init_emb(self):
        """ Initialize weights of Linear layers """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)