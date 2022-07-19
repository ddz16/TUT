'''
    Adapted from https://github.com/ChinaYi/ASFormer
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.MyLayers import LocalAttention


def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)


class EncoderLayer(nn.Module):
    def __init__(self, l_seg, dilation, d_model, h, activation, attention_dropout, ffn_dropout, alpha):
        """
        in ASFormer paper: window size = dilation, but it is more reasonable to set the window size to an odd number
                           therfore, in our implement, window size = dilation + 1 !!!
                           h = 1
                           activation = relu
                           attention_dropout = 0
                           ffn_dropout = 0.5
        """
        super(EncoderLayer, self).__init__()
        
        self.attention = LocalAttention(d_model, d_model, d_model, d_model // 2, h, attention_dropout, rpe=None)
        
        self.conv1 = nn.Conv1d(d_model, d_model, 3, padding=dilation, dilation=dilation)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
        
        self.conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(ffn_dropout)

        self.l_seg = l_seg
        self.dilation = dilation
        self.alpha = alpha

    def forward(self, x, input_mask):

        out = self.activation(self.conv1(x))  # ConvFFN

        residual = out
        out = self.norm(out)
        out, _ = self.attention(out, out, out, input_mask, self.l_seg, 2 * (self.dilation // 2) + 1, False)
        out = self.alpha * out + residual

        out = self.dropout(self.conv2(out))

        return (x + out) * input_mask.unsqueeze(1)


class DecoderLayer(nn.Module):
    def __init__(self, l_seg, dilation, d_model, h, activation, attention_dropout, ffn_dropout, alpha):
        """
        in ASFormer paper: window size = dilation, but it is more reasonable to set the window size to an odd number
                           therfore, in our implement, window size = dilation + 1 !!!
                           h = 1
                           activation = relu
                           attention_dropout = 0
                           ffn_dropout = 0.5
        """
        super(DecoderLayer, self).__init__()
        
        self.attention = LocalAttention(d_model, d_model, d_model, d_model // 2, h, attention_dropout, rpe=None)
        
        self.conv1 = nn.Conv1d(d_model, d_model, 3, padding=dilation, dilation=dilation)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
        
        self.conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(ffn_dropout)

        self.l_seg = l_seg
        self.dilation = dilation
        self.alpha = alpha

    def forward(self, x, cross, input_mask):
        """
        in ASFormer code: v comes from the last stage while q and k come from the last layer in the same stage
                          This is different from the ASFormer paper! Please see https://github.com/ChinaYi/ASFormer/issues/1
        """

        out = self.activation(self.conv1(x))  # ConvFFN

        residual = out
        out = self.norm(out)
        out, _ = self.attention(out, out, cross, input_mask, self.l_seg, 2 * (self.dilation // 2) + 1, False)
        out = self.alpha * out + residual

        out = self.dropout(self.conv2(out))

        return (x + out) * input_mask.unsqueeze(1)


class Encoder(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers, l_seg, d_model, n_heads, activation, attention_dropout, ffn_dropout, input_dropout, alpha):
        super(Encoder, self).__init__()

        self.conv_1x1_in = nn.Conv1d(input_dim, d_model, 1)

        self.encoder_layers = nn.ModuleList([
                EncoderLayer(
                    l_seg=l_seg,
                    dilation=2 ** i,
                    d_model=d_model,
                    h=n_heads,
                    activation=activation,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    alpha=alpha,
                ) for i in range(num_layers)
            ])
        
        self.input_dropout = input_dropout
        self.dropout = nn.Dropout2d(p=input_dropout)
        self.conv_out_layer = nn.Conv1d(d_model, num_classes, 1)
        

    def forward(self, x, input_mask):

        if self.input_dropout > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)
        
        feature = self.conv_1x1_in(x)

        for layer in self.encoder_layers:
            feature = layer(feature, input_mask)
        
        out = self.conv_out_layer(feature) * input_mask.unsqueeze(1)

        return out, feature


class Decoder(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers, l_seg, d_model, n_heads, activation, attention_dropout, ffn_dropout, alpha):
        super(Decoder, self).__init__()

        self.conv_1x1_in = nn.Conv1d(input_dim, d_model, 1)

        self.decoder_layers = nn.ModuleList([
                DecoderLayer(
                    l_seg=l_seg,
                    dilation=2 ** i,
                    d_model=d_model,
                    h=n_heads,
                    activation=activation,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    alpha=alpha,
                ) for i in range(num_layers)
            ])
        
        self.conv_out_layer = nn.Conv1d(d_model, num_classes, 1)

    def forward(self, x, cross, input_mask):
        
        feature = self.conv_1x1_in(x)

        for layer in self.decoder_layers:
            feature = layer(feature, cross, input_mask)
        
        out = self.conv_out_layer(feature) * input_mask.unsqueeze(1)

        return out, feature
    

class ASFormer(nn.Module):
    def __init__(self, configs):
        super(ASFormer, self).__init__()
        # assert configs.num_layers_PG == configs.num_layers_R
        assert configs.d_model_PG == configs.d_model_R

        self.PG = Encoder(
            input_dim=configs.input_dim,
            num_classes=configs.num_classes,
            num_layers=configs.num_layers_PG,
            l_seg=configs.l_seg,
            d_model=configs.d_model_PG,
            n_heads=1,
            activation='relu',
            attention_dropout=0,
            ffn_dropout=0.5,
            input_dropout=configs.input_dropout,
            alpha=1,
        )

        self.Rs = nn.ModuleList([Decoder(
            input_dim=configs.num_classes,
            num_classes=configs.num_classes,
            num_layers=configs.num_layers_R,
            l_seg=configs.l_seg,
            d_model=configs.d_model_R,
            n_heads=1,
            activation='relu',
            attention_dropout=0,
            ffn_dropout=0.5,
            alpha=exponential_descrease(i),
        )   for i in range(configs.num_R)
        ])
        
    def forward(self, x, mask):
        out, feature = self.PG(x, mask)
        outputs = out.unsqueeze(0)
        
        for decoder in self.Rs:
            out, feature = decoder(F.softmax(out, dim=1) * mask.unsqueeze(1), feature * mask.unsqueeze(1), mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
 
        return outputs
            

if __name__ == '__main__':
    pass