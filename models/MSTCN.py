'''
    Adapted from https://github.com/sj-li/MS-TCN2
'''

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


class MS_TCN2(nn.Module):
    def __init__(self, configs):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(configs.num_layers_PG, configs.d_model_PG, configs.input_dim, configs.num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(configs.num_layers_R, configs.d_model_R, configs.num_classes, configs.num_classes)) for s in range(configs.num_R)])

    def forward(self, x, mask):
        out = self.PG(x, mask)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1) * mask.unsqueeze(1), mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, d_model, input_dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(input_dim, d_model, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(d_model, d_model, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(d_model, d_model, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*d_model, d_model, 1)
             for i in range(num_layers)
            ))

        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(d_model, num_classes, 1)

    def forward(self, x, mask):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f) * mask.unsqueeze(1)

        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, d_model, input_dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, d_model, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, d_model, d_model)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(d_model, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)

        for layer in self.layers:
            out = layer(out, mask)

        out = self.conv_out(out) * mask.unsqueeze(1)

        return out

 
class MS_TCN(nn.Module):
    def __init__(self, configs):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(configs.num_layers_PG, configs.d_model_PG, configs.input_dim, configs.num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(configs.num_layers_R, configs.d_model_R, configs.num_classes, configs.num_classes)) for s in range(configs.num_R)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask.unsqueeze(1), mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs

class SS_TCN(nn.Module):
    def __init__(self, num_layers, d_model, input_dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, d_model, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, d_model, d_model)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(d_model, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask.unsqueeze(1)
        return out

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask.unsqueeze(1)
