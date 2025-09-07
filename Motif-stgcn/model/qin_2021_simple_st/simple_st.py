import copy
import sys

from model.att_gcn import Att_GraphConv
from model.hyper_gcn import Hyper_GraphConv
from model.transformers import get_pretrained_transformer

sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory


class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 nonlinear='relu'):
        super().__init__()

        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True,
                activation=nonlinear
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class SpatialLayer(nn.Module):
    def __init__(self, in_channels, joint_num):
        super(SpatialLayer, self).__init__()
        self.spat_layer = nn.Linear(joint_num, joint_num)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.spat_layer(x)
        x = self.bn(x)
        return x

class TemporalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, group_num):
        super(TemporalLayer, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            groups=group_num,
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 ablation='original',
                 to_use_final_fc=True,
                 to_fc_last=True,
                 frame_len=300,
                 nonlinear='relu',
                 **kwargs):
        super(Model, self).__init__()

        # Activation function
        self.nonlinear_f = activation_factory(nonlinear)

        # ZQ ablation studies
        self.ablation = ablation

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        c1 = 81
        c2 = c1 * 2
        c3 = c2 * 2

        # ST 1
        self.spat_l1 = SpatialLayer(in_channels, num_point)
        self.temp_l1 = TemporalLayer(in_channels=in_channels, out_channels=c1,
                                     kernel_size=3, stride=1, dilation=1, group_num=1)
        # ST 2
        self.spat_l2 = SpatialLayer(c1, num_point)
        self.temp_l2 = TemporalLayer(in_channels=c1, out_channels=c2,
                                     kernel_size=3, stride=2, dilation=1, group_num=(c1//in_channels))
        # ST 3
        self.spat_l3 = SpatialLayer(c2, num_point)
        self.temp_l3 = TemporalLayer(in_channels=c2, out_channels=c3,
                                     kernel_size=3, stride=2, dilation=1, group_num=(c2//in_channels))
        self.act_fn = activation_factory(nonlinear)

        # 最后一层加一个fc
        self.to_use_final_fc = to_use_final_fc
        if self.to_use_final_fc:
            self.fc = nn.Linear(c3, num_class)

        # Rank pooling loss
        self.is_use_rank_pool = 'is_use_rank_pool' in kwargs and kwargs['is_use_rank_pool']
        # self.is_use_rank_pool = False
        if self.is_use_rank_pool:
            self.rank_pool_layer = nn.Linear(c3, 1)

        # CAM
        self.is_get_cam = 'is_get_cam' in kwargs and kwargs['is_get_cam']

    def forward(self, x, set_to_fc_last=True):
        # Select channels
        # x = x[:, :3, :, :]
        N, C, T, V, M = x.size()

        # Not use a person's features
        # x[:, :, :, :, 0] = 0

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()  # N*M, C, T, V

        x = self.spat_l1(x)
        x = self.temp_l1(x)
        x = self.act_fn(x)

        x = self.spat_l2(x)
        x = self.temp_l2(x)
        x = self.act_fn(x)

        x = self.spat_l3(x)
        x = self.temp_l3(x)
        x = self.act_fn(x)

        out = x
        out_channels = out.size(1)

        t_dim = out.shape[2]
        out = out.view(N, M, out_channels, t_dim, -1)
        out = out.permute(0, 1, 3, 4, 2)  # N, M, T, V, C
        out = out.mean(3)  # Global Average Pooling (Spatial)

        # Rank pooling
        if self.is_use_rank_pool:
            rank_pool_out = self.rank_pool_layer(out).squeeze()
            rank_pool_out = rank_pool_out.mean(1)
            # rank_out_len = rank_pool_out.shape[-1]
            # min_value = torch.min(rank_pool_out, dim=-1)[0].unsqueeze(-1).repeat(1, rank_out_len)
            # max_value = torch.max(rank_pool_out, dim=-1)[0].unsqueeze(-1).repeat(1, rank_out_len)
            # rank_pool_out = (rank_pool_out - min_value) / (max_value - min_value)

        out = out.mean(2)  # Global Average Pooling (Temporal)
        out = out.mean(1)  # Average pool number of bodies in the sequence

        if self.is_get_cam:  # Keep the out features
            out_feature = out

        if set_to_fc_last:
            if self.to_use_final_fc:
                out = self.fc(out)

        other_outs = {}
        if self.is_use_rank_pool:
            other_outs['rank_pool'] = rank_pool_out
        if self.is_get_cam:  # Get CAM.
            out_max = torch.argmax(out, dim=-1)
            related_weight = self.fc.weight[out_max]
            weighted_features = related_weight * out_feature
            other_outs['pred_idx_cam'] = (out_max, weighted_features)

        return out, other_outs


if __name__ == "__main__":
    pass