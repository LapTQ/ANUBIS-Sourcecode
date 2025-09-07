import math
import sys

from torch.nn import TransformerEncoderLayer, TransformerEncoder

from graph.ang_adjs import get_ang_adjs
from model.hyper_gcn import Hyper_GraphConv

sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph.tools import k_adjacency, normalize_adjacency_matrix
from model.mlp import MLP
from torch.autograd import Variable
from model.activation import activation_factory


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class MultiScale_GraphConv(nn.Module):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A_binary,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0,
                 activation='relu',
                 to_use_hyper_conv=False,
                 **kwargs):
        super().__init__()
        self.num_scales = num_scales

        if disentangled_agg:
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A_powers = torch.Tensor(A_powers)

        half_out_channels = out_channels

        # A_norm = kwargs['A_norm']  # 3,25,25
        # self.a_ctrgc = unit_gcn(in_channels, half_out_channels, A_norm, adaptive=True)

        self.channel_fuser = MLP(out_channels, [out_channels])

        if 'hyper_conv' in kwargs and kwargs['hyper_conv'] == 'ntu':
            hyper_adjs = get_ang_adjs('ntu')
            self.A_powers = torch.cat((self.A_powers, hyper_adjs), dim=0)
            if kwargs['hyper_conv'] == 'ntu':
                self.num_scales += 6
            elif kwargs['hyper_conv'] == 'kinetics':
                self.num_scales += 4

        # self.A_powers_param = torch.nn.Parameter(self.A_powers)

        self.use_mask = use_mask
        if use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A_powers.shape)), -1e-6, 1e-6)

        # 这个MLP根本就不是MLP, 这是卷积, 只是类似于MLP的功能.
        self.mlp = MLP(in_channels * self.num_scales, [half_out_channels], dropout=dropout, activation=activation)

        # Spatial Transformer Attention
        if 'to_use_spatial_transformer' in kwargs and kwargs['to_use_spatial_transformer']:
            self.to_use_spatial_trans = True
            self.trans_conv = nn.Conv2d(out_channels, 1, (1, 1), (1, 1))
            self.temporal_len = kwargs['temporal_len']
            nhead = 5
            nlayers = 2
            trans_dropout = 0.5
            encoder_layers = nn.TransformerEncoderLayer(self.temporal_len,
                                                        nhead=nhead, dropout=trans_dropout)
            self.trans_enc = nn.TransformerEncoder(encoder_layers, nlayers)

            # spatial point normalization
            self.point_norm_layer = nn.Sigmoid()

        else:
            self.to_use_spatial_trans = False

        if 'to_use_sptl_trans_feature' in kwargs and kwargs['to_use_sptl_trans_feature']:
            self.to_use_sptl_trans_feature = True
            self.fea_dim = kwargs['fea_dim']
            encoder_layers = nn.TransformerEncoderLayer(self.fea_dim,
                                                        nhead=kwargs['sptl_trans_feature_n_head'],
                                                        dropout=0.5)
            self.trans_enc_fea = nn.TransformerEncoder(encoder_layers,
                                                       kwargs['sptl_trans_feature_n_layer'])
        else:
            self.to_use_sptl_trans_feature = False

    def forward(self, x):
        N, C, T, V = x.shape
        # ctr_gc_out = self.a_ctrgc(x)
        self.A_powers = self.A_powers.to(x.device)

        A = self.A_powers.to(x.dtype)
        if self.use_mask:
            A = A + self.A_res.to(x.dtype)

        support = torch.einsum('vu,nctu->nctv', A, x)

        support = support.view(N, C, T, self.num_scales, V)
        support = support.permute(0, 3, 1, 2, 4).contiguous().view(N, self.num_scales * C, T, V)

        out = self.mlp(support)

        # 实现kernel中, 只实现了一半.
        # out = torch.einsum('nijtv,njktv->niktv', out.unsqueeze(2), out.unsqueeze(1)).view(
        #     N, self.out_channels * self.out_channels, T, V
        # )

        if self.to_use_spatial_trans:
            out_mean = self.trans_conv(out).squeeze()
            out_mean = out_mean.permute(0, 2, 1)
            out_mean = self.trans_enc(out_mean)
            out_mean = self.point_norm_layer(out_mean)
            out_mean = out_mean.permute(0, 2, 1)
            out_mean = torch.unsqueeze(out_mean, dim=1).repeat(1, out.shape[1], 1, 1)
            out = out_mean * out

        if self.to_use_sptl_trans_feature:
            out = out.permute(2, 3, 0, 1)
            for a_out_idx in range(len(out)):
                a_out = out[a_out_idx]
                a_out = self.trans_enc_fea(a_out)
                out[a_out_idx] = a_out
            out = out.permute(2, 3, 0, 1)

        # concat_out = torch.cat((out, ctr_gc_out), dim=1)
        # concat_out = self.channel_fuser(concat_out)
        return out


if __name__ == "__main__":
    from graph.ntu_rgb_d import AdjMatrixGraph

    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    msgcn = MultiScale_GraphConv(num_scales=15, in_channels=3, out_channels=64, A_binary=A_binary)
    msgcn.forward(torch.randn(16, 3, 30, 25))
