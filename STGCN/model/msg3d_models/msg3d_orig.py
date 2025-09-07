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
                 activation='relu'):
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
                use_Ares=True
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


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3, 5],
                 window_stride=1,
                 window_dilations=[1, 1]):
        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum


class SubVideoTransformer(nn.Module):
    def __init__(self, video_len,
                 subvideo_len,
                 overlapping_len,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3, ablation='original',
                 c3=384,
                 **kwargs):
        super().__init__()
        assert subvideo_len % 4 == 0, \
            'sub video size must be the multiplier of 4'
        self.forward_len = subvideo_len - overlapping_len
        assert (video_len - subvideo_len) % self.forward_len == 0, \
            'each sub video has to be of the same size'

        self.subvideo_len = subvideo_len
        self.subvideo_num = (video_len - subvideo_len) // self.forward_len + 1
        self.a_sub_network = Model(num_class, num_point, num_person, num_gcn_scales, num_g3d_scales,
                                   graph, in_channels, ablation, to_use_final_fc=False, **kwargs)

        # self.subvideo_models = nn.ModuleList()
        # self.subvideo_models.append(a_sub_network)
        # for i in range(self.net_num-1):
        #     self.subvideo_models.append(copy.deepcopy(a_sub_network))

        # 把最后的sequence给处理一下
        self.to_subvideo_combine = kwargs['to_subvideo_combine']
        self.sequence_transformer = None
        if self.to_subvideo_combine == 'transformer':
            nhead = 8
            nlayers = 4
            trans_dropout = 0.5
            encoder_layers = nn.TransformerEncoderLayer(num_class,
                                                        nhead=nhead, dropout=trans_dropout)
            self.sequence_transformer = nn.TransformerEncoder(encoder_layers, nlayers)

        self.fc = nn.Linear(c3, num_class)

        # 合起来每一个subvideo的feature
        self.sequence_pooling = nn.AvgPool2d(kernel_size=(self.subvideo_num, 1))

    def forward(self, input_x):
        N, C, T, V, M = input_x.size()
        video_piece_outs = []
        left_i = 0
        for i in range(self.subvideo_num):
            video_piece_outs.append(input_x[:, :, left_i:(left_i + self.subvideo_len), :, :])
            left_i = left_i + self.forward_len
        video_piece_outs = torch.cat(video_piece_outs, dim=0)
        video_piece_outs = self.a_sub_network(video_piece_outs)

        video_splits = []
        left_i = 0
        for i in range(self.subvideo_num):
            video_splits.append(video_piece_outs[left_i:left_i + N, :])
            left_i = left_i + N
        video_piece_outs = video_splits
        del video_splits

        if self.to_subvideo_combine == 'transformer':
            video_piece_outs = torch.stack(video_piece_outs, dim=0)
            video_piece_outs = self.sequence_transformer(video_piece_outs)
        else:
            video_piece_outs = torch.stack(video_piece_outs, dim=1)
        if self.to_subvideo_combine == 'transformer':
            video_piece_outs = video_piece_outs.permute(1, 0, 2)
        video_piece_outs = self.sequence_pooling(video_piece_outs).squeeze()
        video_piece_outs = self.fc(video_piece_outs)
        return video_piece_outs.squeeze()


class SpatialVelocityTwoStream(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3, ablation='original',
                 c3=384,
                 to_one_hot=False,
                 **kwargs):
        super().__init__()

        self.spatial_net = Model(num_class, num_point, num_person, num_gcn_scales, num_g3d_scales,
                                 graph, in_channels, ablation, to_fc_last=False, **kwargs)
        self.velocity_net = Model(num_class, num_point, num_person, num_gcn_scales, num_g3d_scales,
                                  graph, in_channels, ablation, to_fc_last=False, **kwargs)
        self.spa_vel_conv = torch.nn.Conv2d(2, 1, 1)
        self.fc = nn.Linear(c3, num_class)

        # in channel information
        self.in_channel = in_channels
        self.to_one_hot = to_one_hot
        self.num_point = num_point
        if to_one_hot:
            self.in_channel -= num_point

    def forward(self, input_x):
        spatial_in = input_x[:, :self.in_channel, :, :, :]
        # temporal_in = input_x[:, :self.in_channel, :, :, :]
        temporal_in = input_x[:, self.in_channel:2 * self.in_channel, :, :, :]
        if self.to_one_hot:
            spatial_in = torch.cat((spatial_in, input_x[:, -self.num_point:, :, :, :]), dim=1)
            temporal_in = torch.cat((temporal_in, input_x[:, -self.num_point:, :, :, :]), dim=1)
        spatial_net_out = self.spatial_net(spatial_in).unsqueeze(2)
        velocity_net_out = self.velocity_net(temporal_in).unsqueeze(2)
        N_0, M_0, _, C_0, ST_0 = velocity_net_out.shape
        out = torch.cat((spatial_net_out, velocity_net_out), dim=2).view(N_0 * M_0, 2, C_0, ST_0)
        out = self.spa_vel_conv(out).squeeze().view(N_0, M_0, C_0, ST_0)
        out = out.mean(3)  # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)  # Average pool number of bodies in the sequence
        out = self.fc(out)
        return out


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
                 **kwargs):
        super(Model, self).__init__()

        # ZQ ablation studies
        self.ablation = ablation

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        # c1 = 96
        # c2 = c1 * 2  # 192
        # c3 = c2 * 2  # 384

        c1 = 96
        self.c1 = c1
        # c2 = c1 * 2  # 192  # Original implementation
        c2 = c1 * 4  # 384  # Original implementation
        # c2 = c1
        self.c2 = c2
        # c3 = c2 * 2  # 384  # Original implementation
        c3 = c1 * 8  # 768  # Compatible with pretrained transformer
        # c3 = c1
        self.c3 = c3

        # r=3 STGC blocks

        # 这是新的SGCN1
        if True:
            # MSG3D
            # self.gcn3d1 = MultiWindow_MS_G3D(in_channels, c1, A_binary, num_g3d_scales, window_stride=1)

            self.sgcn1_msgcn = MS_GCN(num_gcn_scales, in_channels, c1, A_binary, disentangled_agg=True,
                                      **kwargs, temporal_len=frame_len, fea_dim=c1, to_use_hyper_conv=True)
            self.sgcn1_ms_tcn_1 = MS_TCN(c1, c1)
            self.sgcn1_ms_tcn_2 = MS_TCN(c1, c1)
            self.sgcn1_ms_tcn_2.act = nn.Identity()

        if 'to_use_temporal_transformer' in kwargs and kwargs['to_use_temporal_transformer']:
            self.tcn1 = MS_TCN(c1, c1, **kwargs,
                               section_size=kwargs['section_sizes'][0], num_point=num_point,
                               fea_dim=c1)
        else:
            self.tcn1 = MS_TCN(c1, c1, **kwargs, fea_dim=c1)

        # 这是新的SGCN2
        if True:
            # MSG3D
            # self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)

            self.sgcn2_msgcn = MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True,
                                      **kwargs, temporal_len=frame_len, fea_dim=c1)
            self.sgcn2_ms_tcn_1 = MS_TCN(c1, c2, stride=2)
            self.sgcn2_ms_tcn_2 = MS_TCN(c2, c2)
            self.sgcn2_ms_tcn_2.act = nn.Identity()

        if 'to_use_temporal_transformer' in kwargs and kwargs['to_use_temporal_transformer']:
            self.tcn2 = MS_TCN(c2, c2, **kwargs,
                               section_size=kwargs['section_sizes'][1], num_point=num_point,
                               fea_dim=c2)
        else:
            self.tcn2 = MS_TCN(c2, c2, **kwargs, fea_dim=c2)

        # 这是新的SGCN3
        if True:
            # MSG3D
            # self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)

            self.sgcn3_msgcn = MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True,
                                      **kwargs, temporal_len=frame_len // 2, fea_dim=c2)
            self.sgcn3_ms_tcn_1 = MS_TCN(c2, c3, stride=2)
            self.sgcn3_ms_tcn_2 = MS_TCN(c3, c3)
            self.sgcn3_ms_tcn_2.act = nn.Identity()

        if 'to_use_temporal_transformer' in kwargs and kwargs['to_use_temporal_transformer']:
            self.tcn3 = MS_TCN(c3, c3, **kwargs,
                               section_size=kwargs['section_sizes'][2], num_point=num_point,
                               fea_dim=c3)
        else:
            self.tcn3 = MS_TCN(c3, c3, **kwargs, fea_dim=c3)

        self.use_temporal_transformer = False

        # 最后一层加一个fc
        self.to_use_final_fc = to_use_final_fc
        if self.to_use_final_fc:
            self.fc = nn.Linear(c3, num_class)

        # Concat multi-skip
        self.fc_multi_skip = nn.Sequential(
            # nn.Linear(c1 + c2 + c3, c3),
            nn.Linear(c1 + c3, c3),
            nn.ReLU(),
            nn.Linear(c3, c3),
            nn.ReLU()
        )

        if self.use_temporal_transformer:
            nhead = num_gcn_scales
            nlayers = 2
            trans_dropout = 0.5
            encoder_layers = nn.TransformerEncoderLayer(self.trans_feature_dim,
                                                        nhead=5, dropout=trans_dropout)
            self.trans_enc = nn.TransformerEncoder(encoder_layers, nlayers)

        self.to_use_sptl_trans_fea = 'to_use_sptl_trans_feature' in kwargs and kwargs['to_use_sptl_trans_feature']

        # For two stream networks
        self.to_fc_last = to_fc_last

        # Angle with the body center
        if 'to_adj_angle_weight' in kwargs and kwargs['to_adj_angle_weight']:
            self.to_use_angle_adj = True
            self.adj_weight_conv_0 = nn.Conv2d(c1, c1, 1)
            self.adj_weight_conv_1 = nn.Conv2d(c2, c2, 1)
            self.adj_weight_beta = 1.0
        else:
            self.to_use_angle_adj = False

        # Attention graph
        if 'att_conv_layer' in kwargs:
            self.to_use_att_conv_layer = True
            self.att_conv_layer = Att_GraphConv(
                in_channels=in_channels, out_channels=c1
            )
            self.att_conv_layer_2 = Att_GraphConv(
                in_channels=in_channels, out_channels=c1
            )
        else:
            self.to_use_att_conv_layer = False

        # Nerf and DCT
        if 'nerf_dct_pip' in kwargs and kwargs['nerf_dct_pip']:
            self.nerf_dct_pip = True
            self.nerf_1x1conv_1 = MLP(in_channels * 5, [in_channels])
            self.dct_1x1conv_1 = MLP(c1 * 4, [c1])

            self.nerf_1x1conv_2 = MLP(c1 * 5, [c1])
            self.dct_1x1conv_2 = MLP(c1 * 4, [c1])

            self.nerf_1x1conv_3 = MLP(c2 * 5, [c2])
            self.dct_1x1conv_3 = MLP(c2 * 4, [c2])
        else:
            self.nerf_dct_pip = False

        # Finetune transformer
        self.to_finetune_transformer = 'is_using_pretrained_transformer' in kwargs and \
                                       kwargs['is_using_pretrained_transformer']
        if self.to_finetune_transformer:  # Finetune transformer
            self.pretrained_transformer = get_pretrained_transformer()

    def enlarge_dim(self, x_in, enlarge_type):
        class Embedder_Nerf:
            def __init__(self):
                multires = 2
                self.kwargs = {
                    'include_input': True,
                    'input_dims': 1,
                    'max_freq_log2': multires - 1,
                    'num_freqs': multires,
                    'log_sampling': True,
                    'periodic_fns': [torch.sin, torch.cos],
                }

                self.create_embedding_fn()

            def get_out_dim(self):
                return int(2. ** self.kwargs['num_freqs']) + 1

            def create_embedding_fn(self):
                embed_fns = []
                d = self.kwargs['input_dims']
                out_dim = 0
                if self.kwargs['include_input']:
                    embed_fns.append(lambda x: x)
                    out_dim += d

                max_freq = self.kwargs['max_freq_log2']
                N_freqs = self.kwargs['num_freqs']

                if self.kwargs['log_sampling']:
                    freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
                else:
                    freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

                for freq in freq_bands:
                    for p_fn in self.kwargs['periodic_fns']:
                        # embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                        embed_fns.append(lambda x, p_fn=p_fn, freq=freq: x * p_fn(freq))
                        out_dim += d

                self.embed_fns = embed_fns
                self.out_dim = out_dim

            def embed(self, inputs, dim):
                return torch.cat([fn(inputs) for fn in self.embed_fns], dim)

        class Embedder_DCT:
            def __init__(self):
                multires = 3
                self.frm_len = 300.0
                self.kwargs = {
                    'include_input': True,
                    'input_dims': 1,
                    'max_freq_log2': multires - 1,
                    'num_freqs': multires,
                    'log_sampling': True,
                    # 'periodic_fns': [torch.sin, torch.cos],
                    'periodic_fns': [torch.cos],
                }

                self.create_embedding_fn()

            def get_out_dim(self):
                return int(2. ** self.kwargs['num_freqs']) + 1

            def create_embedding_fn(self):
                embed_fns = []
                d = self.kwargs['input_dims']
                out_dim = 0
                if self.kwargs['include_input']:
                    embed_fns.append(lambda x, y: x)
                    out_dim += d

                max_freq = self.kwargs['max_freq_log2']
                N_freqs = self.kwargs['num_freqs']

                freq_bands = []
                for k in range(1, N_freqs + 1):
                    freq_bands.append(math.pi / self.frm_len * k)
                freq_bands = torch.tensor(freq_bands)

                for freq in freq_bands:
                    for p_fn in self.kwargs['periodic_fns']:
                        # embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                        embed_fns.append(lambda x, frm_idx, p_fn=p_fn, freq=freq: (x * p_fn(freq * (frm_idx + 1 / 2))))
                        out_dim += d

                self.embed_fns = embed_fns
                self.out_dim = out_dim

            def embed(self, inputs, dim):
                t_len_all = inputs.shape[2]
                time_list = []
                for t_idx in range(t_len_all):
                    a_series = inputs[:, :, t_idx, :].unsqueeze(2)
                    new_time_list = torch.cat([fn(a_series, t_idx) for fn in self.embed_fns], dim)
                    # print('new_time_list: ', new_time_list.squeeze())
                    time_list.append(new_time_list)
                rtn = torch.cat(time_list, 2)
                return rtn

        if enlarge_type == 'nerf':
            the_class = Embedder_Nerf()
            return the_class.embed(x_in, dim=1)
        elif enlarge_type == 'dct':
            the_class = Embedder_DCT()
            return the_class.embed(x_in, dim=1)
        else:
            raise NotImplementedError

    def forward(self, x, set_to_fc_last=True):
        # Select channels
        # x = x[:, :3, :, :]
        N, C, T, V, M = x.size()

        # Not use a person's features
        # x[:, :, :, :, 0] = 0

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        if self.to_use_angle_adj:
            x_avg_t = x[:, :3, :, :, :]
            x_norm = x_avg_t / x_avg_t.norm(p=2, keepdim=True, dim=1)
            x_angle_adj = torch.matmul(x_norm.permute(0, 4, 2, 3, 1), x_norm.permute(0, 4, 2, 1, 3))
            x_angle_adj[x_angle_adj != x_angle_adj] = 0
            x_angle_adj = x_angle_adj
            # x_angle_adj = x_angle_adj - torch.eye(x_angle_adj.shape[-1]).to(x_angle_adj.device)
            x_angle_adj = torch.relu(x_angle_adj)
            # x_angle_adj = torch.abs(x_angle_adj)
            x_angle_adj = torch.softmax(x_angle_adj, dim=-2)
        else:
            x_angle_adj = None

        # print('N: ', N, 'C: ', C, 'T: ', T, 'V: ', V, 'M: ', M)

        # Hyper graph
        if self.to_use_att_conv_layer:
            hyper_out = self.att_conv_layer(x_cp)
            # hyper_out_2 = self.att_conv_layer_2(x_cp)
            # hyper_out_3 = self.hyper_conv_layer_3(x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous())

        # 把两个人那个维度给合并了
        if x_angle_adj is not None:
            x_angle_adj = x_angle_adj.view(N * M, T, V, V)

        # Apply activation to the sum of the pathways
        if self.ablation == 'original':
            ###### First Component ######
            x_g3d = self.gcn3d1(x)
            x = self.sgcn1_msgcn(x)
            x = self.sgcn1_ms_tcn_1(x)
            x = self.sgcn1_ms_tcn_2(x) + x_g3d
            x = F.relu(x, inplace=True)
            x = self.tcn1(x)
            ###### End First Component ######

            ###### Second Component ######
            x_g3d = self.gcn3d2(x)
            x = self.sgcn2_msgcn(x)
            x = self.sgcn2_ms_tcn_1(x)
            x = self.sgcn2_ms_tcn_2(x) + x_g3d
            x = F.relu(x, inplace=True)
            x = self.tcn2(x)
            ###### End Second Component ######

            ###### Third Component ######
            x_g3d = self.gcn3d3(x)
            x = self.sgcn3_msgcn(x)
            x = self.sgcn3_ms_tcn_1(x)
            x = self.sgcn3_ms_tcn_2(x) + x_g3d
            x = F.relu(x, inplace=True)
            x = self.tcn3(x)
            ###### End Third Component ######

        elif self.ablation == 'sgcn_only':
            ###### First Component ######
            x = self.sgcn1_msgcn(x)
            x = self.sgcn1_ms_tcn_1(x)
            x = self.sgcn1_ms_tcn_2(x)
            x = F.relu(x, inplace=True)
            x = self.tcn1(x)
            ###### End First Component ######

            ###### Second Component ######
            x = self.sgcn2_msgcn(x)
            x = self.sgcn2_ms_tcn_1(x)
            x = self.sgcn2_ms_tcn_2(x)
            x = F.relu(x, inplace=True)
            x = self.tcn2(x)
            ###### End Second Component ######

            ###### Third Component ######
            x = self.sgcn3_msgcn(x)
            x = self.sgcn3_ms_tcn_1(x)
            x = self.sgcn3_ms_tcn_2(x)
            x = F.relu(x, inplace=True)
            x = self.tcn3(x)
            ###### End Third Component ######

        elif self.ablation == 'gcn3d_only':
            x = F.relu(self.gcn3d1(x), inplace=True)
            x = self.tcn1(x)

            x = F.relu(self.gcn3d2(x), inplace=True)
            x = self.tcn2(x)

            x = F.relu(self.gcn3d3(x), inplace=True)
            x = self.tcn3(x)

        ### BEGIN: Pretrained transformer ###
        if self.to_finetune_transformer:
            x_n, x_c, x_t, x_v = x.shape
            out = x
            out = out.view(N*M, x_c, x_t, x_v)
            out = out.mean(3)
            out = out.repeat(1, 1, 4).view(N*M, x_c, x_t*4)[:, :, :196].permute(0, 2, 1)
            out = self.pretrained_transformer(out)
            out = out.view(N, M, -1)
            out = out.mean(1)  # Average pool number of bodies in the sequence
        ### END: of pretrained transformer ###

        if not self.to_finetune_transformer:  # normal
            out = x
            t_dim = out.shape[2]
            out_channels = out.size(1)

            out = out.view(N, M, out_channels, -1)
            out = out.mean(3)  # Global Average Pooling (Spatial+Temporal)
            out = out.mean(1)  # Average pool number of bodies in the sequence

            if set_to_fc_last:
                if self.to_use_final_fc:
                    out = self.fc(out)
        return out


if __name__ == "__main__":
    # For debugging purposes
    import sys

    sys.path.append('..')

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 50, 25, 2
    x = torch.randn(N, C, T, V, M)
    model.forward(x)

    print('Model total # params:', count_params(model))

    sub_video_transformer = SubVideoTransformer(video_len=300,
                                                subvideo_len=60,
                                                num_class=60,
                                                num_point=25,
                                                num_person=2,
                                                num_gcn_scales=13,
                                                num_g3d_scales=6,
                                                graph='graph.ntu_rgb_d.AdjMatrixGraph'
                                                )
