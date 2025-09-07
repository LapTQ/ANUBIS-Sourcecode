import torch
from torch import nn
from torch.nn import functional as F
from .utils_stgcn.graph import Graph


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class ST_GCN(nn.Module):
    def __init__(self, in_channels, num_point, num_person, num_frames,
                 num_class, graph_args, drop_prob, gcn_kernel_size, **kwargs):
        super().__init__()
        
        C, T, V, M = (in_channels, num_frames, num_point, num_person)
        data_shape = (in_channels, num_frames, num_point, num_person)

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        self.register_buffer('A', A)
        
        # data normalization
        self.data_bn = nn.BatchNorm1d(C * V * M)

        # st-gcn networks
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_layer(C, 64, gcn_kernel_size, 1, A, drop_prob, residual=False),
            st_gcn_layer(64, 64, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(64, 64, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(64, 64, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(64, 128, gcn_kernel_size, 2, A, drop_prob),
            st_gcn_layer(128, 128, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(128, 128, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(128, 256, gcn_kernel_size, 2, A, drop_prob),
            st_gcn_layer(256, 256, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(256, 256, gcn_kernel_size, 1, A, drop_prob),
        ))

        # edge importance weights
        self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(A.shape)) for _ in range(len(self.st_gcn_networks))])
        # fcn
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        
        # 添加身体部位划分方法
        self.num_point = num_point

    def partDivison(self, graph):
        """将骨骼点分成不同的身体部位进行分析"""
        _, k, u, v = graph.size()  # n k u v
        
        # 根据骨骼点索引划分身体部位
        # 这里的索引需要根据您的骨骼点定义进行调整
        head = [26, 27, 28, 29, 30, 31 ]
        left_arm = [4, 5, 6, 7, 8, 9, 10]
        right_arm = [11, 12, 13, 14, 15, 16, 17]
        torso = [0, 1, 2, 3]
        left_leg = [18, 19, 20, 21]
        right_leg = [22, 23, 24, 25]
        
        graph_list = []
        part_list = [head, torso, right_arm, left_arm, right_leg, left_leg]
        
        # 确保索引不超出范围
        for part in part_list:
            valid_indices = [i for i in part if i < self.num_point]
            if not valid_indices:
                continue
                
            # 计算该部位的平均图
            if len(valid_indices) > 0:
                part_graph = graph[:,:,valid_indices,:].mean(dim=2, keepdim=True)
                graph_list.append(part_graph)
                
        # 合并不同部位的图
        if graph_list:
            graph = torch.cat(graph_list, 2)
            
            # 处理目标节点
            graph_list = []
            for part in part_list:
                valid_indices = [i for i in part if i < self.num_point]
                if len(valid_indices) > 0:
                    part_graph = graph[:,:,:,valid_indices].mean(dim=-1, keepdim=True)
                    graph_list.append(part_graph)
                    
            if graph_list:
                return torch.cat(graph_list, -1)
        
        # 如果无法处理，返回原图
        return graph

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # 保存最后一层的图信息
        graph = None
        
        # forward
        for i in range(len(self.st_gcn_networks)):
            gcn = self.st_gcn_networks[i]
            importance = self.edge_importance[i]
            x, layer_graph = gcn(x, self.A * importance)
            
            # 只保存最后一层的图信息
            if i == len(self.st_gcn_networks) - 1:
                graph = layer_graph

        # extract feature
        _, c, t, v = x.shape
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # global pooling
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(N, -1)
        
        # 处理图信息
        if graph is not None:
            graph = graph.view(N, M, -1, v, v).mean(1).view(N, -1)
            # 可选：进行身体部位划分
            # graph = self.partDivison(graph)

        return x, graph


class st_gcn_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, A, drop_prob=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        # spatial network
        self.gcn = SpatialGraphConv(in_channels, out_channels, kernel_size[1]+1)

        # temporal network
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0],1), (stride,1), padding),
            nn.BatchNorm2d(out_channels),
        )

        # residual
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(out_channels))

        # output
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # residual
        res = self.residual(x)

        # spatial gcn
        x, graph = self.gcn(x, A)

        # temporal 1d-cnn
        x = self.tcn(x)

        # output
        x = self.relu(x + res)
        return x, graph


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = s_kernel_size

        # weights of different spatial classes
        self.conv = nn.Conv2d(in_channels, out_channels * s_kernel_size, kernel_size=1)

    def forward(self, x, A):
        x = self.conv(x)  # shape: (n, kc, t, v)
        n, kc, t, v = x.shape
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)  # (n, k, c/k, t, v)

        # 计算注意力图（graph）
        x_mean = x.mean(dim=3)  # (n, k, c/k, v)
        x_mean = x_mean.unsqueeze(-1)  # 添加维度 -> (n, k, c/k, v, 1)
        attention = torch.matmul( 
            x_mean,  # (n, k, c/k, v, 1)
            x_mean.transpose(3, 4)  # (n, k, c/k, 1, v)
        )  # 结果形状: (n, k, c/k, v, v)
        graph = attention.mean(dim=1).mean(dim=1)  # (n, v, v)

        # 空间图卷积
        output = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size]))
        return output, graph


if __name__ == '__main__':
    import main
    main.main()