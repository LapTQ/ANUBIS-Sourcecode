import sys
sys.path.extend(['../'])
import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', 
                 random_choose=False, random_shift=False, random_move=False, 
                 random_rot=False, window_size=-1, normalization=False, 
                 debug=False, use_mmap=True, bone=False, vel=False):
        """
        适配Azure Kinect 32节点的数据加载器
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # 加载标签
        try:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        
        # 将标签转换为numpy数组
        self.label = np.array(self.label)
        
        # 加载数据
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        print(f"数据加载成功，形状: {self.data.shape}")
        print(f"标签数量: {len(self.label)}")
        
        # 如果是调试模式，只使用前100个样本
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100] if isinstance(self.sample_name, list) else list(range(100))

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        
        # 获取有效帧数
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        # 调整数据大小
        if self.window_size > 0:
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        # 数据增强
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
            
        if self.bone:
            # Azure Kinect的骨骼连接
            # 直接在这里定义骨骼连接对，避免导入问题
            anubis_pairs = (
                (1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 7),
                (11, 2), (12, 11), (13, 12), (14, 13), (15, 14), (16, 15), (17, 14), (18, 0),
                (19, 18), (20, 19), (21, 20), (22, 0), (23, 22), (24, 23), (25, 24), (26, 3),
                (27, 26), (28, 27), (29, 28), (30, 27), (31, 30)
            )
            
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in anubis_pairs:
                bone_data_numpy[:, :, v1] = data_numpy[:, :, v1] - data_numpy[:, :, v2]
            
            # 保持骨盆中心（节点0）的轨迹
            bone_data_numpy[:, :, 0] = data_numpy[:, :, 0]
            data_numpy = bone_data_numpy
        else:
            # 对于关节模态，使用骨盆中心（节点0）作为参考
            center_joint = 0  # Azure Kinect的骨盆中心
            trajectory = data_numpy[:, :, center_joint]
            data_numpy = data_numpy - data_numpy[:, :, center_joint:center_joint+1]
            data_numpy[:, :, center_joint] = trajectory

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
            
        # 归一化
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
            
        # 移除NaN和inf
        data_numpy = np.nan_to_num(data_numpy)
        data_numpy[data_numpy == -np.inf] = 0
        data_numpy[data_numpy == np.inf] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        if score is None or len(score) == 0:
            return 0.0
            
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label) if i < rank.shape[0]]
        return sum(hit_top_k) * 1.0 / len(hit_top_k) if len(hit_top_k) > 0 else 0.0