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
        适配Azure Kinect 32节点的数据加载器 - STTFormer版本
        与NTU数据集加载器兼容
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
        if self.label_path:
            try:
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f, encoding='latin1')
            
            self.label = np.array(self.label)
        

        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        print(f"data.shape:{self.data.shape}")
        
        # 检查数据形状，确保是 N x C x T x V x M 格式
        if len(self.data.shape) == 3:
            # 如果是 N x T x (V*C) 格式，需要重塑
            N, T, VC = self.data.shape
            V = 32  # Anubis有32个节点
            C = VC // V
            self.data = self.data.reshape((N, T, V, C)).transpose(0, 3, 1, 2)

            self.data = np.expand_dims(self.data, axis=-1)
        elif len(self.data.shape) == 4:
            # 如果是 N x C x T x V 格式，添加人数维度
            self.data = np.expand_dims(self.data, axis=-1)
        

        if self.data.shape[-1] == 1:
            self.data = np.concatenate([self.data, np.zeros_like(self.data)], axis=-1)
        
        print(f"adjust.data.shape: {self.data.shape}")
        print(f"label.len: {len(self.label)}")
        
        # 如果没有sample_name，创建默认的
        if not hasattr(self, 'sample_name') or self.sample_name is None:
            self.sample_name = [f'{self.split}_{i}' for i in range(len(self.data))]
        
        # 如果是调试模式，只使用前100个样本
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

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
            from .bone_pair_anu import anubis_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in anubis_pairs:
                bone_data_numpy[:, :, v1] = data_numpy[:, :, v1] - data_numpy[:, :, v2]
            data_numpy = bone_data_numpy
            
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