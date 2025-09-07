import numpy as np
from torch.utils.data import Dataset
from feeders import tools
import pickle

class Feeder(Dataset):
    def __init__(self, data_path, label_path, p_interval=1, split='train', 
                 random_choose=False, random_shift=False, random_move=False,
                 random_rot=False, window_size=-1, normalization=False, 
                 debug=False, use_mmap=False, vel=False, sort=False):
        """
        参数说明：
        data_path: .npy数据文件路径
        label_path: .pkl标签文件路径
        split: 数据集划分（train/test）
        p_interval: 采样区间控制
        window_size: 输出序列长度
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.vel = vel
        
        self.load_data()
        
        if sort:
            self.get_n_per_class()
            self.sort()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # 加载数据
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        
        # 加载标签
        if self.label_path is not None:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
            self.label = np.array(self.label)
        
        # 取最小公共长度
        min_length = min(len(self.data), len(self.label))
        self.data = self.data[:min_length]
        self.label = self.label[:min_length]
        
        # 同步过滤无效数据
        valid_mask = ~np.isnan(self.data.mean(axis=(1,2,3,4)))
        self.data = self.data[valid_mask]
        
        # 确保标签过滤使用相同的掩码（前提是数据标签已对齐）
        if len(self.label) == len(valid_mask):  # 只有长度相同时才能过滤
            self.label = self.label[valid_mask]
        else:
            print("警告: 标签数量与数据不匹配，跳过标签过滤")
        print(self.data.shape)
        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]

    def get_n_per_class(self):
        self.n_per_cls = np.bincount(self.label)
        self.csum_n_per_cls = np.insert(np.cumsum(self.n_per_cls), 0, 0)

    def sort(self):
        sorted_idx = self.label.argsort()
        self.data = self.data[sorted_idx]
        self.label = self.label[sorted_idx]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):  
        data_numpy = self.data[index]  # C,T,V,M
        label = self.label[index]

        
        # 有效帧检测
        valid_frame_num = np.sum(data_numpy.sum(axis=(0,2,3)) != 0)
        
        # 数据预处理流程
        data_numpy = tools.valid_crop_resize(
            data_numpy, 
            valid_frame_num, 
            [self.p_interval] if isinstance(self.p_interval, (int, float)) else self.p_interval,
            self.window_size
        )
        
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
            
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)