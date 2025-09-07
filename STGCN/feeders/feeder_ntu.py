import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from numpy import inf

import scipy.fftpack

from feeders import tools

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, debug=False, 
                 random_choose=False, random_shift=False, random_move=False,
                   window_size=-1, normalization=False, use_mmap=True, **kwargs):
        """

        参数:
        - data_path: .npz 文件路径，包含数据和标签。
        - label_path: 不需要，设置为 None。
        - debug: 是否启用调试模式（只加载少量数据）。
        - random_choose: 是否随机选择部分帧。
        - random_shift: 是否随机平移帧。
        - random_move: 是否随机移动帧。
        - window_size: 窗口大小。
        - normalization: 是否对数据进行归一化。
        - use_mmap: 是否使用内存映射加载数据。
        - kwargs: 其他参数。
        """
        self.debug = debug
        self.data_path = data_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.kwargs = kwargs

        self.load_data()
        if normalization:
            self.get_mean_map()

    def get_a_dataloader(self):
        a_dataset = TensorDataset(torch.tensor(self.data))
        a_dataloader = torch.utils.data.DataLoader(
            dataset=a_dataset,
            batch_size=self.load_bch_sz,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        return a_dataloader

    def load_chunk(self, start_idx, end_idx):
        npz_data = np.load(self.data_path, mmap_mode='r')
        self.data = npz_data['x_train'][start_idx:end_idx]
        self.label = npz_data['y_train'][start_idx:end_idx]

    def load_data(self):
        npz_data = np.load(self.data_path, mmap_mode='r')
        self.data = npz_data['x_train']  # 原始数据形状为 (N, T, V * C * M)
        self.label = npz_data['y_train']

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 3, 25, 2)).transpose(0, 2, 1, 3, 4)  
        
        self.sample_name = [ str(i) for i in range(len(self.data))]
        if 'x_test' in npz_data:
            self.test_data = npz_data['x_test']
            self.test_label = npz_data['y_test']
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, self.test_data.shape[1], 25, 2))

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            if hasattr(self, 'test_data'):
                self.test_data = self.test_data[:100]
                self.test_label = self.test_label[:100]
                
    def dct_data(self, dct_op):
        dct_out = scipy.fftpack.dct(self.data, axis=2)
        if dct_op == 'overwrite':
            self.data = dct_out
        elif dct_op == 'concat':
            self.data = np.concatenate((self.data, dct_out), axis=1)
        elif dct_op == 'lengthen':
            dct_out = dct_out[:, :, :(self.frame_len // 2), :, :]
            self.data = np.concatenate((self.data, dct_out), axis=2)

    def process_data(self, process_type):
        rtn_data = []
        rtn_label = []
        if process_type == 'single_person':
            data_idx = 0
            for a_data in self.data:
                for ppl_id in range(self.data.shape[-1]):
                    a_ppl = a_data[:, :, :, ppl_id]
                    # comment the below one
                    # if np.max(a_ppl) > 0.01:

                    # Keep all data (some mutual data also contain zero)
                    if np.max(a_ppl) > -1:
                        rtn_data.append(np.expand_dims(a_ppl, axis=-1))
                        rtn_label.append(self.label[data_idx])
                data_idx += 1
        else:
            raise NotImplementedError
        rtn_data = np.stack(rtn_data, axis=0)
        rtn_label = np.stack(rtn_label, axis=0)

        # relabel data to consider actor and receiver
        rtn_label = self.relabel_by_energy()

        self.data = rtn_data
        self.label = rtn_label

    def get_energy(self, s):  # ctv
        index = s.sum(-1).sum(0) != 0  # select valid frames
        s = s[:, index, :]
        if len(s) != 0:
            s = s[0, :, :].std() + s[1, :, :].std() + s[2, :, :].std()  # three channels
        else:
            s = 0
        return s

    def relabel_by_energy(self):
        rtn_label = []
        for a_idx, a_data in enumerate(self.data):
            person_1 = a_data[:, :, :, 0]  # C,T,V
            person_2 = a_data[:, :, :, 1]  # C,T,V
            energy_1 = self.get_energy(person_1)
            energy_2 = self.get_energy(person_2)
            if energy_1 > energy_2:  # first kicks the second
                rtn_label.append((1, 0))
            else:
                rtn_label.append((0, 1))
        return np.concatenate(rtn_label, axis=0)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)


        data_numpy = np.nan_to_num(data_numpy)
        data_numpy[data_numpy == -inf] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        if score is None or len(score) == 0:
            return 0.0
        
        rank = score.argsort(axis=1)  # 在类别维度上进行排序
        hit_top_k = []

        # 将 one-hot 标签转换为类别索引
        true_labels = np.argmax(self.label, axis=1)  # 将 one-hot 标签转换为类别索引

        for i, true_label in enumerate(true_labels):
            if i < rank.shape[0]:  # 确保索引在范围内
                # rank 的形状应 >= top_k 才能比较
                if rank.shape[1] >= top_k:
                    hit_top_k.append(true_label in rank[i, -top_k:])  # 检查真实标签是否在 top-k 预测中
                else:
                    hit_top_k.append(False)

        # 防止分母为 0
        return sum(hit_top_k) * 1.0 / max(len(hit_top_k), 1)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=0)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)

if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "E:/skletonbasedactionrecognition/code/BlockGCN-main/data/ntu/NTU60_CS.npz"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path=None, graph=graph, is_3d=True)