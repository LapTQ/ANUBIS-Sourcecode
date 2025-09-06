import sys
sys.path.extend(['../'])
import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from numpy import inf

import scipy.fftpack

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 tgt_labels=None, frame_len=300, **kwargs):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.tgt_labels = tgt_labels
        self.kwargs = kwargs

        # other parameters
        self.frame_len = frame_len

        self.load_data()

        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C T V M:
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
    
        # 将 self.label 转换为 NumPy 数组
        self.label = np.array(self.label)
        print(f"label.len: {len(self.label)}")
        print(f"label.shape: {self.label.shape}")

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        print(f"data.shape: {self.data.shape}")
        
        if len(self.data) > len(self.label):
            self.data = self.data[:len(self.label)]
            print(f"data.shape: {self.data.shape}")

        # Use tgt labels
        if self.tgt_labels is not None:
            self.label = np.array(self.label)
            tmp_data = None
            tmp_label = None
            for a_tgt_label in self.tgt_labels:
                selected_idxes = np.array(self.label) == a_tgt_label
                if tmp_data is None:
                    tmp_data = self.data[selected_idxes]
                    tmp_label = self.label[selected_idxes]
                else:
                    tmp_data = np.concatenate((tmp_data, self.data[selected_idxes]), axis=0)
                    tmp_label = np.concatenate((tmp_label, self.label[selected_idxes]), axis=0)
            self.data = tmp_data
            self.label = tmp_label

        if 'process_type' in self.kwargs:
            self.process_data(process_type=self.kwargs['process_type'])

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        # Discrete cosine transform
        if 'dct' in self.kwargs:
            self.dct_data(self.kwargs['dct'])
            print('Discrete cosine transform completed. DCT type: ', self.kwargs['dct'])

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
        return min(32731, len(self.label))

    def __iter__(self):
        return self

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

        # Remove NAN
        data_numpy = np.nan_to_num(data_numpy)
        data_numpy[data_numpy == -inf] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


