import numpy as np
from graph import tools

# 主图定义 (32节点)
num_node = 32
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8),
    (10, 7), (11, 2), (12, 11), (13, 12), (14, 13), (15, 14), (16, 15),
    (17, 14), (18, 0), (19, 18), (20, 19), (21, 20), (22, 0), (23, 22),
    (24, 23), (25, 24), (26, 3), (27, 26), (28, 27), (29, 28), (30, 27),
    (31, 30)
]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# 第一级子图 (13个关键节点)
indices_1 = [
    0,   # PELVIS
    2,   # SPINE_CHEST
    3,   # NECK
    5,   # SHOULDER_LEFT
    7,   # WRIST_LEFT
    12,  # SHOULDER_RIGHT
    14,  # WRIST_RIGHT
    19,  # KNEE_LEFT
    20,  # ANKLE_LEFT
    23,  # KNEE_RIGHT
    24,  # ANKLE_RIGHT
    26,  # HEAD
    27   # NOSE
]
num_node_1 = len(indices_1)
inward_ori_index_1 = [
    (1, 1),    # SPINE_CHEST自连接
    (0, 1),    # PELVIS -> SPINE_CHEST
    (2, 1),    # NECK -> SPINE_CHEST
    (3, 1),    # SHOULDER_LEFT -> SPINE_CHEST
    (4, 3),    # WRIST_LEFT -> SHOULDER_LEFT
    (5, 1),    # SHOULDER_RIGHT -> SPINE_CHEST
    (6, 5),    # WRIST_RIGHT -> SHOULDER_RIGHT
    (7, 0),    # KNEE_LEFT -> PELVIS
    (8, 7),    # ANKLE_LEFT -> KNEE_LEFT
    (9, 0),    # KNEE_RIGHT -> PELVIS
    (10, 9),   # ANKLE_RIGHT -> KNEE_RIGHT
    (11, 2),   # HEAD -> NECK
    (12, 11)   # NOSE -> HEAD
]
inward_1 = [(i-1, j-1) for (i, j) in inward_ori_index_1]  # 转换为0-based
outward_1 = [(j, i) for (i, j) in inward_1]
self_link_1 = [(i, i) for i in range(num_node_1)]

# 第二级子图 (5个核心节点)
indices_2 = [
    1,  # SPINE_CHEST
    4,  # WRIST_LEFT
    6,  # WRIST_RIGHT
    7,  # KNEE_LEFT
    9   # KNEE_RIGHT
]
num_node_2 = len(indices_2)
inward_ori_index_2 = [
    (0, 0),   # SPINE_CHEST自连接
    (1, 0),   # WRIST_LEFT -> SPINE_CHEST
    (2, 0),   # WRIST_RIGHT -> SPINE_CHEST
    (3, 0),   # KNEE_LEFT -> SPINE_CHEST
    (4, 0),   # KNEE_RIGHT -> SPINE_CHEST
    (1, 2),   # WRIST_LEFT <-> WRIST_RIGHT
    (3, 4)    # KNEE_LEFT <-> KNEE_RIGHT
]
inward_2 = [(i-1, j-1) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]
self_link_2 = [(i, i) for i in range(num_node_2)]

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        
        # 确保以下属性全部定义
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_binary = tools.get_adjacency_matrix(neighbor, num_node)
        self.A_outward_binary = tools.get_adjacency_matrix(outward, num_node)  # 新增
        self.A_inward_binary = tools.get_adjacency_matrix(inward, num_node)    # 新增
        
        # 子图相关属性（如果代码中用到）
        self.A1 = tools.get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1) if 'num_node_1' in locals() else None
        self.A2 = tools.get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2) if 'num_node_2' in locals() else None
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2 * np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            return tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        elif labeling_mode == 'uniform':
            return tools.get_uniform_graph(self.num_node, self.self_link, self.neighbor)
        else:
            raise ValueError(f'Unknown labeling mode: {labeling_mode}')

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        else:
            raise ValueError(f'Unknown labeling mode: {labeling_mode}')
        return A

    def get_subgraph_matrices(self):
        """返回所有子图矩阵的字典"""
        return {
            'A': self.A,          # 主图矩阵
            'A1': self.A1,        # 13节点子图
            'A2': self.A2,        # 5节点子图
            'A_A1': self.A_A1,    # 主图→子图1映射
            'A1_A2': self.A1_A2   # 子图1→子图2映射
        }