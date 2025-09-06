import os
import numpy as np

class CLSExamplar(object):
    def __init__(self, topo_str, base_dir='cls_matrix', num_class=102, num_point=32):
        """
        Anubis数据集的CLSExamplar类
        
        Args:
            topo_str: 拓扑字符串（用于文件名）
            base_dir: 矩阵文件保存目录
            num_class: 类别数量（102）
            num_point: 关节点数量（32）
        """
        matrix_path = os.path.join(os.path.dirname(__file__), base_dir, topo_str + '.npy')
        
        # 检查是否存在预生成的矩阵
        if os.path.exists(matrix_path):
            self.A = np.load(matrix_path)
            print(f"Loaded exemplar matrix from {matrix_path}")
        else:
            # 如果文件不存在，生成新的矩阵
            print(f"Matrix file not found at {matrix_path}, generating new matrix...")
            self.A = self.generate_exemplar_matrix(num_class, num_point)
            
            # 创建目录并保存
            os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
            np.save(matrix_path, self.A)
            print(f"Saved generated matrix to {matrix_path}")
    
    def generate_exemplar_matrix(self, num_class=102, num_point=32):
        """
        生成exemplar矩阵
        """
        # 生成exemplar矩阵
        # 形状: [num_class, num_point, num_point]
        exemplar_matrix = np.zeros((num_class, num_point, num_point), dtype=np.float32)
        
        # Azure Kinect的关节连接关系
        inward_ori_index = [
            (1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7),
            (9, 8), (10, 7), (11, 2), (12, 11), (13, 12), (14, 13), (15, 14),
            (16, 15), (17, 14), (18, 0), (19, 18), (20, 19), (21, 20), (22, 0),
            (23, 22), (24, 23), (25, 24), (26, 3), (27, 26), (28, 27), (29, 28),
            (30, 27), (31, 30)
        ]
        
        # 创建基础邻接矩阵
        base_adj = np.eye(num_point, dtype=np.float32)
        for i, j in inward_ori_index:
            base_adj[i, j] = 1
            base_adj[j, i] = 1  # 无向图
        
        # 为每个类别生成略有不同的exemplar矩阵
        for c in range(num_class):
            # 基础：使用归一化的邻接矩阵
            exemplar_matrix[c] = base_adj.copy()
            
            # 为不同的动作类别添加不同的权重模式
            # 策略1：根据类别索引调整不同身体部位的权重
            if c < 20:  # 假设前20个类别是上肢动作
                # 增强手臂相关节点的连接（节点4-10是手臂）
                for i in range(4, 11):
                    for j in range(4, 11):
                        if i != j and exemplar_matrix[c, i, j] > 0:
                            exemplar_matrix[c, i, j] *= 1.5
                            
            elif c < 40:  # 假设20-40是下肢动作
                # 增强腿部相关节点的连接（节点18-25是腿部）
                for i in range(18, 26):
                    for j in range(18, 26):
                        if i != j and exemplar_matrix[c, i, j] > 0:
                            exemplar_matrix[c, i, j] *= 1.5
                            
            elif c < 60:  # 假设40-60是全身动作
                # 增强躯干节点的连接
                trunk_nodes = [0, 1, 2, 3, 11, 26]
                for i in trunk_nodes:
                    for j in trunk_nodes:
                        if i != j and exemplar_matrix[c, i, j] > 0:
                            exemplar_matrix[c, i, j] *= 1.3
            
            # 策略2：添加一些随机性
            np.random.seed(c)  # 确保每次生成相同的矩阵
            noise = np.random.normal(0, 0.01, (num_point, num_point))
            exemplar_matrix[c] += noise * (exemplar_matrix[c] > 0)
            
            # 确保对角线为1
            np.fill_diagonal(exemplar_matrix[c], 1.0)
            
            # 对称化
            exemplar_matrix[c] = (exemplar_matrix[c] + exemplar_matrix[c].T) / 2
            
            # 归一化
            D = np.sum(exemplar_matrix[c], axis=1)
            D[D == 0] = 1
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
            exemplar_matrix[c] = D_inv_sqrt @ exemplar_matrix[c] @ D_inv_sqrt
        
        return exemplar_matrix


# 如果直接运行这个文件，生成并保存矩阵
if __name__ == '__main__':
    # 创建CLSExamplar实例，这会自动生成并保存矩阵
    exemplar = CLSExamplar('what_will_[J]_act_like_when_[C]-with-punctuation')
    
    # 验证
    print(f"Generated exemplar matrix with shape: {exemplar.A.shape}")
    print(f"Min value: {exemplar.A.min()}, Max value: {exemplar.A.max()}")
    print(f"Mean value: {exemplar.A.mean()}")
    
    # 可视化第一个类别的exemplar矩阵
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(exemplar.A[0], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Exemplar Matrix for Class 0')
    plt.xlabel('Joint Index')
    plt.ylabel('Joint Index')
    plt.show()