import torch
import numpy as np
import sys
import os

# 添加当前目录和上级目录到路径，确保可以导入所需模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# 调整导入路径，根据实际目录结构可能需要修改
try:
    from thop import profile
except ImportError:
    print("请安装thop库: pip install thop")
    profile = None

# 导入模型，根据您的实际目录结构调整

from st_gcn import ST_GCN, import_class


def count_parameters(model):
    """计算模型的可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型配置
    num_class =  102
    num_point = 32  # azure_kinect中的节点数为32
    num_person = 2  # 每个样本的人数，通常为2
    
    # 重要：使用完整的路径来引用图结构
    # 根据您的实际目录结构调整
    # 假设azure_kinect.py在graph目录下
    graph = 'graph.azure_kinect.Graph'
    
    # 在导入前将Graph类复制到本地（可选方法）
    # 以下代码仅在graph模块导入失败时使用
    try:
        # 尝试通过import_class导入
        Graph = import_class(graph)
        print(f"成功导入Graph: {graph}")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"无法导入{graph}，错误: {e}")
        print("使用内嵌Graph类定义...")
        

    
    # 设置图参数
    graph_args = {'labeling_mode': 'spatial', 'CoM': 1}
    
    try:
        # 创建模型
        print("正在创建模型...")
        model = Model(
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            graph=graph,
            graph_args=graph_args,
            in_channels=3,
            drop_out=0.2, 
            adaptive=True
        ).to(device)
        
        # 计算参数数量
        total_params = count_parameters(model)
        print(f"模型参数总数: {total_params:,}")
        print(f"模型参数总数 (百万): {total_params/1e6:.2f}M")
        
        # 计算FLOPs
        if profile:
            # 创建一个示例输入
            dummy_input = torch.randn(1, 3, 300, num_point, num_person).to(device)
            
            # 使用thop计算FLOPs
            try:
                macs, params = profile(model, inputs=(dummy_input,))
                print(f"MACs: {macs:,}")
                print(f"GFLOPs: {macs/1e9:.2f}G")  # MAC约等于2*FLOP
                print(f"FLOPs: {macs*2/1e9:.2f}G")  # 部分研究者使用2*MAC作为FLOP
            except Exception as e:
                print(f"使用thop计算FLOPs时出错: {e}")
        else:
            print("未安装thop库，跳过GFLOPs计算")
    
    except Exception as e:
        print(f"创建或使用模型时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()