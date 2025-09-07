from st_gcn_GCL import ST_GCN
from ptflops import get_model_complexity_info
import torch
import sys

# 配置模型参数
config = {
    'in_channels': 3,          # 输入通道数 (C)
    'num_point': 32,           # 关键点数量 (V)
    'num_person': 2,           # 人数 (M)
    'num_frames': 300,          # 时间步长/窗口大小 (T)
    'num_class': 102,           # 分类类别数
    'graph_args': {            # 图参数 (根据实际graph.py定义)
        'layout': 'anubis',
        'strategy': 'spatial'
    },
    'drop_prob': 0.5,          # dropout概率
    'gcn_kernel_size': (9, 3)  # 时空卷积核大小
}

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = ST_GCN(**config).to(device)

# 输入形状构造器 (N=1, C, T, V, M)
def input_constructor(x_shape):
    return {
        'x': torch.randn(1, *x_shape).to(device),
        # 其他输入参数按需补充
    }

# 输入形状 (C, T, V, M)
input_shape = (config['in_channels'], config['num_frames'], 
               config['num_point'], config['num_person'])

try:
    # 计算MACs和参数量
    macs, params = get_model_complexity_info(
        model,
        input_shape,
        input_constructor=input_constructor,
        as_strings=False,
        print_per_layer_stat=True,
        verbose=True
    )
    
    if macs is not None and params is not None:
        # 计算理论GFLOPs (FLOPs = MACs * 2)
        gflops = macs * 2 / 1e9
        params_million = params / 1e6
        
        print("\n===== 模型计算量报告 =====")
        print(f"输入形状: (1, {', '.join(map(str, input_shape))})")
        print(f"理论GFLOPs: {gflops:.2f}")
        print(f"总参数量: {params_million:.2f}M")
        
        # 关键参数影响说明
        print("\n[注] GFLOPs与以下参数直接相关:")
        print(f"- 时间窗口大小(num_frames): {config['num_frames']} (增大将线性增加计算量)")
        print(f"- 关键点数量(num_point): {config['num_point']} (平方级影响图卷积计算量)")
    else:
        print("计算失败，请检查模型前向传播")

except Exception as e:
    print(f"计算过程中出错: {str(e)}")
    print("可能的原因:")
    print("1. 输入形状与模型不匹配")
    print("2. graph_args参数未正确定义")
    print("3. 自定义层未注册FLOPs计算")