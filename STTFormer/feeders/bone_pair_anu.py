# Azure Kinect 32节点的骨骼连接对
# 基于anubis.py中的inward_ori_index定义

anubis_pairs = [
    (1, 0),    # 骨盆 -> 骨盆中心
    (2, 1),    # 脊柱_肚脐 -> 骨盆
    (3, 2),    # 脊柱_胸部 -> 脊柱_肚脐
    (4, 2),    # 颈部 -> 脊柱_肚脐
    (5, 4),    # 头部 -> 颈部
    (6, 5),    # 头顶 -> 头部
    (7, 6),    # 鼻子 -> 头顶
    (8, 7),    # 左眼 -> 鼻子
    (9, 8),    # 左耳 -> 左眼
    (10, 7),   # 右眼 -> 鼻子
    (11, 2),   # 右肩 -> 脊柱_肚脐
    (12, 11),  # 右肘 -> 右肩
    (13, 12),  # 右腕 -> 右肘
    (14, 13),  # 右手 -> 右腕
    (15, 14),  # 右手指尖 -> 右手
    (16, 15),  # 右拇指 -> 右手指尖
    (17, 14),  # 右拇指尖 -> 右手
    (18, 0),   # 左髋 -> 骨盆中心
    (19, 18),  # 左膝 -> 左髋
    (20, 19),  # 左踝 -> 左膝
    (21, 20),  # 左脚 -> 左踝
    (22, 0),   # 右髋 -> 骨盆中心
    (23, 22),  # 右膝 -> 右髋
    (24, 23),  # 右踝 -> 右膝
    (25, 24),  # 右脚 -> 右踝
    (26, 3),   # 左肩 -> 脊柱_胸部
    (27, 26),  # 左肘 -> 左肩
    (28, 27),  # 左腕 -> 左肘
    (29, 28),  # 左手 -> 左腕
    (30, 27),  # 左手指尖 -> 左肘
    (31, 30),  # 左拇指 -> 左手指尖
]

# 节点名称映射（可选，用于调试）
joint_names = {
    0: 'PELVIS',
    1: 'SPINE_NAVAL', 
    2: 'SPINE_CHEST',
    3: 'NECK',
    4: 'CLAVICLE_LEFT',
    5: 'SHOULDER_LEFT',
    6: 'ELBOW_LEFT',
    7: 'WRIST_LEFT',
    8: 'HAND_LEFT',
    9: 'HANDTIP_LEFT',
    10: 'THUMB_LEFT',
    11: 'CLAVICLE_RIGHT',
    12: 'SHOULDER_RIGHT',
    13: 'ELBOW_RIGHT',
    14: 'WRIST_RIGHT',
    15: 'HAND_RIGHT',
    16: 'HANDTIP_RIGHT',
    17: 'THUMB_RIGHT',
    18: 'HIP_LEFT',
    19: 'KNEE_LEFT',
    20: 'ANKLE_LEFT',
    21: 'FOOT_LEFT',
    22: 'HIP_RIGHT',
    23: 'KNEE_RIGHT',
    24: 'ANKLE_RIGHT',
    25: 'FOOT_RIGHT',
    26: 'HEAD',
    27: 'NOSE',
    28: 'EYE_LEFT',
    29: 'EAR_LEFT',
    30: 'EYE_RIGHT',
    31: 'EAR_RIGHT'
}