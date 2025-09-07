import json
from collections import defaultdict

from sklearn.metrics import confusion_matrix

from metadata.class_labels import ntu120_code_labels, anu_bullying_pair_labels, bly_labels
# from test_fields.kinetics_analysis import get_kinetics_dict
import numpy as np


def get_result_confusion_jsons(gt, pred, data_type, acc_f_name_prefix=None):
    if 'ntu' in data_type:
        code_labels = ntu120_code_labels
    elif 'anubis' in data_type:
        code_labels = bly_labels
        gt = np.array(gt)[:, 0]
    # elif 'kinetics' in data_type:
    #     code_labels = get_kinetics_dict()
    else:
        raise NotImplementedError

    correct_dict = defaultdict(list)
    for idx in range(len(gt)):
        gt_label = np.argmax(gt[idx]) if isinstance(gt[idx], np.ndarray) else gt[idx]
        correct_dict[gt_label].append(int(pred[idx] == gt_label))
    correct_dict_ = correct_dict.copy()

    for a_key in correct_dict:
        correct_dict[a_key] = '{:.6f}'.format(sum(correct_dict[a_key]) / len(correct_dict[a_key]))

    label_acc = {}
    for a_key in correct_dict:
        label_acc[code_labels[int(a_key) + 1]] = float(correct_dict[a_key])

    label_acc = dict(sorted(label_acc.items(), key=lambda item: item[1]))
    label_acc_keys = list(label_acc.keys())

    # 如果 gt 是 one-hot 格式，转换为整数索引
    if isinstance(gt, np.ndarray) and len(gt.shape) > 1:
        gt = np.argmax(gt, axis=1)

# 确认 pred 也是整数标签（防止维度不匹配）
    if isinstance(pred, np.ndarray) and len(pred.shape) > 1:
        pred = np.argmax(pred, axis=1)

    conf_mat = confusion_matrix(gt, pred)


    most_confused = {}
    for i in correct_dict.keys():
        confusion_0 = np.argsort(conf_mat[int(i)])[::-1][0]
        confusion_1 = np.argsort(conf_mat[int(i)])[::-1][1]

        most_confused[code_labels[int(i) + 1]] = [
            "{}  {}".format(code_labels[confusion_0 + 1], conf_mat[int(i)][confusion_0]),
            "{}  {}".format(code_labels[confusion_1 + 1], conf_mat[int(i)][confusion_1]),
            "{}".format(len(correct_dict_[i]))
        ]
    most_confused_ = {}
    for i in label_acc_keys:
        most_confused_[i] = most_confused[i]

    if acc_f_name_prefix is not None:
        with open('{}_confusion_matrix.json'.format(acc_f_name_prefix), 'w') as f:
            json.dump(most_confused_, f, indent=4)
        with open('{}_accuracy_per_class.json'.format(acc_f_name_prefix), 'w') as f:
            json.dump(label_acc, f, indent=4)

    return label_acc, most_confused_

