import pickle
import random

import numpy as np
from torch.utils.data import Dataset

from sklearn.metrics import recall_score, f1_score


def normalize_by_minmax(seq_kpts):
    kmin = np.min(seq_kpts, axis=2, keepdims=True)
    kmax = np.max(seq_kpts, axis=2, keepdims=True)
    seq_kpts = (seq_kpts - kmin) / (kmax - kmin + 1e-8)

    return seq_kpts


def select_keypoints(seq_kpts, layout):
    if layout == "coco":
        return seq_kpts
    elif layout == "coco_onlyhand":
        return seq_kpts[:, :, 5:11, :]
    elif layout == "coco_headless":
        return seq_kpts[:, :, 5:17, :]
    raise ValueError(f"Unknown layout: {layout}")


def rescale_to_neg1_pos1(seq_kpts):
    return seq_kpts * 2 - 1


class JointToBone:

    def __init__(self, dataset, target="keypoint"):
        self.dataset = dataset
        self.target = target
        if self.dataset not in [
            "nturgb+d",
            "openpose",
            "openpose_new",
            "coco",
            "coco_new",
            "coco_headless",
            "coco_onlyhand",
        ]:
            raise ValueError(f"The dataset type {self.dataset} is not supported")
        if self.dataset == "nturgb+d":
            self.pairs = (
                (0, 1),
                (1, 20),
                (2, 20),
                (3, 2),
                (4, 20),
                (5, 4),
                (6, 5),
                (7, 6),
                (8, 20),
                (9, 8),
                (10, 9),
                (11, 10),
                (12, 0),
                (13, 12),
                (14, 13),
                (15, 14),
                (16, 0),
                (17, 16),
                (18, 17),
                (19, 18),
                (21, 22),
                (20, 20),
                (22, 7),
                (23, 24),
                (24, 11),
            )
        elif self.dataset == "openpose":
            self.pairs = (
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 1),
                (6, 5),
                (7, 6),
                (8, 2),
                (9, 8),
                (10, 9),
                (11, 5),
                (12, 11),
                (13, 12),
                (14, 0),
                (15, 0),
                (16, 14),
                (17, 15),
            )
        elif self.dataset == "openpose_new":
            self.pairs = (
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 1),
                (6, 5),
                (7, 6),
                (8, 18),
                (9, 8),
                (10, 9),
                (11, 18),
                (12, 11),
                (13, 12),
                (14, 0),
                (15, 0),
                (16, 14),
                (17, 15),
                (18, 19),
                (19, 1),
            )
        elif self.dataset == "coco":
            self.pairs = (
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 1),
                (4, 2),
                (5, 0),
                (6, 0),
                (7, 5),
                (8, 6),
                (9, 7),
                (10, 8),
                (11, 0),
                (12, 0),
                (13, 11),
                (14, 12),
                (15, 13),
                (16, 14),
            )
        elif self.dataset == "coco_new":
            self.pairs = (
                (0, 19),
                (1, 0),
                (2, 0),
                (3, 1),
                (4, 2),
                (5, 19),
                (6, 19),
                (7, 5),
                (8, 6),
                (9, 7),
                (10, 8),
                (11, 17),
                (12, 17),
                (13, 11),
                (14, 12),
                (15, 13),
                (16, 14),
                (17, 18),
                (18, 19),
                (19, 19),
            )
        elif self.dataset == "coco_headless":
            self.pairs = (
                (0, 1),
                (1, 0),
                (2, 0),
                (3, 2),
                (4, 2),
                (5, 3),
                (6, 0),
                (7, 1),
                (8, 6),
                (9, 7),
                (10, 8),
                (11, 9),
            )
        elif self.dataset == "coco_onlyhand":
            self.pairs = (
                (0, 1),
                (1, 0),
                (2, 0),
                (3, 2),
                (4, 2),
                (5, 3),
            )

    def __call__(self, results):

        keypoint = results["keypoint"]
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in [
                "openpose",
                "openpose_new",
                "coco",
                "coco_new",
                "handmp",
            ]:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results


class JointToKB:

    def __init__(self, dataset="nturgb+d", target="keypoint"):
        self.dataset = dataset
        self.target = target
        if self.dataset not in [
            "nturgb+d",
            "openpose",
            "openpose_new",
            "coco",
            "coco_new",
            "coco_headless",
            "coco_onlyhand",
        ]:
            raise ValueError(f"The dataset type {self.dataset} is not supported")
        if self.dataset == "nturgb+d":
            self.pairs = (
                (0, 20),
                (1, 1),
                (2, 2),
                (3, 20),
                (4, 4),
                (5, 20),
                (6, 4),
                (7, 5),
                (8, 8),
                (9, 20),
                (10, 8),
                (11, 9),
                (12, 1),
                (13, 0),
                (14, 12),
                (15, 13),
                (16, 1),
                (17, 0),
                (18, 16),
                (19, 17),
                (21, 7),
                (20, 20),
                (22, 6),
                (23, 11),
                (24, 10),
            )
        elif self.dataset == "openpose":
            self.pairs = (
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 1),
                (4, 2),
                (5, 5),
                (6, 1),
                (7, 5),
                (8, 1),
                (9, 2),
                (10, 8),
                (11, 1),
                (12, 5),
                (13, 11),
                (14, 1),
                (15, 1),
                (16, 0),
                (17, 0),
            )
        elif self.dataset == "openpose_new":
            self.pairs = (
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 1),
                (4, 2),
                (5, 5),
                (6, 1),
                (7, 5),
                (8, 19),
                (9, 18),
                (10, 8),
                (11, 19),
                (12, 18),
                (13, 11),
                (14, 1),
                (15, 1),
                (16, 0),
                (17, 0),
                (18, 1),
                (19, 19),
            )
        elif self.dataset == "coco":
            self.pairs = (
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 0),
                (4, 0),
                (5, 5),
                (6, 6),
                (7, 0),
                (8, 0),
                (9, 5),
                (10, 6),
                (11, 11),
                (12, 12),
                (13, 0),
                (14, 0),
                (15, 11),
                (16, 12),
            )
        elif self.dataset == "coco_new":
            self.pairs = (
                (0, 0),
                (1, 19),
                (2, 19),
                (3, 0),
                (4, 0),
                (5, 5),
                (6, 6),
                (7, 19),
                (8, 19),
                (9, 5),
                (10, 6),
                (11, 18),
                (12, 18),
                (13, 17),
                (14, 17),
                (15, 11),
                (16, 12),
                (17, 19),
                (18, 18),
                (19, 19),
            )
        elif self.dataset == "coco_headless":
            self.pairs = (
                (0, 0),
                (1, 1),
                (2, 1),
                (3, 0),
                (4, 0),
                (5, 1),
                (6, 6),
                (7, 7),
                (8, 0),
                (9, 1),
                (10, 6),
                (11, 7),
            )
        elif self.dataset == "coco_onlyhand":
            self.pairs = (
                (0, 0),
                (1, 1),
                (2, 1),
                (3, 0),
                (4, 0),
                (5, 1),
            )

    def __call__(self, results):

        keypoint = results["keypoint"]
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in [
                "openpose",
                "coco",
                "coco_headless",
                "coco_onlyhand",
            ]:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results


class ToMotion:

    def __init__(self, dataset="nturgb+d", source="keypoint", target="motion"):
        self.dataset = dataset
        self.source = source
        self.target = target

    def __call__(self, results):
        data = results[self.source]
        M, T, V, C = data.shape
        motion = np.zeros_like(data)

        assert C in [2, 3]
        motion[:, : T - 1] = np.diff(data, axis=1)
        if C == 3 and self.dataset in ["openpose", "coco"]:
            score = (data[:, : T - 1, :, 2] + data[:, 1:, :, 2]) / 2
            motion[:, : T - 1, :, 2] = score

        results[self.target] = motion

        return results


class MergeSkeFeat:
    def __init__(self, feat_list=["keypoint"], target="keypoint", axis=-1):
        """Merge different feats (ndarray) by concatenate them in the last axis."""

        self.feat_list = feat_list
        self.target = target
        self.axis = axis

    def __call__(self, results):
        feats = []
        for name in self.feat_list:
            feats.append(results.pop(name))
        feats = np.concatenate(feats, axis=self.axis)
        results[self.target] = feats
        return results


class Rename:
    """Rename the key in results.

    Args:
        mapping (dict): The keys in results that need to be renamed. The key of
            the dict is the original name, while the value is the new name. If
            the original name not found in results, do nothing.
            Default: dict().
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, results):
        for key, value in self.mapping.items():
            if key in results:
                assert isinstance(key, str) and isinstance(value, str)
                assert value not in results, "the new name already exists in " "results"
                results[value] = results[key]
                results.pop(key)
        return results


class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        # assert isinstance(transforms, Sequence)
        self.transforms = transforms

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class GenSkeFeat:
    def __init__(self, dataset="nturgb+d", feats=["j"], axis=-1):
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if "b" in feats or "bm" in feats:
            ops.append(JointToBone(dataset=dataset, target="b"))
        if "k" in feats or "km" in feats:
            ops.append(JointToKB(dataset=dataset, target="k"))
        ops.append(Rename({"keypoint": "j"}))
        if "jm" in feats:
            ops.append(ToMotion(dataset=dataset, source="j", target="jm"))
        if "bm" in feats:
            ops.append(ToMotion(dataset=dataset, source="b", target="bm"))
        if "km" in feats:
            ops.append(ToMotion(dataset=dataset, source="k", target="km"))
        ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        self.ops = Compose(ops)

    def __call__(self, results):
        if "keypoint_score" in results and "keypoint" in results:
            assert self.dataset != "nturgb+d"
            assert (
                results["keypoint"].shape[-1] == 2
            ), "Only 2D keypoints have keypoint_score. "
            keypoint = results.pop("keypoint")
            keypoint_score = results.pop("keypoint_score")
            results["keypoint"] = np.concatenate(
                [keypoint, keypoint_score[..., None]], -1
            )
        return self.ops(results)


def harmonic_mean_recall(y_true, y_pred, **kwargs):
    class_weights = kwargs["class_weights"]
    per_class_recall = recall_score(y_true, y_pred, average=None)

    if class_weights is None:
        class_weights = np.ones_like(per_class_recall)
    else:
        class_weights = np.array(class_weights)

    assert len(class_weights) == len(per_class_recall), f"Number of weights ({len(class_weights)}) must match number of classes ({len(per_class_recall)})"

    # (If a class has 0 recall but weight is 0, we can safely ignore it)
    if np.any((per_class_recall == 0) & (class_weights > 0)):
        return 0.0

    # We only compute for classes where weight > 0 to avoid division by zero errors
    # or 0/0 ambiguity for ignored classes.
    active_indices = class_weights > 0

    active_weights = class_weights[active_indices]
    active_recalls = per_class_recall[active_indices]

    if np.any(active_recalls == 0):
        return 0.0

    active_weights = active_weights * active_weights
    harmonic_mean = np.sum(active_weights) / np.sum(active_weights / (active_recalls + 1e-8))

    return harmonic_mean


class Feeder(Dataset):
    """
    Feeder loading skeleton action sequence paths from .txt files per class.

    Data is passed via:
    - classes: list of class names (required, cannot be None).
    - train_data: dict of {class_name: [list of txt paths]} when split='train'.
    - val_data: dict of {class_name: [list of txt paths]} when split='val' or 'test'.

    Each line of a .txt file is the path to a .pkl file containing a (T, V, C)
    float array of keypoints.
    """

    def __init__(
        self,
        classes: list,
        train_data: dict = None,
        val_data: dict = None,
        split="train",
        layout="coco",
        class_map=None,
        ske_feat="j",
        debug=False,  # unused; main.py reads train_feeder_args["debug"]
        eval_metrics=None,
        class_weights=None,
        preload=False,
        seed=42,
        balance_classes=False,
        target_balanced_samples=None,
    ):
        self.classes = list(classes)
        self.train_data = train_data
        self.val_data = val_data
        self.split = split
        self.layout = layout
        self.ske_feat = ske_feat
        self.eval_metrics = eval_metrics
        self.class_weights = class_weights
        self.preload = preload
        self.seed = seed
        self.balance_classes = balance_classes
        self.target_balanced_samples = target_balanced_samples
        self.class_map = np.array(class_map) if class_map is not None else None

        self.load_data()

        self.gen_ske_feat = {
            "j": GenSkeFeat(dataset=self.layout, feats=["j"]),
            "b": GenSkeFeat(dataset=self.layout, feats=["b"]),
            "k": GenSkeFeat(dataset=self.layout, feats=["k"]),
            "jm": GenSkeFeat(dataset=self.layout, feats=["jm"]),
            "bm": GenSkeFeat(dataset=self.layout, feats=["bm"]),
            "km": GenSkeFeat(dataset=self.layout, feats=["km"]),
        }

    def load_data(self):
        if self.split == "train":
            data_dict = self.train_data
        elif self.split in ["val", "test"]:
            data_dict = self.val_data
        else:
            raise ValueError(f"Unknown split: {self.split}")
        assert data_dict is not None, f"No data provided for split '{self.split}'"

        self.file_paths = []
        self.label = []
        class_counts = {}

        for idx, cls_name in enumerate(self.classes):
            txt_files: list = data_dict[cls_name]
            cls_paths = []
            for txt_file in txt_files:
                with open(txt_file, "r", encoding="utf-8") as f:
                    paths = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.strip().startswith("#")
                    ]
                cls_paths.extend(paths)

            if self.balance_classes and self.split == "train" and len(cls_paths) > 0:
                assert self.target_balanced_samples is not None
                if len(cls_paths) > self.target_balanced_samples:
                    rng = random.Random(self.seed + idx)
                    cls_paths = rng.sample(cls_paths, self.target_balanced_samples)
                elif len(cls_paths) < self.target_balanced_samples:
                    repeat = max(1, round(self.target_balanced_samples / len(cls_paths)))
                    cls_paths = cls_paths * repeat

            self.file_paths.extend(cls_paths)
            self.label.extend([idx] * len(cls_paths))
            class_counts[cls_name] = len(cls_paths)

        self.label = np.array(self.label, dtype=np.int64)
        self.sample_name = list(self.file_paths)

        print(
            f"[{self.split.upper()}] Initialized with {len(self.file_paths)} samples across {len(self.classes)} classes."
        )
        for idx, cls_name in enumerate(self.classes):
            print(f"  [{idx:2d}] {cls_name}: {class_counts[cls_name]} samples")

        if self.preload:
            print(
                f"[{self.split.upper()}] Preloading {len(self.file_paths)} samples into memory..."
            )
            self.data = np.array([self.load_file(p) for p in self.file_paths])
        else:
            self.data = None

    def load_file(self, file_path):
        """Loads a single sequence file .pkl into a (T, V, C) float32 numpy array."""
        with open(file_path, "rb") as f:
            arr = pickle.load(f)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return arr

    def get_raw_data(self, index):
        if self.data is not None:
            return self.data[index]
        return self.load_file(self.file_paths[index])

    def __len__(self):
        return len(self.file_paths)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.get_raw_data(index)  # TVC
        label = self.label[index]

        data_numpy = data_numpy.transpose(2, 0, 1)  # CTV
        data_numpy = data_numpy[..., np.newaxis]  # CTVM

        # preprocess
        data_numpy = data_numpy.transpose(
            3, 1, 2, 0
        )  # MTVC => ~BTVC assuming only 1 person
        data_numpy = normalize_by_minmax(data_numpy)
        data_numpy = select_keypoints(data_numpy, self.layout)
        data_numpy = rescale_to_neg1_pos1(data_numpy)
        data_numpy = self.gen_ske_feat[self.ske_feat]({"keypoint": data_numpy})[
            "keypoint"
        ]
        data_numpy = data_numpy.transpose(3, 1, 2, 0)  # CTVM

        return data_numpy, label, index

    def top_k(self, score, top_k):
        assert self.eval_metrics in [
            "acc",
            "harmonic_mean_recall",
            "recall_macro",
            "macro_f1",
        ], f"Unknown eval_metrics: {self.eval_metrics}"
        if self.eval_metrics == "acc":
            return self.top_k_acc(score, top_k)
        elif self.eval_metrics == "harmonic_mean_recall":
            return self.harmonic_mean_recall(score)
        elif self.eval_metrics == "recall_macro":
            return self.recall_macro(score)
        elif self.eval_metrics == "macro_f1":
            return self.macro_f1(score)

    def _get_rank(self, score):
        if self.class_map is None:
            self.class_map = np.arange(score.shape[1])
        else:
            assert len(self.class_map) == score.shape[1]
        return self.class_map[score.argsort()]

    def top_k_acc(self, score, top_k):
        assert self.class_weights is None, "Top-k accuracy is not supported with class weights by LapTQ"
        rank = self._get_rank(score)
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def harmonic_mean_recall(self, score):
        rank = self._get_rank(score)
        return harmonic_mean_recall(self.label, rank[:, -1], class_weights=self.class_weights)

    def recall_macro(self, score):
        assert self.class_weights is None, "Recall macro is not supported with class weights by LapTQ"
        rank = self._get_rank(score)
        return recall_score(self.label, rank[:, -1], average="macro")

    def macro_f1(self, score):
        assert self.class_weights is None, "macro_f1 is not supporting class weights by LapTQ"
        rank = self._get_rank(score)
        return f1_score(
            self.label,
            rank[:, -1],
            labels=self.class_map,
            average="macro",
            zero_division=0,
        )
