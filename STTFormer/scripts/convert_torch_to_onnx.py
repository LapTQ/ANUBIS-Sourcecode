import os
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.append(str(Path(__file__).parent.parent))

from collections import OrderedDict

from model.sttformer import Model

model_paths = OrderedDict(
    [
        (
            "j",
            "outputs/train/fs26/STTFormer/v239--satudora_107cia--2s-15frames--cluster-skeleton-8--j.old/model_0/best.pt",
        ),
        # (
        #     "b",
        #     "outputs/train/fs26/STTFormer/v240--satudora_107cia--2s-15frames--cluster-skeleton-8--b.old/model_0/best.pt",
        # ),
        # (
        #     "jm",
        #     "outputs/train/fs26/STTFormer/v241--satudora_107cia--2s-15frames--cluster-skeleton-8--jm.old/model_0/best.pt",
        # ),
        # (
        #     "bm",
        #     "outputs/train/fs26/STTFormer/v242--satudora_107cia--2s-15frames--cluster-skeleton-8--bm.old/model_0/best.pt",
        # ),
    ]
)

pathf_output = "outputs/convert-torch-to-onnx/fs26/{}.onnx".format(
    "--".join(
        [
            "{}-{}".format(path.split("/")[-3].split("--")[0], key)
            for key, path in model_paths.items()
        ]
    )
)

ls_models = OrderedDict()
for key, model_path in model_paths.items():
    model = Model(
        len_parts=3,
        num_classes=8,
        num_joints=12,
        num_frames=15,
        num_heads=3,
        num_persons=1,
        num_channels=2,
        kernel_size=[3, 5],
        use_pes=True,
        config=[
            [64, 64, 16],
            [64, 64, 16],
            [64, 128, 32],
            [128, 128, 32],
            [128, 256, 64],
            [256, 256, 64],
            [256, 256, 64],
            [256, 256, 64],
        ],
    )

    if os.path.exists(model_path):
        weights = torch.load(model_path)
        weights = OrderedDict([[k.split("module.")[-1], v] for k, v in weights.items()])
        model.load_state_dict(weights)
    else:
        print(
            f"Warning: Model weights not found at {model_path}. Proceeding with random init."
        )
    model.eval()
    ls_models[key] = model


bone_pairs = (
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


class EnsembleModel(nn.Module):
    def __init__(self, ls_models):
        super().__init__()
        self.model_j = ls_models.get("j", None)
        self.model_b = ls_models.get("b", None)
        self.model_jm = ls_models.get("jm", None)
        self.model_bm = ls_models.get("bm", None)

    def forward(self, x):
        B, C, T, V = x.shape
        x_j = x.unsqueeze(-1)  # N C T V M

        x_b = torch.zeros_like(x_j)
        for v1, v2 in bone_pairs:
            x_b[..., v1, 0] = x_j[..., v1, 0] - x_j[..., v2, 0]

        x_jm = torch.zeros_like(x_j)
        x_jm[..., : T - 1, :, :] = x_j[..., 1:, :, :] - x_j[..., : T - 1, :, :]

        x_bm = torch.zeros_like(x_b)
        x_bm[..., : T - 1, :, :] = x_b[..., 1:, :, :] - x_b[..., : T - 1, :, :]

        if self.model_j is not None:
            output_j, feat_j = self.model_j(x_j)
        else:
            output_j, feat_j = None, None

        if self.model_b is not None:
            output_b, feat_b = self.model_b(x_b)
        else:
            output_b, feat_b = None, None

        if self.model_jm is not None:
            output_jm, feat_jm = self.model_jm(x_jm)
        else:
            output_jm, feat_jm = None, None

        if self.model_bm is not None:
            output_bm, feat_bm = self.model_bm(x_bm)
        else:
            output_bm, feat_bm = None, None

        return [
            i
            for p in [
                [
                    output_j, 
                    # feat_j,
                ],
                [
                    output_b, 
                    # feat_b,
                ],
                [
                    output_jm, 
                    # feat_jm,
                ],
                [
                    output_bm, 
                    # feat_bm,
                ],
            ]
            for i in p
            if p[0] is not None
        ]


model = EnsembleModel(ls_models)

dummy_input_shape = (5, 2, 15, 12)
B, C, T, V = dummy_input_shape
dummy_keypoint = torch.randn(dummy_input_shape)

example_inputs = (dummy_keypoint,)

os.makedirs(os.path.dirname(pathf_output), exist_ok=True)

output_names = [
    it
    for p in [
        (
            f"output_{key}", 
            # f"feat_{key}",
        ) for key in ls_models.keys()]
    for it in p
]

torch.onnx.export(
    model,
    example_inputs,
    pathf_output,
    input_names=[
        "input1",
    ],
    output_names=output_names,
    dynamic_axes={
        "input1": {
            0: "batch_size",
        },
        **{
            it: {0: "batch_size"}
            for p in [(f"output_{key}", f"feat_{key}") for key in ls_models.keys()]
            for it in p
        },
    },
    opset_version=12,
)

print("Model exported to {}".format(pathf_output))

import numpy as np
import onnx
import onnxruntime as ort


class ONNXPredictor:
    def __init__(self, **kwargs):
        model_path = kwargs["model_path"]
        enable_CUDAExecutionProvider = kwargs["enable_CUDAExecutionProvider"]
        enable_CPUExecutionProvider = kwargs["enable_CPUExecutionProvider"]

        providers = []
        if enable_CUDAExecutionProvider:
            providers.append("CUDAExecutionProvider")
        if enable_CPUExecutionProvider:
            providers.append("CPUExecutionProvider")

        self.model = onnx.load(model_path)
        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
        )

        onnx.checker.check_model(self.model)

    def predict(self, inputs, output_names):
        inputs = {name: np.array(inputs[name], dtype=np.float32) for name in inputs}
        outputs = self.session.run(output_names, inputs)
        outputs = {name: outputs[i] for i, name in enumerate(output_names)}
        return {"outputs": outputs}


predictor = ONNXPredictor(
    model_path=pathf_output,
    enable_CUDAExecutionProvider=True,
    enable_CPUExecutionProvider=False,
)

print(
    predictor.predict(
        inputs={
            "input1": np.random.randn(13, 2, 15, 12),
        },
        output_names=output_names,
    )
)
