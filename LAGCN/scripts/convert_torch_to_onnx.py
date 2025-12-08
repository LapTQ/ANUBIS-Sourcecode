import torch
from torch import nn
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from collections import OrderedDict
from model.lagcn.lagcn import Model


model_paths = OrderedDict(
    [
        (
            "j",
            "/home/laptq/laptq-fs26-shoplifting-detection/runs/LAGCN/fs26/v219--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2/model_8/best.pt",
        ),
        (
            "b",
            "/home/laptq/laptq-fs26-shoplifting-detection/runs/LAGCN/fs26/v220--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--b/model_5/best.pt",
        ),
        (
            "jm",
            "/home/laptq/laptq-fs26-shoplifting-detection/runs/LAGCN/fs26/v221--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--jm/model_7/best.pt",
        ),
        (
            "bm",
            "/home/laptq/laptq-fs26-shoplifting-detection/runs/LAGCN/fs26/v222--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--bm/model_2/best.pt",
        ),
    ]
)
pathf_output = "/home/laptq/laptq-fs26-shoplifting-detection/outputs/convert-torch-to-onnx/fs26/LAGCN/{}.onnx".format(
    "--".join(
        [
            "{}-{}".format(path.split('/')[-3].split("--")[0], key)
            for key, path in model_paths.items()
        ]
    )
)

ls_models = OrderedDict()
for key, model_path in model_paths.items():
    model = Model(
        num_class=14,
        num_point=12,
        num_person=1,
        in_channels=2,
        graph="graph.coco_headless.Graph",
        graph_args=dict(labeling_mode="spatial"),
        examplar="graph.cls_examplar_coco_headless_v2.CLSExamplar",
        examplar_args=dict(
            topo_str=None,
            num_class=14,
            num_point=12,
        ),
    )

    weights = torch.load(model_path)
    weights = OrderedDict([[k.split("module.")[-1], v] for k, v in weights.items()])
    model.load_state_dict(weights)
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

        x_j = x

        x_b = torch.zeros_like(x_j)
        for v1, v2 in bone_pairs:
            x_b[..., v1] = x_j[..., v1] - x_j[..., v2]

        x_jm = torch.zeros_like(x_j)
        x_jm[..., : T - 1, :] = x_j[..., 1:, :] - x_j[..., : T - 1, :]

        x_bm = torch.zeros_like(x_b)
        x_bm[..., : T - 1, :] = x_b[..., 1:, :] - x_b[..., : T - 1, :]

        if self.model_j is not None:
            output_j, _, feat_j = self.model_j(x_j)
        else:
            output_j, feat_j = None, None
        if self.model_b is not None:
            output_b, _, feat_b = self.model_b(x_b)
        else:
            output_b, feat_b = None, None
        if self.model_jm is not None:
            output_jm, _, feat_jm = self.model_jm(x_jm)
        else:
            output_jm, feat_jm = None, None
        if self.model_bm is not None:
            output_bm, _, feat_bm = self.model_bm(x_bm)
        else:
            output_bm, feat_bm = None, None
        return [
            i
            for p in [
                [output_j, feat_j],
                [output_b, feat_b],
                [output_jm, feat_jm],
                [output_bm, feat_bm],
            ]
            for i in p
            if p[0] is not None
        ]


model = EnsembleModel(ls_models)

dummy_input_shape = (5, 2, 15, 12)
B, C, T, V = dummy_input_shape
dummy_keypoint = torch.randn(dummy_input_shape)

# Example inputs
example_inputs = (dummy_keypoint,)

os.makedirs(os.path.dirname(pathf_output), exist_ok=True)
# Export to ONNX
torch.onnx.export(
    model,
    example_inputs,
    pathf_output,
    input_names=[
        "input1",
    ],
    output_names=[
        it
        for p in [(f"output_{key}", f"feat_{key}") for key in ls_models.keys()]
        for it in p
    ],
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

# --minShapes=input1:1x2x16x12x1,input2:16 --optShapes=input1:16x2x16x12x1,input2:16 --maxShapes=input1:32x2x16x12x1,input2:16 --shapes=input1:5x2x16x12x1,input2:16

import onnxruntime as ort
import onnx
import numpy as np


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
        output_names=[
            it
            for p in [(f"output_{key}", f"feat_{key}") for key in ls_models.keys()]
            for it in p
        ],
    )
)
