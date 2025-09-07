import torch
import timm
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_pretrained_transformer(num_class=120):
    vit_base_patch16_224_cls_120 = timm.create_model('vit_base_patch16_224',
                                                     pretrained=False, num_classes=num_class).to('cuda')
    vit_base_patch16_224_cls_120.patch_embed = Identity()
    return vit_base_patch16_224_cls_120


if __name__ == '__main__':
    rand_input = torch.randn((50, 196, 768)).to('cuda')
    a_model = get_pretrained_transformer()
    a_model(rand_input)
