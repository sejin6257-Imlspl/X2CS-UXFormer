"""Stripped version of https://github.com/richzhang/PerceptualSimilarity"""
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from taming.util import get_ckpt_path


class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

        ckpt = get_ckpt_path("vgg_lpips", "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        print(f"loaded pretrained LPIPS loss from {ckpt}")
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        in0, in1 = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0), self.net(in1)
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

        val = None
        for kk in range(len(self.chns)):
            f0 = _normalize_tensor(outs0[kk])
            f1 = _normalize_tensor(outs1[kk])
            diff = (f0 - f1) ** 2
            res = lins[kk].model(diff).mean([2, 3], keepdim=True)
            val = res if val is None else val + res
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class vgg16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        features = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential(*[features[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[features[x] for x in range(4, 9)])
        self.slice3 = nn.Sequential(*[features[x] for x in range(9, 16)])
        self.slice4 = nn.Sequential(*[features[x] for x in range(16, 23)])
        self.slice5 = nn.Sequential(*[features[x] for x in range(23, 30)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return outputs(h1, h2, h3, h4, h5)


def _normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
    return x / (norm_factor + eps)