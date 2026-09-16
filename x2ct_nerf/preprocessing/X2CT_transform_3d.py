# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------
import numpy as np
import torch


class List_Compose(object):
    """여러 transform을 순서대로 적용. 각 transform은 (fn,) 형태의 1-tuple로 전달."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            fn = t[0]
            img = fn(img)
        return img


class Limit_Min_Max_Threshold(object):
    """value > max -> max, value < min -> min. img: (H, W) or (D, H, W)."""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, img):
        img_copy = img.copy()
        img_copy[img_copy > self.max] = self.max
        img_copy[img_copy < self.min] = self.min
        return img_copy


class Normalization(object):
    """값 범위를 [min, max] -> [0, 1]로 정규화."""

    def __init__(self, min, max, round_v=6):
        self.range = np.array((min, max), dtype=np.float32)
        self.round_v = round_v

    def __call__(self, img):
        img_copy = img.copy()
        img_copy = np.round(
            (img_copy - self.range[0]) / (self.range[1] - self.range[0]), self.round_v
        )
        return img_copy


class ToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img.astype(np.float32))
