import math
import h5py
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from x2ct_nerf.preprocessing.X2CT_transform_3d import (
    List_Compose, Limit_Min_Max_Threshold, Normalization, ToTensor,
)


class HosiMultiInputWithMask(Dataset):
    SEG_SLICE_ROOT = "/data1/upperairway_hosi/crop_seg_slice"
    XRAY_ROOT = "/data1/upperairway_hosi/drr_output_hu_1"

    def __init__(self, paths, opt: dict):
        self.opt = opt
        self.ct_size = opt["ct_size"]
        self.xray_size = opt["xray_size"]
        self.input_types = opt["input_type"]
        self.num_seg_classes = opt.get("num_seg_classes", 6)

        self.CT_MIN_MAX = opt["CT_MIN_MAX"]
        self.XRAY_MIN_MAX = opt["XRAY_MIN_MAX"]

        self.labels = {"file_path_": paths}
        self._length = len(paths)

        self.mapping_camera_type2pose = {
            "PA": torch.tensor([0, 0]),
            "Lateral": torch.tensor([math.pi / 2, math.pi / 2]),
        }
        self._set_preprocessing()

    def __len__(self):
        return self._length

    def _set_preprocessing(self):
        self.dict_preprocessing = {}
        ct_augment_list = self.opt.get("ct_augment_list", [])
        xray_augment_list = self.opt.get("xray_augment_list", ["normalization"])

        for input_type in self.input_types:
            augment_list = []
            if input_type in ["ct", "ctslice"]:
                if "min_max_th" in ct_augment_list:
                    augment_list.append((Limit_Min_Max_Threshold(*self.CT_MIN_MAX),))
                if "normalization" in ct_augment_list:
                    augment_list.append((Normalization(*self.CT_MIN_MAX),))
            elif input_type in ["PA", "Lateral"]:
                if "normalization" in xray_augment_list:
                    augment_list.append((Normalization(*self.XRAY_MIN_MAX),))
            augment_list.append((ToTensor(),))
            self.dict_preprocessing[input_type] = List_Compose(augment_list)

    def _load_h5(self, path, key):
        with h5py.File(path, "r") as f:
            return np.asarray(f[key])

    def _load_png(self, path):
        return np.asarray(imageio.imread(path))

    def _get_ctslice(self, ct_path):
        img = self._load_h5(ct_path, "ct")
        return np.stack([img, img, img], axis=-1)

    def _get_mask_path(self, ct_path):
        parts = ct_path.split("/")
        patient = parts[-3]
        slice_file = parts[-1]
        return f"{self.SEG_SLICE_ROOT}/{patient}/{slice_file}"

    def _get_mask(self, ct_path):
        mask = self._load_h5(self._get_mask_path(ct_path), "seg")
        return mask.astype(np.int64)

    def _get_xray_path(self, ct_path, input_type):
        patient = ct_path.split("/")[-3]
        suffix = "xray1" if input_type == "PA" else "xray2"
        return f"{self.XRAY_ROOT}/{patient}/{suffix}.png"

    def _apply_xray_transform(self, img, input_type):
        if input_type == "PA":
            img = np.fliplr(img)
        elif input_type == "Lateral":
            img = np.flipud(np.transpose(img, (1, 0)))
        return img

    def __getitem__(self, i):
        example = {}
        ct_path = self.labels["file_path_"][i]

        for idx, input_type in enumerate(self.input_types):
            if idx == 0:  # ctslice
                img = self._get_ctslice(ct_path)
                example[input_type] = self.dict_preprocessing[input_type](img)
            elif input_type in ["PA", "Lateral"]:
                img = self._load_png(self._get_xray_path(ct_path, input_type))
                img = self._apply_xray_transform(img, input_type)
                img = np.stack([img, img, img], axis=-1)
                example[input_type] = self.dict_preprocessing[input_type](img)
                example[f"{input_type}_cam"] = self.mapping_camera_type2pose[input_type]

        mask = self._get_mask(ct_path)
        example["mask"] = torch.from_numpy(mask).long()

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class HosiTrain(Dataset):
    def __init__(self, training_images_list_file, opt):
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = HosiMultiInputWithMask(paths=paths, opt=opt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class HosiTest(Dataset):
    def __init__(self, test_images_list_file, opt):
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = HosiMultiInputWithMask(paths=paths, opt=opt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]