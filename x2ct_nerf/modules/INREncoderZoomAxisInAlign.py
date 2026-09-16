import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from x2ct_nerf.modules.nerf import model_utils


class INREncoderZoomAxisInAlign(nn.Module):
    """
    X-ray feature를 3D CT 좌표계로 back-projection하는 encoder.
    각 axis(axial/coronal/sagittal) slice별로 대응하는 3D point를 계산하고,
    PerspectiveINRNet을 통해 point feature를 얻은 뒤 CT/mask ground truth를 crop해 반환.
    """

    def __init__(self, params):
        super().__init__()
        params = OmegaConf.to_container(params, resolve=True)
        self.metadata = params
        self.N_rays_ctslice_grad_on = self.metadata["N_rand_recon"]
        self.npoints_per_chunk = self.metadata["chunk"]
        self.no_grad_encoder = self.metadata["no_grad_cond_encoder"]
        self.cond_list = self.metadata["cond_list"]

        network_module = self.metadata["main_model_of_encoder"]["network_module"]
        network_module, net_class = network_module.rsplit(".", 1)
        cfg, self.network_query_fn = model_utils.update_nerf_params(
            **self.metadata["main_model_of_encoder"]["params"]["cfg"]["nerf_params"]
        )
        self.metadata["main_model_of_encoder"]["params"]["cfg"]["nerf_params"]["cfg"] = cfg

        network_module = getattr(__import__(network_module, fromlist=[net_class]), net_class)
        self.network_fn = network_module(**self.metadata["main_model_of_encoder"]["params"])
        self.output_ch = self.network_fn.output_ch

        self.ct_res = torch.tensor([self.metadata["ct_res"]] * 3)
        self.feature_res = torch.tensor([self.metadata["feature_res"]] * 3)
        assert (self.metadata["ct_res"] % self.metadata["feature_res"]) == 0
        self.fstep = int(self.metadata["ct_res"] / self.metadata["feature_res"])

        self.max_length_gt_coord = self.metadata["ct_res"]
        self.max_length_world_coord = 1
        self.center_gt_coord = (self.ct_res - 1) / 2

    def gt2world_coordinate(self, pts_in_gt):
        device = pts_in_gt.get_device()
        pts_in_gt = pts_in_gt - self.center_gt_coord.to(device)
        return pts_in_gt / self.max_length_gt_coord * self.max_length_world_coord

    def rendering_from_ctslice(self, ct_slice, output_ct_res):
        """(no zoom) X-ray slice 전체를 output_ct_res 그리드로 리샘플링."""
        lin = torch.linspace(-1, 1, output_ct_res)
        pts = torch.meshgrid(lin, lin)
        pts_z = torch.zeros_like(pts[0])
        pts = torch.cat((pts[0].unsqueeze(-1), pts[1].unsqueeze(-1), pts_z.unsqueeze(-1)), dim=-1)
        pts = pts.unsqueeze(0).unsqueeze(0).to(ct_slice.device)

        raw = F.grid_sample(ct_slice.unsqueeze(0), pts, mode="bilinear",
                             align_corners=True, padding_mode="zeros").squeeze(0).permute(0, 1, 3, 2)
        return raw.repeat(1, 3, 1, 1)

    def rendering_from_mask(self, mask, output_ct_res):
        """(no zoom) segmentation mask 전체를 output_ct_res 그리드로 리샘플링 (label 보존, nearest)."""
        lin = torch.linspace(-1, 1, output_ct_res)
        pts = torch.meshgrid(lin, lin)
        pts_z = torch.zeros_like(pts[0])
        pts = torch.cat((pts[0].unsqueeze(-1), pts[1].unsqueeze(-1), pts_z.unsqueeze(-1)), dim=-1)
        pts = pts.unsqueeze(0).unsqueeze(0).to(mask.device)

        raw = F.grid_sample(mask.unsqueeze(0), pts, mode="nearest",
                             align_corners=True, padding_mode="zeros").squeeze(0).permute(0, 1, 3, 2)
        return raw

    def get_rays_for_no_rendering(self, inputs: dict):
        batch_size, _, H, W = inputs[inputs["image_key"]].shape
        res = self.ct_res
        pts_in_gt = torch.meshgrid(
            torch.linspace(0, res[0] - 1, res[0]),
            torch.linspace(0, res[1] - 1, res[1]),
            torch.linspace(0, res[2] - 1, res[2]),
        )
        pts_in_gt = torch.cat((pts_in_gt[0].unsqueeze(-1), pts_in_gt[1].unsqueeze(-1),
                               pts_in_gt[2].unsqueeze(-1)), dim=-1).unsqueeze(0).cuda()
        transformed_points = self.gt2world_coordinate(pts_in_gt)
        device = transformed_points.get_device()
        self.ct_res = self.ct_res.to(device)

        transformed_feature_points, gt_ctslices, axis = [], [], []
        has_mask = "mask" in inputs
        gt_masks = [] if has_mask else None

        for b in range(batch_size):
            file_path = inputs["file_path_"][b].split("/")[-1]
            recon_axis, slice_idx = os.path.splitext(file_path)[0].split("_")
            slice_idx = int(slice_idx)
            gt_ctslice = inputs[inputs["image_key"]][b:b + 1]

            if recon_axis == "sagittal":
                feature_point = transformed_points[0, ::self.fstep, ::self.fstep, slice_idx:slice_idx + 1, :]
                offset = (feature_point[1, 1, :, :] - feature_point[0, 0, :, :]) * (self.fstep - 1) / self.fstep
                feature_point = feature_point + offset / 2
            elif recon_axis == "coronal":
                feature_point = transformed_points[0, ::self.fstep, slice_idx:slice_idx + 1, ::self.fstep, :]
                offset = (feature_point[1, :, 1, :] - feature_point[0, :, 0, :]) * (self.fstep - 1) / self.fstep
                feature_point = (feature_point + offset / 2).permute(0, 2, 1, 3)
            else:  # axial
                feature_point = transformed_points[0, slice_idx:slice_idx + 1, ::self.fstep, ::self.fstep, :]
                offset = (feature_point[:, 1, 1, :] - feature_point[:, 0, 0, :]) * (self.fstep - 1) / self.fstep
                feature_point = (feature_point + offset / 2).permute(1, 2, 0, 3)

            axis.append(recon_axis)
            transformed_feature_points.append(feature_point)
            gt_ctslices.append(self.rendering_from_ctslice(gt_ctslice, output_ct_res=self.ct_res[0].item()))
            if has_mask:
                gt_masks.append(self.rendering_from_mask(inputs["mask"][b:b + 1],
                                                          output_ct_res=self.ct_res[0].item()))

        transformed_feature_points = torch.stack(transformed_feature_points)
        gt_ctslices = torch.cat(gt_ctslices, dim=0)
        if has_mask:
            gt_masks = torch.cat(gt_masks, dim=0)
        return transformed_feature_points, gt_ctslices, axis, gt_masks

    def forward(self, inputs: dict):
        with torch.no_grad():
            transformed_points, gt_ctslices, axis, gt_masks = self.get_rays_for_no_rendering(inputs)
            nrays_grad_on = self.N_rays_ctslice_grad_on
            batch_size, H, W, N_samples, coord_dim = transformed_points.shape
            transformed_points = transformed_points.reshape(batch_size, -1, coord_dim)

        skipfeatures = None
        src_imgs = [inputs[k] for k in self.cond_list]
        src_camposes = [inputs[f"{k}_cam"] for k in self.cond_list]
        src_imgs = torch.stack(src_imgs, dim=0)
        src_camposes = torch.stack(src_camposes, dim=0)

        if self.no_grad_encoder:
            with torch.no_grad():
                latent_zs = self.network_fn.encode(src_imgs, src_camposes, transformed_points)
                skipfeatures = self.network_fn.latest_skipfeatures
        else:
            latent_zs = self.network_fn.encode(src_imgs, src_camposes, transformed_points)
            skipfeatures = self.network_fn.latest_skipfeatures

        all_outputs = self.run_nerf(
            (batch_size, H, W, N_samples), transformed_points, nrays_grad_on, latent_zs
        )
        all_outputs["outputs"] = all_outputs["outputs"].permute(0, 2, 1).reshape(batch_size, -1, H, W)
        all_outputs["cropped_ctslice"] = gt_ctslices
        if gt_masks is not None:
            all_outputs["cropped_mask"] = gt_masks
        all_outputs["skipfeatures"] = skipfeatures
        return all_outputs

    def run_nerf(self, org_shape, transformed_points, nrays_grad_on, latent_zs):
        batch_size, H, W, N_samples = org_shape
        nerf_inputs = torch.cat((transformed_points, latent_zs), dim=-1)
        device = transformed_points.get_device()

        all_outputs = {"outputs": torch.zeros((batch_size, transformed_points.shape[1], self.output_ch)).to(device)}
        n_split = (transformed_points.shape[1] // self.npoints_per_chunk) + 1
        for split in range(n_split):
            idx = slice(split * self.npoints_per_chunk, (split + 1) * self.npoints_per_chunk)
            chunk = nerf_inputs[:, idx]
            if chunk.shape[1] == 0:
                continue
            output = self.network_query_fn(chunk, self.network_fn)
            for k in output:
                if output[k].dtype != all_outputs[k].dtype:
                    all_outputs[k] = all_outputs[k].type(output[k].dtype)
                all_outputs[k][:, idx] = output[k]
        return all_outputs