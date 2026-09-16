import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from x2ct_nerf.modules import volume_rendering as vr


class PerspectiveINRNet(nn.Module):
    """
    X-ray cond_encoder(ResNet+Swin) 출력 feature를 3D point에 투영(back-projection)한 뒤,
    DummyNeRF(단순 linear head)로 최종 출력 채널을 만드는 네트워크.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cond_encoder = self._get_model(cfg["cond_encoder_module"])(**cfg["cond_encoder_params"])
        self.cond_encoder_output_ch = self.cond_encoder.output_ch

        cfg["nerf_params"]["cfg"]["input_ch"] += self.cond_encoder_output_ch * cfg["N_cond"]
        self.nerf = self._get_model(cfg["nerf_module"])(**cfg["nerf_params"])
        self.nerf_input_ch = cfg["nerf_params"]["cfg"]["input_ch"]
        self.output_ch = self.nerf.output_ch

    @staticmethod
    def _get_model(module_name):
        module_path, cls_name = module_name.rsplit(".", 1)
        module = __import__(module_path, fromlist=[cls_name])
        return getattr(module, cls_name)

    def forward(self, x):
        x = x[..., -self.nerf_input_ch:]
        return self.nerf(x)

    def encode(self, src_images, src_camposes, render_pts):
        """
        src_images: NS x B x C x H x W (NS=view 수, PA+Lateral=2)
        src_camposes: NS x B x 2 (pitch, yaw)
        render_pts: B x (H*W*N_samples) x 3
        """
        self.latest_skipfeatures = []
        features = None
        for src_image, src_campose in zip(src_images, src_camposes):
            feature, skipfeatures = self.cond_encoder(src_image)
            self.latest_skipfeatures.append(skipfeatures)

            feature = self._project_to_3d(feature, src_campose, render_pts)
            features = torch.cat((features, feature), dim=-1) if features is not None else feature

        return features

    def _project_to_3d(self, features, src_campose, render_pts):
        """
        2D X-ray feature map을 카메라 pose 기준으로 3D point에 투영해 샘플링.
        features: (B, C, H, W)
        src_campose: (B, 2) — pitch, yaw
        render_pts: (B, N, 3)
        """
        device = render_pts.device
        features = features.to(device)
        src_campose = src_campose.to(device)
        batch, n_pts, _ = render_pts.shape

        ones = torch.ones((batch, n_pts, 1), device=device)
        render_pts_h = torch.cat((render_pts, ones), dim=-1)

        pitch, yaw = src_campose[..., 0:1], src_campose[..., 1:2]
        camera_origin, _, _ = vr.sample_camera_positions(n=batch, r=1, device=device, phi=pitch, theta=yaw)
        camera_origin[:, 0][camera_origin[:, 0] == 0] = 1e-5  # 특이점(0-division) 회피

        forward_vector = vr.normalize_vecs(-camera_origin)
        cam2world = vr.create_cam2world_matrix(forward_vector, camera_origin, device=device)
        world2cam = torch.inverse(cam2world.float())

        points_in_src = torch.bmm(world2cam, render_pts_h.permute(0, 2, 1)).permute(0, 2, 1)
        points_uv = -points_in_src[..., :2] / points_in_src[..., 2:3]
        points_xy = points_uv / np.tan((2 * math.pi * self.cfg["fov"] / 360) / 2)
        points_xy = torch.cat([points_xy[..., 0:1], -points_xy[..., 1:2]], -1).unsqueeze(2)

        feature_in_src = F.grid_sample(
            features, points_xy, mode="bilinear", align_corners=True, padding_mode="zeros"
        ).permute(0, 2, 3, 1)
        return feature_in_src.reshape(batch, n_pts, -1)