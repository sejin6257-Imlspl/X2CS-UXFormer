import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import timm

from x2ct_nerf.modules.image_encoder.get_transform import get_data_transform


def _load_backbone(pretrained, weight_dir, model_name):
    print(f"model name : {model_name}")
    if pretrained == "imagenet":
        return getattr(torchvision.models, model_name)(pretrained=True)
    if pretrained == "autoenc_LIDC":
        model = getattr(torchvision.models, model_name)(pretrained=False)
        state = torch.load(weight_dir)["network_state_dict"]
        model_dict = model.state_dict()
        model_dict.update({k: v for k, v in state.items() if k in model_dict})
        model.load_state_dict(model_dict)
        return model
    if pretrained is None:
        print("Use ResNet Image Encoder from scratch")
        return getattr(torchvision.models, model_name)(pretrained=False)
    raise NotImplementedError(f"Unknown pretrained option: {pretrained}")


def _group_norm(channels):
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class XrayToSwinInput(nn.Module):
    """X-ray 텐서를 Swin backbone 입력 규격(256x256, ImageNet 정규화)으로 변환."""

    def __init__(self, out_size=256):
        super().__init__()
        self.resize = T.Resize((out_size, out_size), antialias=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got {x.shape}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.normalize(self.resize(x))


class CrossAttnBlock(nn.Module):
    """Query=CNN(local) feature, Key/Value=Swin(global) feature. Residual 연결."""

    def __init__(self, channels):
        super().__init__()
        self.norm_q = _group_norm(channels)
        self.norm_kv = _group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, q_feat, kv_feat):
        b, c, h, w = q_feat.shape
        q = self.q(self.norm_q(q_feat)).reshape(b, c, h * w).permute(0, 2, 1)
        k = self.k(self.norm_kv(kv_feat)).reshape(b, c, h * w)
        v = self.v(self.norm_kv(kv_feat)).reshape(b, c, h * w)

        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=2)
        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, c, h, w)
        return q_feat + self.proj_out(out)


_FUSION_STAGES = {
    "layer0": {"resnet_ch": 64,  "proj_ch": 96},
    "layer1": {"resnet_ch": 256, "proj_ch": 192},
    "layer2": {"resnet_ch": 512, "proj_ch": 384},
}


class ResNetEncoder(nn.Module):
    """
    CNN(local) + Swin(global) 듀얼 브랜치 X-ray 인코더.
    - decoder skip connection: 항상 `feature_layer`까지의 모든 fusion feature 사용.
    - back-projection 입력: 기본값은 raw local feature, `use_fusion_backproj=True`면
      `feature_layer` 지점의 local-global fusion feature.
    - `save_feature_every > 0`이면 N번째 (PA, Lateral) 쌍마다 fused feature map을
      흑백 heatmap PNG로 자동 저장. cond_list 순서(PA가 먼저)를 전제로 함.
    """

    def __init__(self, cfg):
        super().__init__()
        if not isinstance(cfg, dict):
            cfg = vars(cfg)
        self.downsample = nn.ModuleDict({
            stage: nn.Conv2d(info["resnet_ch"], info["resnet_ch"], kernel_size=2, stride=2)
            for stage, info in _FUSION_STAGES.items()
        })
        self.in_channels = cfg["in_channels"]
        self.latent_dim = cfg["latent_dim"]
        self.input_img_size = cfg["input_img_size"]
        self.encoder_freeze_layer = cfg["encoder_freeze_layer"]
        self.feature_layer = cfg["feature_layer"]
        self.pretrained = cfg["pretrained"]
        self.weight_dir = cfg["weight_dir"]
        self.autocast = cfg.get("autocast", False)
        self.use_fusion_backproj = cfg.get("use_fusion_backproj", False)

        # feature map 시각화 저장 옵션 (0이면 비활성)
        self._save_every = cfg.get("save_feature_every", 0)
        self._save_dir = cfg.get("save_feature_dir", "./feature_maps")
        self._pending_pa_skipfeatures = None
        self._pair_count = 0

        assert not (self.in_channels == 1 and self.encoder_freeze_layer)
        assert self.feature_layer in ["layer1", "layer2", "layer3", "layer4", "all"]
        assert self.pretrained in [None, "autoenc_LIDC", "imagenet"]
        if self.use_fusion_backproj:
            assert self.feature_layer in _FUSION_STAGES, (
                f"use_fusion_backproj currently supports feature_layer in "
                f"{list(_FUSION_STAGES)}, got {self.feature_layer}"
            )

        model_name = cfg.get("model_name", "resnet34")
        self.model = _load_backbone(self.pretrained, self.weight_dir, model_name)
        self.transform = get_data_transform(in_channels=self.in_channels, input_img_size=self.input_img_size)

        self.swin_preprocess = XrayToSwinInput(out_size=256)
        self.swin_backbone = timm.create_model(
            "swinv2_tiny_window8_256.ms_in1k",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2),
            in_chans=3,
        )

        if self.in_channels != 3:
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.proj = nn.ModuleDict({
            stage: nn.Conv2d(info["resnet_ch"], info["proj_ch"], 1)
            for stage, info in _FUSION_STAGES.items()
        })
        self.cross_attn = nn.ModuleDict({
            stage: CrossAttnBlock(info["proj_ch"])
            for stage, info in _FUSION_STAGES.items()
        })

        if self.feature_layer != "all":
            del self.model.fc
            del self.model.avgpool
            for layer in ["layer1", "layer2", "layer3", "layer4"][::-1]:
                if layer == self.feature_layer:
                    break
                delattr(self.model, layer)
            self.output_dim = self._resolve_output_dim()
        else:
            self.output_dim = self.model.fc.out_features
            if self.output_dim != self.latent_dim:
                self.model.fc = nn.Linear(self.model.fc.in_features, self.latent_dim)

        self.output_ch = self.latent_dim

        if self.use_fusion_backproj:
            fusion_ch = _FUSION_STAGES[self.feature_layer]["proj_ch"]
            self.fusion_out_proj = nn.Conv2d(fusion_ch, self.latent_dim, kernel_size=1)
        elif self.feature_layer != "all" and self.output_dim != self.latent_dim:
            self.local_out_proj = nn.Sequential(
                nn.Conv2d(self.output_dim, self.latent_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.latent_dim),
                nn.ReLU(),
            )

        if self.pretrained and self.encoder_freeze_layer:
            for name, param in self.model.named_parameters():
                if self.encoder_freeze_layer in name:
                    break
                param.requires_grad = False

        # --- Swin branch freeze (새로 추가) ---
        # ResNet 쪽과 동일한 패턴: swin_freeze_layer 문자열이 이름에 처음 등장하는
        # 파라미터를 만나면 멈추고, 그 이전 파라미터들만 requires_grad=False로 고정.
        # 예: swin_freeze_layer="layers_1" -> patch_embed + layers_0(첫 stage) 고정.
        self.swin_freeze_layer = cfg.get("swin_freeze_layer", None)
        if self.swin_freeze_layer:
            for name, param in self.swin_backbone.named_parameters():
                if self.swin_freeze_layer in name:
                    break
                param.requires_grad = False

    def _resolve_output_dim(self):
        if self.feature_layer in _FUSION_STAGES:
            return _FUSION_STAGES[self.feature_layer]["resnet_ch"]
        convs = [m for name, m in getattr(self.model, self.feature_layer)[-1].named_modules()
                 if name.startswith("conv")]
        return convs[-1].out_channels

    def _fuse(self, stage, local_feat, swin_feat, skipfeatures):
        local_feat = self.downsample[stage](local_feat)   # F.interpolate 대신 학습 가능한 stride-2 conv
        local_feat = self.proj[stage](local_feat)
        fused = self.cross_attn[stage](local_feat, swin_feat)
        skipfeatures[stage] = fused
        return fused

    def _save_feature_maps(self, pa_skipfeatures, lat_skipfeatures):
        """PA/Lateral의 layer0/1/2 fused feature map(채널 평균)을 흑백 heatmap으로 한 장에 저장."""
        import matplotlib.pyplot as plt

        os.makedirs(self._save_dir, exist_ok=True)
        stages = list(pa_skipfeatures.keys())  # ["layer0", "layer1", "layer2"]

        fig, axes = plt.subplots(2, len(stages), figsize=(5 * len(stages), 10))
        for col, stage in enumerate(stages):
            pa_map = pa_skipfeatures[stage][0].mean(dim=0).detach().cpu().numpy()
            lat_map = lat_skipfeatures[stage][0].mean(dim=0).detach().cpu().numpy()

            axes[0, col].imshow(pa_map, cmap="gray")
            axes[0, col].set_title(f"PA {stage} ({pa_skipfeatures[stage].shape[1]}ch)")
            axes[0, col].axis("off")

            axes[1, col].imshow(lat_map, cmap="gray")
            axes[1, col].set_title(f"Lateral {stage} ({lat_skipfeatures[stage].shape[1]}ch)")
            axes[1, col].axis("off")

        plt.tight_layout()
        save_path = os.path.join(self._save_dir, f"pair{self._pair_count:06d}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"[ResNetEncoder] feature map 저장: {save_path}")

    def forward(self, x):
        x_in = x
        x = self.transform(x)
        swin_feats = self.swin_backbone(self.swin_preprocess(x_in))
        swin = {
            "layer0": swin_feats[0].permute(0, 3, 1, 2).contiguous(),
            "layer1": swin_feats[1].permute(0, 3, 1, 2).contiguous(),
            "layer2": swin_feats[2].permute(0, 3, 1, 2).contiguous(),
        }

        skipfeatures = {}
        fusion_feat = None

        x = self.model.relu(self.model.bn1(self.model.conv1(x)))

        fused0 = self._fuse("layer0", x, swin["layer0"], skipfeatures)
        if self.feature_layer == "layer0":
            fusion_feat = fused0

        x = self.model.maxpool(x)

        for stage in ["layer1", "layer2", "layer3", "layer4"]:
            x = getattr(self.model, stage)(x)
            if stage in _FUSION_STAGES:
                fused = self._fuse(stage, x, swin[stage], skipfeatures)
                if self.feature_layer == stage:
                    fusion_feat = fused
            if stage == self.feature_layer:
                break

        # --- PA/Lateral 짝지어서 N번째 쌍마다 저장 (cond_list 순서: PA -> Lateral 전제) ---
        if self._save_every > 0:
            if self._pending_pa_skipfeatures is None:
                self._pending_pa_skipfeatures = {k: v.detach() for k, v in skipfeatures.items()}
            else:
                self._pair_count += 1
                if self._pair_count % self._save_every == 0:
                    self._save_feature_maps(self._pending_pa_skipfeatures, skipfeatures)
                self._pending_pa_skipfeatures = None

        if self.use_fusion_backproj:
            assert fusion_feat is not None, f"fusion_feat not set for feature_layer={self.feature_layer}"
            x = self.fusion_out_proj(fusion_feat)
        elif self.feature_layer != "all":
            if hasattr(self, "local_out_proj"):
                x = self.local_out_proj(x)
        else:
            x = self.model.fc(torch.flatten(self.model.avgpool(x), 1))

        if self.autocast:
            return x
        return x.float(), skipfeatures
