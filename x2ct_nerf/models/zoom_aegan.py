import torch
from torchvision.transforms import Resize

from x2ct_nerf.models.aegan import AEModel


class INRAEZoomModel(AEModel):
    """Reconstruction-only 모델. CT ground truth와 X-ray(PA+Lateral) 조건으로 3D CT를 복원."""

    def __init__(self, ddconfig, lossconfig, ckpt_path=None, ignore_keys=[],
                 image_key="image", monitor=None, metadata={}):
        super().__init__(ddconfig, lossconfig, ckpt_path=None, ignore_keys=ignore_keys,
                          image_key=image_key, monitor=monitor, metadata=metadata)

        self.gt_key = metadata.get("gt_key")
        print(f"gt_key : {self.gt_key}")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def get_input(self, batch):
        xs = {}
        for k, x in batch.items():
            if k in ["ctslice", "PA", "Lateral"]:
                if len(x.shape) == 3:
                    x = x[..., None]
                x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                xs[k] = x.float().to(self.device)
            else:
                xs[k] = x
        return xs

    def forward(self, input):
        feature = self.encode(input)
        x = input[self.image_key] if self.gt_key == self.image_key else feature[self.gt_key]

        skipfeatures = feature.get("skipfeatures", None)
        feature = feature["outputs"]

        dec = self.decode(feature, skipfeatures=skipfeatures)
        if dec.shape[1] == 1:
            dec = torch.cat([dec] * 3, dim=1)

        return {"outputs": dec, "skipfeatures": skipfeatures}, x

    def training_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        batch["image_key"] = self.image_key
        xrec_dict, x = self(batch)

        loss, log_dict = self.loss(x, xrec_dict, split="train")
        log_dict["train/psnr"] = self.psnr(xrec_dict["outputs"], x)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.print_loss(log_dict)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        batch["image_key"] = self.image_key
        xrec_dict, x = self(batch)

        loss, log_dict = self.loss(x, xrec_dict, split="val")
        log_dict["val/psnr"] = self.psnr(xrec_dict["outputs"], x)
        self.log(self.monitor, log_dict[self.monitor], prog_bar=True, logger=True,
                  on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.print_loss(log_dict)

    def configure_optimizers(self):
        return torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate, betas=(0.5, 0.9),
        )

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        batch = self.get_input(batch)
        batch["image_key"] = self.image_key
        x = batch[self.image_key].to(self.device)

        h = self.encoder(batch)
        skipfeatures = h.get("skipfeatures", None)
        x_rec = self.decode(h["outputs"], skipfeatures=skipfeatures)
        if x_rec.shape[1] == 1:
            x_rec = torch.cat([x_rec] * 3, dim=1)
        input_resolution = x_rec.shape[-1]

        for k, v in batch.items():
            if k in self.metadata["encoder_params"]["params"]["cond_list"]:
                log[k] = v

        log["inputs"] = torch.cat((x[:, 1:2], x[:, 1:2], x[:, 1:2]), dim=1)
        if self.gt_key in h:
            crop_ct = h[self.gt_key]
            log[self.gt_key] = torch.cat((crop_ct[:, 1:2],) * 3, dim=1)
        log["reconstructions"] = torch.cat((x_rec[:, 1:2].clone(),) * 3, dim=1)

        for k, v in log.items():
            if v.shape[-1] != input_resolution:
                log[k] = Resize(input_resolution)(v)
        return log