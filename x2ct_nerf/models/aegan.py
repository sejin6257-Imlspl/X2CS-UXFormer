import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from importlib import import_module

from utils.metrics import mse2psnr, img2mse


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    obj = getattr(import_module(module), cls)
    return obj(**config.get("params", dict()))


class AEModel(pl.LightningModule):
    """Autoencoder 기반 최상위 모델. Encoder/Decoder를 config로 조립하고 recon loss로 학습."""

    def __init__(self, ddconfig, lossconfig, ckpt_path=None, ignore_keys=[],
                 image_key="image", monitor=None, metadata={}):
        super().__init__()
        self.metadata = metadata
        self.image_key = image_key
        self.print_loss_per_step = metadata.get("print_loss_per_step", 50)

        encoder_module, decoder_module = self._get_encoder_decoder_module()
        encoder_params = metadata.get("encoder_params", ddconfig)
        decoder_params = metadata.get("decoder_params", ddconfig)
        self.encoder = encoder_module(**encoder_params)
        self.decoder = decoder_module(**decoder_params)
        self.loss = instantiate_from_config(lossconfig)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor

        self.psnr = lambda recon, gt: mse2psnr(img2mse(recon.contiguous(), gt.contiguous()))

    def _get_encoder_decoder_module(self):
        enc_path = self.metadata.get("encoder_module")
        dec_path = self.metadata.get("decoder_module", "taming.modules.diffusionmodules.model.Decoder")
        enc_module, enc_cls = enc_path.rsplit(".", 1)
        dec_module, dec_cls = dec_path.rsplit(".", 1)
        return getattr(import_module(enc_module), enc_cls), getattr(import_module(dec_module), dec_cls)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, feature, skipfeatures=None):
        return self.decoder(feature, skipfeatures=skipfeatures)

    def forward(self, input):
        feature = self.encode(input)
        assert not isinstance(feature, dict)
        return {"outputs": self.decode(feature)}

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

    @torch.no_grad()
    def print_loss(self, loss_dict):
        if self.global_step % self.print_loss_per_step != 0:
            return
        losses = ""
        split = "train"
        for k, v in loss_dict.items():
            split, loss_name = k.split("/")
            losses += f", {loss_name} : {float(v):.3f}"
        print(f"[{split} Step{self.global_step}] : {losses[2:]}")

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)
        loss, log_dict = self.loss(x, xrec["outputs"], split="train")
        log_dict["train/psnr"] = self.psnr(xrec["outputs"], x)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.print_loss(log_dict)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)
        loss, log_dict = self.loss(x, xrec["outputs"], split="val")
        log_dict["val/psnr"] = self.psnr(xrec["outputs"], x)
        self.log(self.monitor, log_dict[self.monitor], prog_bar=True, logger=True,
                  on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.print_loss(log_dict)

    def configure_optimizers(self):
        return torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate, betas=(0.5, 0.9),
        )

    def get_last_layer(self):
        try:
            return self.decoder.conv_out.weight
        except AttributeError:
            return None

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        x = self.get_input(batch, self.image_key).to(self.device)
        xrec = self(x)["outputs"]
        return {"inputs": x, "reconstructions": xrec}