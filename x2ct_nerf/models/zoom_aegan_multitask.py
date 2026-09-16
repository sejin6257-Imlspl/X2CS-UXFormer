import torch
import torch.nn.functional as F
from torchvision.transforms import Resize

from x2ct_nerf.models.zoom_aegan import INRAEZoomModel
from x2ct_nerf.modules.losses.seg_loss import SegLoss


class INRAEZoomMultiTaskModel(INRAEZoomModel):
    """Reconstruction + Segmentation multi-task 모델.
    GradNorm(loss-level weighting) + AdaTask(optimizer-level scaling) 결합."""

    def __init__(self, ddconfig, lossconfig, num_seg_classes, seg_decoder_params=None,
                 lambda_rec=1.0, lambda_seg=1.0, gradnorm_alpha=0.4, gradnorm_lr=0.025,
                 seg_loss_config=None, ckpt_path=None, ignore_keys=[], image_key="image",
                 monitor=None, metadata={}):
        super().__init__(ddconfig, lossconfig, ckpt_path=None, ignore_keys=ignore_keys,
                          image_key=image_key, monitor=monitor, metadata=metadata)

        self.num_seg_classes = num_seg_classes
        self.num_tasks = 2  # [recon, seg]

        _, decoder_module = self._get_encoder_decoder_module()
        seg_dec_params = dict(seg_decoder_params) if seg_decoder_params else dict(ddconfig)
        seg_dec_params["out_ch"] = num_seg_classes
        self.seg_decoder = decoder_module(**seg_dec_params)

        seg_loss_config = seg_loss_config or {}
        self.seg_loss_fn = SegLoss(num_classes=num_seg_classes, **seg_loss_config)

        self.gradnorm_alpha = gradnorm_alpha
        self.gradnorm_lr = gradnorm_lr
        self.task_weights = torch.nn.Parameter(torch.tensor([float(lambda_rec), float(lambda_seg)]))
        self.register_buffer("initial_losses", torch.ones(self.num_tasks))
        self.register_buffer("initial_losses_set", torch.tensor(False))

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.automatic_optimization = False

    def get_input(self, batch):
        xs = super().get_input(batch)
        if "mask" in batch:
            mask = batch["mask"]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            xs["mask"] = mask.float().to(self.device)
        return xs

    def forward(self, input):
        feature = self.encode(input)
        x = input[self.image_key] if self.gt_key == self.image_key else feature[self.gt_key]

        cropped_mask = feature.get("cropped_mask", None)
        skipfeatures = feature.get("skipfeatures", None)
        z = feature["outputs"]

        rec = self.decode(z, skipfeatures=skipfeatures)
        if rec.shape[1] == 1:
            rec = torch.cat([rec] * 3, dim=1)

        seg_logits = self.seg_decoder(z, skipfeatures=skipfeatures)
        return {"outputs": rec, "seg_logits": seg_logits, "skipfeatures": skipfeatures,
                "shared_z": z}, x, cropped_mask

    def training_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        batch["image_key"] = self.image_key
        out, x, cropped_mask = self(batch)
        rec_dict = {"outputs": out["outputs"]}

        rec_loss, log_dict = self.loss(x, rec_dict, split="train")
        log_dict["train/psnr"] = self.psnr(rec_dict["outputs"], x)

        seg_loss = torch.tensor(0.0, device=self.device)
        has_seg = cropped_mask is not None
        if has_seg:
            target = cropped_mask.squeeze(1).long()
            seg_loss, seg_log = self.seg_loss_fn(out["seg_logits"], target)
            for k, v in seg_log.items():
                log_dict[f"train/seg_{k}"] = v

        task_losses = torch.stack([rec_loss.mean(), seg_loss.mean()])

        if has_seg and not bool(self.initial_losses_set):
            self.initial_losses = task_losses.detach().clamp(min=1e-8)
            self.initial_losses_set = torch.tensor(True, device=self.device)

        do_combo = has_seg and bool(self.initial_losses_set)
        gradnorm_loss = torch.tensor(0.0, device=self.device)

        if do_combo:
            shared_z = out["shared_z"]
            base_norms = torch.stack([
                torch.norm(torch.autograd.grad(task_losses[i], shared_z, retain_graph=True)[0]).detach()
                for i in range(self.num_tasks)
            ])
            norms = self.task_weights * base_norms

            loss_ratios = task_losses.detach() / self.initial_losses
            inv_rate = loss_ratios / loss_ratios.mean()
            target_norm = (norms.mean().detach() * inv_rate ** self.gradnorm_alpha).detach()

            gradnorm_loss = torch.abs(norms - target_norm).sum()
            self.task_weights.grad = torch.autograd.grad(gradnorm_loss, self.task_weights)[0]

            w_rec_cur, w_seg_cur = self.task_weights.detach()

            enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
            theta_t = [p.data.clone() for p in enc_params]
            base_lr = self.learning_rate

            # --- recon branch: backward는 weight 없이, encoder lr에만 weight 반영 ---
            self.opt_enc_recon.zero_grad()
            self.opt_dec_recon.zero_grad()
            self.manual_backward(rec_loss, retain_graph=True)
            for g in self.opt_enc_recon.param_groups:
                g["lr"] = base_lr * w_rec_cur.item()
            self.opt_enc_recon.step()
            self.opt_dec_recon.step()   # decoder는 task-specific이라 weight 영향 없이 그대로
            delta_rec = [p.data.clone() - t0 for p, t0 in zip(enc_params, theta_t)]

            with torch.no_grad():
                for p, t0 in zip(enc_params, theta_t):
                    p.data.copy_(t0)

            # --- seg branch: 동일 패턴 ---
            self.opt_enc_seg.zero_grad()
            self.opt_dec_seg.zero_grad()
            self.manual_backward(seg_loss, retain_graph=True)
            for g in self.opt_enc_seg.param_groups:
                g["lr"] = base_lr * w_seg_cur.item()
            self.opt_enc_seg.step()
            self.opt_dec_seg.step()
            delta_seg = [p.data.clone() - t0 for p, t0 in zip(enc_params, theta_t)]

            # --- 진단: delta_rec/delta_seg의 norm이 w_rec/w_seg에 실제로 비례하는지 확인 ---
            with torch.no_grad():
                delta_rec_norm = torch.sqrt(sum((d ** 2).sum() for d in delta_rec))
                delta_seg_norm = torch.sqrt(sum((d ** 2).sum() for d in delta_seg))
                log_dict["diag/delta_rec_norm"] = delta_rec_norm
                log_dict["diag/delta_seg_norm"] = delta_seg_norm
                log_dict["diag/delta_rec_norm_per_w"] = delta_rec_norm / (w_rec_cur + 1e-8)
                log_dict["diag/delta_seg_norm_per_w"] = delta_seg_norm / (w_seg_cur + 1e-8)

            with torch.no_grad():
                for p, t0, dr, ds in zip(enc_params, theta_t, delta_rec, delta_seg):
                    p.data.copy_(t0 + dr + ds)

            self.opt_gradnorm.step()
            with torch.no_grad():
                self.task_weights.data.clamp_(min=1e-3)
                self.task_weights.data.mul_(self.num_tasks / self.task_weights.data.sum())
        else:
            self.opt_enc_recon.zero_grad()
            self.opt_dec_recon.zero_grad()
            self.manual_backward(self.task_weights[0].detach() * rec_loss)
            self.opt_enc_recon.step()
            self.opt_dec_recon.step()

        log_dict["train/seg_loss"] = seg_loss.detach()
        log_dict["train/rec_loss_total"] = rec_loss.detach()
        log_dict["train/total_loss"] = (rec_loss + seg_loss).detach()
        log_dict["gradnorm/w_rec"] = self.task_weights[0].detach()
        log_dict["gradnorm/w_seg"] = self.task_weights[1].detach()
        if do_combo:
            log_dict["gradnorm/loss"] = gradnorm_loss.detach()

        self.log_dict({k: v.item() if hasattr(v, "item") else v for k, v in log_dict.items()},
                       prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.print_loss(log_dict)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        batch["image_key"] = self.image_key
        out, x, cropped_mask = self(batch)
        rec_dict = {"outputs": out["outputs"]}

        rec_loss, log_dict = self.loss(x, rec_dict, split="val")
        log_dict["val/psnr"] = self.psnr(rec_dict["outputs"], x)

        seg_loss = torch.tensor(0.0, device=self.device)
        if cropped_mask is not None:
            target = cropped_mask.squeeze(1).long()
            seg_loss, seg_log = self.seg_loss_fn(out["seg_logits"], target)
            for k, v in seg_log.items():
                log_dict[f"val/seg_{k}"] = v

        log_dict["val/seg_loss"] = seg_loss.detach()
        log_dict["val/rec_loss_total"] = rec_loss.detach()
        log_dict["val/total_loss"] = (rec_loss + seg_loss).detach()
        log_dict["val/eval_loss"] = (rec_loss + seg_loss).detach()

        self.log(self.monitor, log_dict[self.monitor], prog_bar=True, logger=True,
                  on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict({k: v for k, v in log_dict.items() if k != self.monitor},
                       prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.print_loss(log_dict)

    def configure_optimizers(self):
        lr = self.learning_rate
        weight_decay = self.metadata.get("weight_decay", 0.0)
        self.opt_enc_recon = torch.optim.AdamW(self.encoder.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)
        self.opt_enc_seg = torch.optim.AdamW(self.encoder.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)
        self.opt_dec_recon = torch.optim.AdamW(self.decoder.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)
        self.opt_dec_seg = torch.optim.AdamW(self.seg_decoder.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)
        self.opt_gradnorm = torch.optim.Adam([self.task_weights], lr=self.gradnorm_lr)
        return [self.opt_enc_recon, self.opt_enc_seg, self.opt_dec_recon,
                self.opt_dec_seg, self.opt_gradnorm], []

    def _colorize_mask(self, mask):
        palette = torch.tensor([
            [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255],
        ], dtype=torch.float32, device=mask.device) / 255.0
        idx = mask.clamp(0, palette.size(0) - 1).long()
        return palette[idx].permute(0, 3, 1, 2)

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
        seg_logits = self.seg_decoder(h["outputs"], skipfeatures=skipfeatures)
        seg_pred = seg_logits.argmax(dim=1)
        input_resolution = x_rec.shape[-1]

        for k, v in batch.items():
            if k in self.metadata["encoder_params"]["params"]["cond_list"]:
                log[k] = v

        log["inputs"] = torch.cat((x[:, 1:2],) * 3, dim=1)
        if self.gt_key in h:
            crop_ct = h[self.gt_key]
            log[self.gt_key] = torch.cat((crop_ct[:, 1:2],) * 3, dim=1)
        log["reconstructions"] = torch.cat((x_rec[:, 1:2].clone(),) * 3, dim=1)

        if "cropped_mask" in h:
            gt_m = h["cropped_mask"].squeeze(1).long()
            log["seg_gt"] = self._colorize_mask(gt_m)
            log["seg_pred"] = self._colorize_mask(seg_pred)

        if kwargs.get("return_raw_seg", False) and "cropped_mask" in h:
            log["seg_gt_raw"] = h["cropped_mask"].squeeze(1).long()
            log["seg_pred_raw"] = seg_pred

        for k, v in log.items():
            if k.endswith("_raw"):
                continue
            if v.shape[-1] != input_resolution:
                log[k] = Resize(input_resolution)(v)
        return log