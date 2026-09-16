import torch
import pytorch_lightning as pl
from copy import deepcopy


class EMACallback(pl.Callback):
    """
    학습 파라미터의 Exponential Moving Average를 유지하다가,
    validation 시점에만 EMA 가중치로 잠깐 바꿔서 평가하고,
    validation 끝나면 원래(학습 중인) 가중치로 복원한다.
    체크포인트 저장 시 EMA state_dict도 'ema_state_dict' 키로 같이 저장한다.
    """

    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_params = None
        self._backup_params = None

    def on_fit_start(self, trainer, pl_module):
        if self.ema_params is None:
            self.ema_params = {
                name: param.detach().clone()
                for name, param in pl_module.named_parameters()
                if param.requires_grad
            }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if not param.requires_grad:
                    continue
                self.ema_params[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def _swap_to_ema(self, pl_module):
        self._backup_params = {
            name: param.detach().clone()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.ema_params[name])

    def _restore_from_backup(self, pl_module):
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self._backup_params[name])
        self._backup_params = None

    def on_validation_start(self, trainer, pl_module):
        if self.ema_params is not None:
            self._swap_to_ema(pl_module)

    def on_validation_end(self, trainer, pl_module):
        if self._backup_params is not None:
            self._restore_from_backup(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema_params is not None:
            checkpoint["ema_state_dict"] = {k: v.cpu() for k, v in self.ema_params.items()}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_state_dict" in checkpoint:
            device = next(pl_module.parameters()).device
            self.ema_params = {k: v.to(device) for k, v in checkpoint["ema_state_dict"].items()}