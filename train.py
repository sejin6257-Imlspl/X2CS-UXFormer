import os
import sys
import datetime
import argparse
import importlib

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from taming.data.utils import custom_collate

sys.path.append(os.getcwd())


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default = parser.parse_args([])
    return sorted(k for k in vars(default) if getattr(opt, k) != getattr(default, k))


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.dataset_configs = {}
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        for cfg in self.dataset_configs.values():
            instantiate_from_config(cfg)

    def setup(self, stage=None):
        self.datasets = {k: instantiate_from_config(v) for k, v in self.dataset_configs.items()}

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"], batch_size=self.batch_size,
                           num_workers=self.num_workers, collate_fn=custom_collate)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                           num_workers=self.num_workers, collate_fn=custom_collate)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="")
    parser.add_argument("-b", "--base", nargs="*", default=list(),
                         help="paths to base configs, left-to-right merge")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=23)
    parser.add_argument("-f", "--postfix", type=str, default="")
    parser.add_argument("--max_step", type=int, default=None)
    return parser


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    seed_everything(opt.seed)

    cfg_name = os.path.splitext(os.path.split(opt.base[0])[-1])[0] if opt.base else "run"
    nowname = f"{opt.name or cfg_name}_{opt.postfix}_{now}"
    ckptdir = os.path.join("logs", nowname, "checkpoints")
    os.makedirs(ckptdir, exist_ok=True)

    configs = [OmegaConf.to_container(OmegaConf.load(c), resolve=True) for c in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if opt.max_step is not None:
        trainer_config["max_steps"] = opt.max_step
    trainer_opt = argparse.Namespace(**trainer_config)

    model = instantiate_from_config(config.model)

    from x2ct_nerf.callbacks.ema import EMACallback

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckptdir,
        filename="best-{epoch:03d}-{val/eval_loss:.4f}",
        monitor="val/eval_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    ema_callback = EMACallback(decay=0.999)

    trainer = Trainer.from_argparse_args(trainer_opt, callbacks=[checkpoint_callback, ema_callback])

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    ngpu = 1
    gpus_val = getattr(trainer_opt, "gpus", None)
    if gpus_val:
        ngpu = (len(str(gpus_val).strip(",").split(","))
                if not isinstance(gpus_val, int) else max(gpus_val, 1))
    accumulate_grad_batches = getattr(trainer_opt, "accumulate_grad_batches", 1) or 1
    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr

    trainer.fit(model, data)