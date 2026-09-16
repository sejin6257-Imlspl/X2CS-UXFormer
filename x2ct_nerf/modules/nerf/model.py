# x2ct_nerf/modules/nerf/model.py
import torch.nn as nn


class DummyNeRF(nn.Module):
    """실제 NeRF가 아니라, 입력 채널을 output 채널로 projection하는 단순 linear head."""

    def __init__(self, cfg):
        super().__init__()
        self.input_ch = cfg["input_ch"]
        self.output_ch = cfg["output_ch"]
        print(f"[DummyNeRF] Input ch : {self.input_ch}, Output ch : {self.output_ch}")

        if self.input_ch != self.output_ch:
            self.linear = nn.Linear(self.input_ch, self.output_ch)

    def forward(self, x):
        if self.input_ch != self.output_ch:
            x = self.linear(x)
        return {"outputs": x}