import torch
import torch.nn as nn

from x2ct_nerf.modules.losses.lpips import LPIPS


class ReconPerceptualLoss(nn.Module):
    """L1 + perceptual(LPIPS). Discriminator/Laplacian 없음."""

    def __init__(self, pixelloss_weight=1.0, perceptual_weight=1.0):
        super().__init__()
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

    def forward(self, inputs, reconstructions, split="train"):
        rec_loss = self.pixel_weight * torch.mean(
            torch.abs(inputs.contiguous() - reconstructions["outputs"].contiguous())
        )

        p_loss = torch.tensor(0.0, device=rec_loss.device)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_weight * torch.mean(
                self.perceptual_loss(inputs.contiguous(), reconstructions["outputs"].contiguous())
            )

        loss = rec_loss + p_loss

        log = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/rec_loss": rec_loss.detach(),
            f"{split}/p_loss": p_loss.detach(),
        }
        return loss, log