import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0, ignore_bg=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits, target):
        # logits: (B, C, H, W), target: (B, H, W) long
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(
            target.clamp(0, self.num_classes - 1), num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        if self.ignore_bg:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        dims = (0, 2, 3)
        inter = (probs * target_oh).sum(dim=dims)
        card = (probs + target_oh).sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    """CE + Dice combo. forward()가 (loss, log_dict)를 반환."""

    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0,
                 class_weights=None, dice_ignore_bg=True):
        super().__init__()
        self.num_classes = num_classes
        self.ce_w = ce_weight
        self.dice_w = dice_weight
        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=cw)
        self.dice = DiceLoss(num_classes, ignore_bg=dice_ignore_bg)

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        total = self.ce_w * ce_loss + self.dice_w * dice_loss

        log = {"ce": ce_loss.detach(), "dice": dice_loss.detach()}
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            for c in range(self.num_classes):
                p = (preds == c).float()
                t = (target == c).float()
                inter = (p * t).sum()
                card = p.sum() + t.sum()
                log[f"dice_c{c}"] = (2 * inter + 1.0) / (card + 1.0)

        return total, log