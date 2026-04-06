import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, weight=None, num_class=3, dice_weight=1.0, ce_weight=1.0, smooth=1e-6):
        super(DiceCELoss, self).__init__()
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_class = num_class

    def forward(self, pred, target):
        # pred: [B, C, H, W] (logits)
        # gt: [B, H, W]      (class indices)
        
        ce = self.ce_loss(pred, target)

        """
        pred: [B, C, H, W] logits
        target: [B, H, W] integer labels (0, 1, 2, ...)
        """
        num_classes = pred.shape[1]
        pred_softmax = F.softmax(pred, dim=1)  # [B, C, H, W]

        # One-hot encode the target
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()   # [B, C, H, W]

        dice = 0.0
        for c in range(num_classes):
            p = pred_softmax[:, c]
            t = target_one_hot[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice += 1 - ((2 * intersection + self.smooth) / (union + self.smooth))

        dice_loss = dice / num_classes

        # ---- Total Loss ----
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce

        return total_loss
