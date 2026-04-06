import torch
from torch.nn import Module, BCELoss, BCEWithLogitsLoss


class DiceLoss(Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()

        self.smooth = smooth

    def forward(self, pred, gt):

        # Flatten
        pred = pred.contiguous().view(-1)
        gt = gt.contiguous().view(-1)

        intersection = (pred*gt).sum()

        dice = (2. * intersection + self.smooth) / (pred.sum() + gt.sum() + self.smooth)

        return 1-dice
    
class BCEDiceLoss(Module):
    def __init__(self, weight_BCE, weight_Dice):
        super(BCEDiceLoss, self).__init__()

        self.bce_loss = BCELoss()
        self.dice_loss = DiceLoss()

        self.weight_BCE = weight_BCE
        self.weight_Dice = weight_Dice

    def forward(self, pred, gt):

        bce = self.bce_loss(pred,gt)
        dice = self.dice_loss(pred,gt)

        return self.weight_BCE* bce + self.weight_Dice*dice
