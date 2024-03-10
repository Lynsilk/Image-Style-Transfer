import torch
from torch.nn import functional as f


class GanLoss(torch.nn.Module):
    def __init__(self):
        super(GanLoss, self).__init__()

    def forward(self, pred:torch.Tensor, label:torch.Tensor):
        if pred.shape != label.shape:
            label = label.expand_as(pred)
        loss = f.mse_loss(pred, label)
        return loss


class CycleLoss(torch.nn.Module):
    def __init__(self):
        super(CycleLoss, self).__init__()

    def forward(self, real:torch.Tensor, fake:torch.Tensor):
        loss = f.l1_loss(real, fake)
        return loss


class IdentityLoss(torch.nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()

    def forward(self, real:torch.Tensor, fake:torch.Tensor):
        loss = f.l1_loss(real, fake)
        return loss