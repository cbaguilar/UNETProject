import numpy as np
import torch
from torch import nn

def onehot(X, num_classes):
    shape = list(X.shape)
    shape[1] = num_classes
    result = torch.zeros(shape)
    result = result.scatter_(1, X.cpu(), 1)
    return result


class DiceLoss(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def forward(self, y_pred, y_target):
        y_pred = y_pred.contiguous().view(y_pred.shape[0], -1)
        y_target = y_target.contiguous().view(y_target.shape[0], -1)

        intersection = 2 * torch.sum(y_target * y_pred, dim=1)
        union = torch.sum(y_target**self.p + y_pred**self.p, dim=1) 
        
        loss = 1 - intersection / (union + 0.00001)
        
        return loss.mean()

        