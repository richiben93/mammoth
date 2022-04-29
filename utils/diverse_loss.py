import torch
from math import log
from torch import nn


class DiverseLoss(nn.Module):
    def __init__(self, tau: float = 2.0):
        super(DiverseLoss, self).__init__()
        self.tau = tau

    def forward(self, logits):
        mean = torch.mean(logits, dim=1, keepdim=True)
        std = torch.std(logits, dim=1, keepdim=True)
        logits = (logits-mean) / std
        dotlogits = torch.matmul(logits, logits.t()) / self.tau
        loss = torch.logsumexp(dotlogits, dim=1).mean(0) - \
               1 / self.tau - log(int(logits.shape[0]) * 1.)
        return loss