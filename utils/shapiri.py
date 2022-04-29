import torch
from torch import nn

class MaskedShapiroShapingGaussianLoss(nn.Module):

    def __init__(self):
        super(MaskedShapiroShapingGaussianLoss, self).__init__()
        self.distro = torch.distributions.Normal(0, 1)
        self.cache = {}

    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            throw_cols = mask.sum(0) < len(mask)
            x = x.clone()[:, ~throw_cols]

        # sort the data
        sorted_x, _ = torch.sort(x, 0)
        n = x.shape[0]
        if n not in self.cache:
            # blom scores
            m = self.distro.icdf(
                (torch.arange(1, n+1).to(x.device) - 3/8) / 
                (n + 1/4)
            )
            # weights
            self.cache[n] = (m / m.pow(2).sum().pow(0.5)).unsqueeze(1)
        
        c = self.cache[n]
              
        Wp = (c * sorted_x).sum(0).pow(2) / (sorted_x - sorted_x.mean(0)).pow(2).sum(0)
        return -Wp.mean()


    def shapiro_test(self, x: torch.Tensor):
        # sort the data
        sorted_x, _ = torch.sort(x, 0)
        n = x.shape[0]
        if n not in self.cache:
            # blom scores
            m = self.distro.icdf(
                (torch.arange(1, n+1).to(x.device) - 3/8) / 
                (n + 1/4)
            )
            # weights
            self.cache[n] = (m / m.pow(2).sum().pow(0.5)).unsqueeze(1)
        
        c = self.cache[n]
              
        Wp = (c * sorted_x).sum(0).pow(2) / (sorted_x - sorted_x.mean(0)).pow(2).sum(0)
        return Wp