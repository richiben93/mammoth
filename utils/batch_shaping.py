import torch
from torch import nn
from scipy.stats import beta as betadist
from torch.autograd import Function


class BatchShapingLossModuleOld(nn.Module):

    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        super(BatchShapingLoss, self).__init__()
        # self.distro = torch.distributions.beta.Beta(torch.Tensor([alpha]), torch.Tensor([beta]))
        self.dist = betadist(alpha, beta)
        self.cache = {}

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            throw_cols = mask.sum(0) < len(mask)
            x = x.clone()[:, ~throw_cols]

        # sort the data
        sorted_x, _ = torch.sort(x, 0)
        n = x.shape[0]
        loss = 0
        for i in range(x.shape[1]):
            ecdf_i = (torch.arange(1, n + 1).to(x[:, i].device)) / (n + 1)
            # pcdf_i = self.beta_cdf(sorted_x[:, i]).to(ecdf_i.device)
            pcdf_i = torch.from_numpy(self.dist.cdf(sorted_x[:, i].cpu().detach())).to(ecdf_i.device).requires_grad_()
            loss += (pcdf_i - ecdf_i).pow(2).sum(0)
        return loss/n

    def backward(self, x: torch.Tensor):
        k = 1
        pass

    def bs_test(self, x: torch.Tensor):
        pass

    def beta_cdf(self, b: torch.Tensor, npts: int = 1000):
        x = mylinspace(torch.zeros(b.shape[0]).to(b.device)+1e-10, b, npts)
        start = 0
        if self.distro.log_prob(x[0, 0].to('cpu')).exp().item() > 1:
            start = 1
        return torch.concat([torch.trapz(self.distro.log_prob(x[start:, i].to('cpu')).exp()[None, ], x[start:, i].to('cpu')).to(x.device)
                             for i in range(x.shape[1])])


def mylinspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out


class BatchShapingFunction(Function):
    @staticmethod
    def forward(ctx, y, alpha, beta):
        n, h = y.shape[0], y.shape[1]
        sorted, indices = torch.sort(y, 0)
        sorted_values = sorted.detach().cpu().numpy()
        cdf = betadist.cdf(sorted_values, alpha, beta)
        pdf = betadist.pdf(sorted_values, alpha, beta)
        cdf = torch.from_numpy(cdf).type(
            torch.float32).to(y.device)
        pdf = torch.from_numpy(pdf).type(
            torch.float32).to(y.device)
        pdf = torch.nan_to_num(pdf, posinf=0)
        edf = (torch.arange(1, n + 1, dtype=torch.float32).to(
            y.device)).unsqueeze(1) / (n + 1)
        ctx.save_for_backward(indices, pdf, cdf, edf)
        return (cdf - edf).pow(2).sum(0).mean()

    @staticmethod
    def backward(ctx, grad_output):
        indices, pdf, cdf, edf = ctx.saved_tensors
        h = cdf.shape[1]
        grad = -2.0 * pdf * (edf - cdf)
        _, negindices = torch.sort(indices, 0)
        grad = torch.stack([grad[negindices[:, i], i]
                            for i in range(h)], 1)
        return grad * grad_output * (1. / h), None, None


class BatchShapingLoss:
    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input):
        return BatchShapingFunction.apply(input, self.alpha, self.beta)