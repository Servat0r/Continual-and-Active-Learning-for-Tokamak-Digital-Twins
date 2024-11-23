import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss.

    This loss assumes the predicted outputs are the mean (mu) and
    log variance (log_var) of a Gaussian distribution. The target
    is assumed to follow the predicted Gaussian distribution.
    """

    def __init__(self, reduction='mean'):
        super(GaussianNLLLoss, self).__init__()
        self.reduction = reduction
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def forward(self, outputs, y):
        num_features = outputs.shape[1] // 2
        # Split predictions into mean and log_variance
        mean = outputs[:, :num_features]
        #log_variance = outputs[:, num_features:]
        variance = outputs[:, num_features:]
        #variance = torch.exp(log_variance)
        nll = 0.5 * (torch.log(2 * torch.pi * variance) + (y - mean) ** 2 / variance)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


__all__ = ['GaussianNLLLoss']