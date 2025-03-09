from typing import Literal

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss.

    This loss assumes the predicted outputs are the mean (mu) and
    the std (sigma) of a Gaussian distribution. The target
    is assumed to follow the predicted Gaussian distribution.
    """

    def __init__(
            self,
            reduction: Literal["mean", "sum", "none"] = 'mean',
            constant: float = np.log(2 * np.pi), nll_lambda: float = 0.0,
    ):
        """
        :param reduction: Reduction type to apply. Either "mean", "sum" or "none".
        :param constant: Constant term to add to the loss.
        :param nll_lambda: Weight of the term that penalizes sigma magnitude.
        """
        super(GaussianNLLLoss, self).__init__()
        self.reduction = reduction
        self.constant = constant
        self.nll_lambda = nll_lambda
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def forward(self, outputs, targets):
        num_features = outputs.shape[1] // 2
        # Split predictions into mean and variance
        mean = outputs[:, :num_features]
        variance = outputs[:, num_features:] ** 2
        nll = 0.5 * (torch.log(variance) + (targets - mean) ** 2 / variance + self.constant)
        #nll = 0.5 * (log_variance + (targets - mean) ** 2 / torch.exp(log_variance) + self.constant)
        nll += self.nll_lambda * variance
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class MSECosineSimilarityLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, reduction='mean'):
        """
        Custom loss combining MSE and Cosine Similarity.
        :param alpha: Weight for the MSE term.
        :param beta: Weight for the Cosine Similarity term.
        :param reduction: Specifies the reduction to apply to the MSE term: 'none',
        'mean', or 'sum'.
        """
        super(MSECosineSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, outputs, targets):
        mse = self.mse_loss(outputs, targets)

        # Compute Cosine Similarity (negative because we minimize the loss)
        cosine_similarity = F.cosine_similarity(outputs, targets, dim=1)
        cosine_loss = 1 - cosine_similarity.mean()

        # Combine the losses
        combined_loss = self.alpha * mse + self.beta * cosine_loss
        return combined_loss


class RootMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Root Mean Square Error Loss.
        :param reduction: Specifies the reduction to apply: 'none', 'mean', or 'sum'.
        """
        super(RootMSELoss, self).__init__()
        self.reduction = reduction
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def forward(self, outputs, targets):
        mse = (outputs - targets) ** 2
        if self.reduction == 'mean':
            return torch.sqrt(mse.mean())
        elif self.reduction == 'sum':
            return torch.sqrt(mse.sum())
        elif self.reduction == 'none':
            return torch.sqrt(mse)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Weighted Mean Square Error Loss.
        :param reduction: Specifies the reduction to apply: 'none', 'mean', or 'sum'.
        """
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def forward(self, outputs, targets, weights=None):
        """
        Forward pass of weighted MSE loss.
        :param outputs: Model predictions
        :param targets: Ground truth values
        :param weights: Optional tensor of weights per sample. Must be same length as outputs.
        :return: Weighted MSE loss
        """
        return F.mse_loss(outputs, targets, reduction='none', weight=weights)



__all__ = ["GaussianNLLLoss", "MSECosineSimilarityLoss", "RootMSELoss", "WeightedMSELoss"]