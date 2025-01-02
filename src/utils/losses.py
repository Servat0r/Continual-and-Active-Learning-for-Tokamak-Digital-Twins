import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, outputs, targets):
        num_features = outputs.shape[1] // 2
        # Split predictions into mean and log_variance
        mean = outputs[:, :num_features]
        #log_variance = outputs[:, num_features:]
        variance = outputs[:, num_features:]
        #variance = torch.exp(log_variance)
        nll = 0.5 * (torch.log(2 * torch.pi * variance) + (targets - mean) ** 2 / variance)
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


__all__ = ["GaussianNLLLoss", "MSECosineSimilarityLoss"]