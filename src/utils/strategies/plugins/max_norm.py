# TODO: This stuff is JUST A DRAFT!
import copy
from typing import TYPE_CHECKING, Optional
import torch
from avalanche.benchmarks.utils import concat_datasets
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ExperienceBalancedBuffer

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def update_temp(model, grad, lr):
    model_copy = copy.deepcopy(model)
    for g, p in zip(grad, model_copy.parameters()):
        if g is not None:
            p.data = p.data - lr * g
    return model_copy


class MaxNormPlugin(ReplayPlugin):
    """
    Plugin that selects examples to keep based on maximum Frobenius norm of Jacobian.
    """
    def __init__(
        self,
        mem_size: int = 200,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        score_batch_size: int = 256
    ):
        super().__init__(
            mem_size=mem_size,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=task_balanced_dataloader,
            storage_policy=storage_policy
        )
        self.score_batch_size = score_batch_size

    def _compute_jacobian_norm(self, strategy, x, y, tid):
        """Compute Frobenius norm of Jacobian matrix for batch of samples"""
        strategy.model.zero_grad()
        output = avalanche_forward(strategy.model, x, tid)
        loss = strategy._criterion(output, y)
        loss.backward()
        
        # Get gradients from parameters
        grads = []
        for p in strategy.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        
        # Concatenate all gradients and compute norm
        grad_tensor = torch.cat(grads)
        grad_norm = torch.norm(grad_tensor, p='fro')
        
        # Clear gradients for next iteration
        #strategy.model.zero_grad()
        
        return grad_norm

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        # Get all samples from current experience
        dataset = strategy.experience.dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.score_batch_size,  # Process all samples at once
            shuffle=False
        )

        iter_dataloader = iter(dataloader)        
        # Get single batch containing all samples
        x, y, tid = next(iter(dataloader))
        x = x.to(strategy.device)
        y = y.to(strategy.device)
        tid = tid.to(strategy.device)

        # Compute norm for all samples in batch
        norm = self._compute_jacobian_norm(strategy, x, y, tid)
        # Zero all the gradients at the end
        strategy.model.zero_grad()
        
        # Update storage policy with samples having highest norm
        self.storage_policy.update(strategy, **kwargs)


__all__ = ['MaxNormPlugin']


