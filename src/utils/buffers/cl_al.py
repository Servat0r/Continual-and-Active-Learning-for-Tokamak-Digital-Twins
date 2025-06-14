"""
Replay Buffers variations to allow AL-based selection strategies.
"""
from typing import Any
import torch

from avalanche.benchmarks import AvalancheDataset
from avalanche.training import ExemplarsBuffer, BalancedExemplarsBuffer
from bmdal_reg.bmdal.feature_data import TensorFeatureData

from ..active_learning import ALBatchSelector
from ..datasets import CSVRegressionDataset
from ..misc import debug_print, STDOUT


class ActiveLearningSamplingBuffer(ExemplarsBuffer):

    def __init__(self, max_size: int, batch_selector: ALBatchSelector, device='cpu'):
        """
        :param max_size:
        :param batch_selector: Batch Selector object to perform Active Learning - based
        Replay-Buffer items selection.
        """
        super().__init__(max_size)
        self.batch_selector = batch_selector
        self.device = device

    def post_adapt(self, agent, exp):
        self.update_from_dataset(exp.dataset)

    def update_from_dataset(self, new_data: AvalancheDataset, exp_index: int = -1):
        """
        Update the buffer using the given dataset.
        :param new_data:
        :return:
        """
        csv_regression_dataset = new_data._datasets[0]
        X_pool, y_pool = csv_regression_dataset.inputs, csv_regression_dataset.targets
        X_pool_transformed = csv_regression_dataset.transform(X_pool)
        pool_data = TensorFeatureData(X_pool_transformed.to(self.device))
        sampled_idxs = self.batch_selector(pool_data, csv_regression_dataset)
        sampled_idxs = sampled_idxs[:self.max_size]
        X_sampled, y_sampled = X_pool.to(self.device)[sampled_idxs], y_pool.to(self.device)[sampled_idxs]
        new_csv_regression_dataset = CSVRegressionDataset(
            data=None, input_columns=[], output_columns=[],
            inputs=X_sampled, outputs=y_sampled,
            transform=csv_regression_dataset.transform,
            target_transform=csv_regression_dataset.target_transform,
        )
        new_csv_regression_dataset.set_device(self.device)
        self.buffer = AvalancheDataset([new_csv_regression_dataset])
        # Now add the newly selected buffer to Batch Selector Memory
        self.batch_selector.set_new_train_exp(self.buffer, index=exp_index)

    def resize(self, strategy: Any, new_size: int):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = self.buffer.subset(torch.arange(self.max_size))


class ExperienceBalancedActiveLearningBuffer(BalancedExemplarsBuffer[ActiveLearningSamplingBuffer]):
    """
    A variation of avalanche ExperienceBalancedBuffer that uses an ActiveLearningSamplingBuffer
    to maintain every internal buffer, instead of random-selection-based one (ReservoirSamplingBuffer).
    """

    def __init__(
            self, max_size: int,
            adaptive_size: bool = True,
            num_experiences=None,
            batch_selector: ALBatchSelector = None,
            device='cpu',
    ):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        :param batch_selector: Batch Selector object to perform Active Learning - based
        Replay-Buffer items selection.
        """
        super().__init__(max_size, adaptive_size, num_experiences)
        self._num_exps = 0
        self.batch_selector = batch_selector
        self.device = device

    def post_adapt(self, agent, exp):
        self._num_exps += 1
        new_data = exp.dataset
        lens = self.get_group_lengths(self._num_exps)

        new_buffer = ActiveLearningSamplingBuffer(lens[-1], self.batch_selector, device=self.device)
        new_buffer.update_from_dataset(new_data)
        self.buffer_groups[self._num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(agent, ll)


__all__ = [
    'ActiveLearningSamplingBuffer',
    'ExperienceBalancedActiveLearningBuffer',
]
