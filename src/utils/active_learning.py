"""
Collection of classes and methods to handle different Active Learning strategies and frameworks.
This allows to directly call Active Learning selection routines inside the task_training_loop code.
"""
from abc import ABC, abstractmethod
import torch.nn

from avalanche.benchmarks import AvalancheDataset
from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch


class ALBatchSelector(ABC):

    def __init__(self):
        self.device = None
        self.models = None
        self.train_exp = None

    @abstractmethod
    def __call__(
        self, pool_data: TensorFeatureData, pool_dataset: AvalancheDataset,
            train_data: TensorFeatureData = None, y_train: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def set_models(self, models: list[torch.nn.Module]) -> None:
        self.models = models

    def set_device(self, device: str):
        self.device = torch.device(device)

    def set_train_exp(self, train_exp):
        self.train_exp = train_exp.dataset._datasets[0] # CSVRegressionDataset

    def add_train_exp(self, train_exp):
        if self.train_exp is None:
            self.set_train_exp(train_exp)
        else:
            new_csv_dataset = train_exp.dataset._datasets[0]
            new_inputs = torch.concat([self.train_exp.inputs, new_csv_dataset.inputs])
            new_targets = torch.concat([self.train_exp.targets, new_csv_dataset.targets])
            self.train_exp.inputs = new_inputs
            self.train_exp.targets = new_targets
            print(f"New batch X_train dataset has {len(self.train_exp)} rows")


class BMDALBatchSelector(ALBatchSelector):
    """
    Batch Selector for BMDAL algorithms.
    If attribute "models" is not initialized at the beginning, it must be set with
    self.set_models() BEFORE any usage of self.__call__(), otherwise a RuntimeError
    would be raised.
    """
    def __init__(
        self, models: list[torch.nn.Module] = None, batch_size: int = 100,
        selection_method: str = 'lcmd', sel_with_train: bool = False,
        base_kernel: str = 'grad', kernel_transforms: list = None
    ):
        super().__init__()
        self.models = models
        self.batch_size = batch_size
        self.selection_method = selection_method
        self.sel_with_train = sel_with_train
        # TODO Verify if this assumption can actually be made (it simplifies src.utils.buffers objects).
        self.base_kernel = base_kernel
        self.kernel_transforms = kernel_transforms

    def __call__(
        self, pool_data: TensorFeatureData, pool_dataset: AvalancheDataset,
        train_data: TensorFeatureData = None, y_train: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.models is None:
            raise RuntimeError(f"Attribute \"models\" of {type(self).__name__} object is None")
        if train_data is None:
            X_train, y_train = self.train_exp[:]
            train_data = TensorFeatureData(X_train.to(self.device))
            print(
                type(self).__name__,
                X_train.device,
                pool_dataset._datasets[0][:][0].device
            )
        new_idxs, _ = select_batch(
            batch_size=self.batch_size, models=[m.to('cpu') for m in self.models], y_train=y_train,
            data={'train': train_data, 'pool': pool_data},
            selection_method=self.selection_method,
            sel_with_train=self.sel_with_train,
            base_kernel=self.base_kernel,
            kernel_transforms=self.kernel_transforms
        )
        return new_idxs


class MCDropoutBatchSelector(ALBatchSelector):
    """
    Monte Carlo Dropout Batch Selector.
    """
    def __init__(self):
        pass


class DeepEnsembleBatchSelector(ALBatchSelector):
    """
    Deep Ensemble Batch Selector.
    """
    def __init__(self):
        pass


__all__ = [
    'ALBatchSelector',
    'BMDALBatchSelector',
    'MCDropoutBatchSelector',
    'DeepEnsembleBatchSelector',
]
