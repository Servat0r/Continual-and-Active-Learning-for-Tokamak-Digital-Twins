"""
Collection of classes and methods to handle different Active Learning strategies and frameworks.
This allows to directly call Active Learning selection routines inside the task_training_loop code.
"""
from abc import ABC, abstractmethod
from typing import Literal

import torch.nn

from avalanche.benchmarks import AvalancheDataset, DatasetExperience
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

    def set_train_exp(self, train_exp: DatasetExperience | AvalancheDataset):
        if isinstance(train_exp, DatasetExperience):
            self.train_exp = train_exp.dataset._datasets[0] # CSVRegressionDataset
        elif isinstance(train_exp, AvalancheDataset):
            self.train_exp = train_exp._datasets[0]

    def add_train_exp(self, train_exp: DatasetExperience | AvalancheDataset):
        if self.train_exp is None:
            self.set_train_exp(train_exp)
        else:
            if isinstance(train_exp, DatasetExperience):
                new_csv_dataset = train_exp.dataset._datasets[0]
            elif isinstance(train_exp, AvalancheDataset):
                new_csv_dataset = train_exp._datasets[0]
            else:
                raise ValueError(
                    f"train_exp must be DatasetExperience or AvalancheDataset, got {type(train_exp)}"
                )
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
        base_kernel: str = 'grad', kernel_transforms: list = None,
        initial_selection_method: Literal["random", "maxdiag"] = "random"
    ):
        super().__init__()
        self.models = models
        self.batch_size = batch_size
        self.selection_method = selection_method
        self.sel_with_train = sel_with_train
        # TODO Verify if this assumption can actually be made (it simplifies src.utils.buffers objects).
        self.base_kernel = base_kernel
        self.kernel_transforms = kernel_transforms
        self.initial_selection_method = initial_selection_method

    def __call__(
        self, pool_data: TensorFeatureData, pool_dataset: AvalancheDataset,
        train_data: TensorFeatureData = None, y_train: torch.Tensor = None,
    ) -> torch.Tensor:
        selection_method = self.selection_method
        if self.models is None:
            raise RuntimeError(f"Attribute \"models\" of {type(self).__name__} object is None")
        if train_data is None:
            if self.train_exp is not None:
                X_train, y_train = self.train_exp[:]
            else:
                # Determine the feature size from pool_data
                shape, dtype = list(pool_data.data.shape), pool_data.data.dtype
                shape[0] = 1 # One-element tensor (was empty before but this caused a index-out-of-bounds error in batch_select)
                X_train = torch.zeros(shape, dtype=dtype).to(self.device)
                selection_method = self.initial_selection_method
            train_data = TensorFeatureData(X_train.to(self.device))
        new_idxs, _ = select_batch(
            batch_size=self.batch_size, models=[m.to(self.device) for m in self.models], y_train=y_train,
            data={'train': train_data, 'pool': pool_data},
            selection_method=selection_method,
            sel_with_train=self.sel_with_train,
            base_kernel=self.base_kernel,
            kernel_transforms=self.kernel_transforms
        )
        print(f"Selected {len(new_idxs)} samples from pool")
        return new_idxs


class MCDropoutBatchSelector(ALBatchSelector):
    """
    Monte Carlo Dropout Batch Selector.
    Selects samples based on the predictive variance across multiple forward passes
    with dropout enabled.
    """
    def __init__(
        self,
        model: torch.nn.Module = None,
        batch_size: int = 100,
        n_passes: int = 10,
        random_tie_break: bool = True
    ):
        """
        :param model: The model on which to perform MC Dropout
        :param batch_size: Number of samples to select
        :param n_passes: Number of stochastic forward passes
        :param random_tie_break: Whether to shuffle indices with near-equal variance
        """
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.n_passes = n_passes
        self.random_tie_break = random_tie_break

    def __call__(
        self,
        pool_data: TensorFeatureData,
        pool_dataset: AvalancheDataset,
        train_data: TensorFeatureData = None,
        y_train: torch.Tensor = None
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError(f"Attribute 'model' of {type(self).__name__} is None")

        # Device synchronization
        if self.device is not None:
            self.model.to(self.device)
        pool_x = pool_data.data
        if self.device is not None:
            pool_data.data = pool_x.to(self.device)
        
        if train_data is None:
            if self.train_exp is not None:
                X_train, y_train = self.train_exp[:]
            else:
                shape, dtype = list(pool_data.data.shape), pool_data.data.dtype
                shape[0] = 0  # no samples
                X_train = torch.zeros(shape, dtype=dtype)

        all_preds = []
        # For having dropout active but without gradient calculations,
        # set model in train() mode and use torch.no_grad()
        self.model.train()
        with torch.no_grad():
            for _ in range(self.n_passes):
                preds = self.model(pool_x)
                # Store with an extra dimension for pass
                all_preds.append(preds.unsqueeze(0))

        # Stack all predictions along axis=0 => shape: (n_passes, N, something)
        all_preds = torch.cat(all_preds, dim=0)

        # all_preds has shape: (n_passes, N, output_size)
        # 1) Compute variance across the n_passes dimension => shape: (N, output_size)
        # 2) Then reduce from (N, output_size) to (N,) by averaging over output_size
        variances = all_preds.var(dim=0).mean(dim=-1)

        # 3) Now pick the top batch_size samples by highest variance
        #    topk returns (values, indices)
        _, topk_idxs = variances.topk(self.batch_size, largest=True)
        # If necessary, shuffle the indices to break ties
        if self.random_tie_break:
            topk_idxs = topk_idxs[torch.randperm(len(topk_idxs))]
        return topk_idxs


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
