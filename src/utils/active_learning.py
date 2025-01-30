"""
Collection of classes and methods to handle different Active Learning strategies and frameworks.
This allows to directly call Active Learning selection routines inside the task_training_loop code.
"""
from abc import ABC, abstractmethod
from typing import Literal
from sortedcontainers import SortedDict

import torch.nn

from avalanche.benchmarks import AvalancheDataset, DatasetExperience
from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch

from .misc import debug_print, STDOUT


class ALBatchSelector(ABC):

    def __init__(self):
        self.device = None
        self.models = None
        self.train_experiences = SortedDict()
        self.train_inputs = None
        self.train_targets = None

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

    def add_train_exp(self, train_exp: DatasetExperience | AvalancheDataset, index: int = -1):
        index = index if index >= 0 else len(self.train_experiences)
        if index in self.train_experiences.keys():
            debug_print(f"[red]Experience {index} alredy present in Batch Selector[/red]", file=STDOUT)
            return
        if isinstance(train_exp, DatasetExperience):
            self.train_experiences[index] = train_exp.dataset
        elif isinstance(train_exp, AvalancheDataset):
            self.train_experiences[index] = train_exp
        else:
            raise TypeError(f"Incorrect type {train_exp.__class__.__name__} for \"train_exp\"")


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
        initial_selection_method: Literal["random", "maxdiag"] = "random",
        debug_log_file: str = 'batch_selector.log',
    ):
        super().__init__()
        self.models = models
        self.batch_size = batch_size
        self.selection_method = selection_method
        self.sel_with_train = sel_with_train
        self.base_kernel = base_kernel
        self.kernel_transforms = kernel_transforms
        self.initial_selection_method = initial_selection_method
        self.debug_log_file = open(debug_log_file, 'w')
    
    def close(self):
        self.debug_log_file.close()
    
    def __call__(
        self, pool_data: TensorFeatureData, pool_dataset: AvalancheDataset,
        train_data: TensorFeatureData = None, y_train: torch.Tensor = None,
    ) -> torch.Tensor:
        selection_method = self.selection_method
        if self.models is None:
            raise RuntimeError(f"Attribute \"models\" of {type(self).__name__} object is None")
        if train_data is None:
            #if self.train_inputs is not None:
            if len(self.train_experiences) > 0:
                # Dynamic building of training data (hence avoiding GPU memory occupation during training)
                train_inputs = [
                    train_exp._datasets[0].inputs for key, train_exp in self.train_experiences.items()
                ]
                X_train = torch.concat(train_inputs, dim=0)
                y_train = None
                #X_train, y_train = self.train_inputs, self.train_targets
            else:
                # Determine the feature size from pool_data
                shape, dtype = list(pool_data.data.shape), pool_data.data.dtype
                shape[0] = 1 # One-element tensor (was empty before but this caused a index-out-of-bounds error in batch_select)
                X_train = torch.zeros(shape, dtype=dtype)
                selection_method = self.initial_selection_method
            train_data = TensorFeatureData(X_train)
        debug_print(f"[red]Using selection method '{selection_method}'[/red]", file=STDOUT)
        # Synchronize all of them on the same device
        orig_device = pool_data.data.device
        model_device = next(self.models[0].parameters()).device
        train_data = TensorFeatureData(train_data.data.to(model_device))
        pool_data = TensorFeatureData(pool_data.data.to(model_device))
        y_train2 = y_train.to(model_device) if y_train is not None else y_train
        is_training = self.models[0].training
        self.models[0].eval()
        debug_print(torch.cuda.memory_summary(device='cuda'), file=self.debug_log_file)
        new_idxs, _ = select_batch(
            batch_size=self.batch_size, models=self.models, y_train=y_train2,
            data={'train': train_data, 'pool': pool_data},
            selection_method=selection_method,
            sel_with_train=self.sel_with_train,
            base_kernel=self.base_kernel,
            kernel_transforms=self.kernel_transforms
        )
        if is_training:
            self.models[0].train()
        print(f"Selected {len(new_idxs)} samples from pool")
        return new_idxs.to(orig_device)


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
            if self.train_inputs is not None:
                X_train, y_train = self.train_inputs, self.train_targets
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
