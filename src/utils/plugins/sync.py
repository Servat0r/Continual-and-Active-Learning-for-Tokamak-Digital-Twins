import os
import torch.cuda as cuda
from typing import Any
from tqdm import tqdm

from avalanche.core import Template
from avalanche.training.plugins import SupervisedPlugin


class CUDASynchronizationPlugin(SupervisedPlugin):

    
    def __init__(self, after_train_epoch: bool = True, after_train_exp: bool = True, after_eval_exp: bool = True):
        super().__init__()
        self.__after_train_epoch = after_train_epoch
        self.__after_train_exp = after_train_exp
        self.__after_eval_exp = after_eval_exp

    def __conditional_sync(self):
        if cuda.is_available():
            cuda.synchronize()
    
    def after_training_epoch(self, strategy: Any, *args, **kwargs) -> Any:
        if self.__after_train_epoch:
            self.__conditional_sync()
            return super().after_training_epoch(strategy, *args, **kwargs)
    
    def after_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        if self.__after_train_exp:
            self.__conditional_sync()
            return super().after_training_exp(strategy, *args, **kwargs)
    
    def after_eval_exp(self, strategy: Any, *args, **kwargs) -> Any:
        if self.__after_eval_exp:
            self.__conditional_sync()
            return super().after_eval_exp(strategy, *args, **kwargs)


__all__ = ['CUDASynchronizationPlugin']
