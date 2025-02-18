import os
from typing import Any
from tqdm import tqdm

from avalanche.core import Template
from avalanche.training.plugins import SupervisedPlugin


class ValidationStreamPlugin(SupervisedPlugin):

    def __init__(self, val_stream: Any, debug_log_file: str = None):
        super().__init__()
        self.training_exp_id = 0
        self.val_stream = None
        self.val_experiences = None
        if debug_log_file is not None:
            self.debug_log_file = open(debug_log_file, 'w')
        else:
            self.debug_log_file = os.devnull
        self.set_validation_stream(val_stream)
        self.last_results = None

    def set_validation_stream(self, val_stream: Any):
        self.val_stream = val_stream
        self.val_experiences = [val_exp for val_exp in val_stream]
        for i in range(len(self.val_experiences)):
            for j in range(i):
                assert self.val_experiences[i] != self.val_experiences[j]

    #def before_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
    #    self.training_exp_id = strategy.experience.current_experience
    
    def after_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        self.last_results = None

    def after_training_epoch(self, strategy: Template, *args, **kwargs) -> Any:
        self.training_exp_id = strategy.experience.current_experience
        val_exp = self.val_experiences[self.training_exp_id]
        last_results = strategy.eval(val_exp)
        self.last_results = {}
        for key, value in last_results.items():
            key = key.replace('_Exp', '_Epoch')
            if '_Stream' in key:
                continue
            if ('eval_phase' in key) and (f'Exp00{self.training_exp_id}' in key):
                self.last_results[key] = value
        if self.debug_log_file is not None:
            self.debug_log_file.write(f"[{type(self).__name__}] [{self.training_exp_id}]: {self.last_results}\n")
        return self.last_results


class TqdmTrainingEpochsPlugin(SupervisedPlugin):

    def __init__(self, num_exp: int, num_epochs: int):
        super().__init__()
        self.num_exp = num_exp
        self.num_epochs = num_epochs
        self.tqdm = None

    def before_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.tqdm = tqdm(total=self.num_epochs, desc=f'Training exp {self.num_exp}')

    def after_training_epoch(self, strategy: Template, *args, **kwargs) -> Any:
        self.tqdm.update(1)

    def after_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.tqdm.close()


__all__ = [
    'ValidationStreamPlugin',
    'TqdmTrainingEpochsPlugin'
]