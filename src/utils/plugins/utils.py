from typing import Any
from tqdm import tqdm

from avalanche.core import Template
from avalanche.training.plugins import SupervisedPlugin


class ValidationStreamPlugin(SupervisedPlugin):

    def __init__(self, val_stream: Any):
        super().__init__()
        self.training_exp_id = None
        self.val_stream = None
        self.val_experiences = None
        self.set_validation_stream(val_stream)
        self.last_results = None

    def set_validation_stream(self, val_stream: Any):
        self.val_stream = val_stream
        self.val_experiences = []
        for val_exp in val_stream:
            self.val_experiences.append(val_exp)

    def before_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.training_exp_id = strategy.experience.current_experience

    def after_training_epoch(self, strategy: Template, *args, **kwargs) -> Any:
        val_exp = self.val_experiences[self.training_exp_id]
        self.last_results = strategy.eval(val_exp)
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