import torch.nn as nn

from avalanche.models.base_model import BaseModel


class CompleteModel(BaseModel):

    def get_features(self, x):
        return self.regressor.get_features(x)

    def __init__(
        self, *, classifier: nn.Module = None, regressor: nn.Module = None,
        classifier_class: type = None, regressor_class: type = None,
        classifier_args: dict = None, regressor_args: dict = None
    ):
        if classifier is None:
            self.classifier = classifier_class(**classifier_args)
        else:
            self.classifier = classifier
        if regressor is None:
            self.regressor = regressor_class(**regressor_args)
        else:
            self.regressor = regressor
        self.dtype = self.regressor.dtype

    def forward(self, x):
        classifier_result = self.classifier.classify(x, self.dtype)
        regressor_result = self.regressor(x)
        return classifier_result * regressor_result


__all__ = ['CompleteModel']
