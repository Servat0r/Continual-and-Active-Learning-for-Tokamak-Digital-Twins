import copy

import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import get_last_fc_layer, freeze_everything
from avalanche.models.base_model import BaseModel

from ...logging import CustomCSVLogger
from ...misc import stdout_debug_print


class DoubleLFLPlugin(SupervisedPlugin):
    """
    A variant of Avalanche LFLPlugin.
    """

    def __init__(self, lambda_e: int | float, evaluation_metric: str = 'R2Score_Exp'):
        """
        :param lambda_e: Euclidean loss hyper parameter
        """
        super().__init__()

        self.lambda_e = lambda_e
        self.prev_model = None
        self.prev_metric_values = None
        self._eval_stream = None
        self.evaluation_metric = evaluation_metric

    def set_eval_stream(self, eval_stream):
        self._eval_stream = eval_stream
    
    def _euclidean_loss(self, features, prev_features):
        """
        Compute euclidean loss
        """
        return torch.nn.functional.mse_loss(features, prev_features)

    def penalty(self, x, model, lambda_e):
        """
        Compute weighted euclidean loss
        """
        if self.prev_model is None:
            return 0
        else:
            features, prev_features = self.compute_features(model, x)
            dist_loss = self._euclidean_loss(features, prev_features)
            return lambda_e * dist_loss

    def compute_features(self, model, x):
        """
        Compute features from prev model and current model
        """
        prev_model = self.prev_model
        assert prev_model is not None
        model.eval()
        prev_model.eval()

        features = model.get_features(x)
        prev_features = prev_model.get_features(x)

        return features, prev_features

    def before_backward(self, strategy, **kwargs):
        """
        Add euclidean loss between prev and current features as penalty
        """
        lambda_e = (
            self.lambda_e[strategy.clock.train_exp_counter]
            if isinstance(self.lambda_e, (list, tuple))
            else self.lambda_e
        )

        penalty = self.penalty(strategy.mb_x, strategy.model, lambda_e)
        strategy.loss += penalty

    def __compute_current_metrics(self, strategy, current_experience, model=None):
        current_metrics = []
        # Temporarily suspend the logger to avoid writing two times the evaluation results
        if model is not None:
            old_model = strategy.model
            strategy.model = model
        for logger in strategy.evaluator.loggers:
            if isinstance(logger, CustomCSVLogger):
                logger.suspend()
        for index, eval_exp in enumerate(self._eval_stream):
            #if index > current_experience: # We don't consider (for now) Forward Transfer
            if (current_experience > 0) and (index != current_experience - 1):
                continue
            current_results = strategy.eval(eval_exp)
            current_results = {k: v for k, v in current_results.items() if 'eval_phase/eval_stream' in k}
            for key, value in current_results.items():
                if (self.evaluation_metric in key) and ('eval_phase/eval_stream' in key) and (f"Exp00{index}" in key):
                    current_metrics.append(value)
                    break
        if model is None:
            stdout_debug_print(f"Current metrics: {current_metrics} (no model)", color='green')
        else:
            stdout_debug_print(f"Current metrics: {current_metrics} (prev model)", color='green')
        if model is not None:
            strategy.model = old_model
        # Resume the logger now
        for logger in strategy.evaluator.loggers:
            if isinstance(logger, CustomCSVLogger):
                logger.resume()        
        current_metrics_average = sum(current_metrics) / len(current_metrics)
        return current_metrics, current_metrics_average
    
    def select_best_model(self, strategy):
        prev_model, current_model = self.prev_model, strategy.model
        current_experience = strategy.experience.current_experience
        current_metrics, current_metrics_average = self.__compute_current_metrics(strategy, current_experience, model=None)
        if prev_model is None:
            best_model = current_model
            self.prev_metric_values = current_metrics
            stdout_debug_print(f"Selected current model after experience {current_experience}", color='red')
        else:
            if self._eval_stream is None:
                raise RuntimeError(f"Eval Stream is not set for {self.__class__.__name__}")
            prev_metrics, prev_metrics_average = self.__compute_current_metrics(strategy, current_experience, model=self.prev_model)
            if current_metrics_average >= prev_metrics_average:
                best_model = current_model
                self.prev_metric_values = current_metrics
                stdout_debug_print(f"Selected current model after experience {current_experience}", color='red')
            else:
                stdout_debug_print(f"Kept previous model after experience {current_experience}", color='red')
                best_model = None
                self.prev_metric_values = prev_metrics
        return best_model
    
    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        and freeze the prev model and freeze the last layer of current model
        """
        best_model = self.select_best_model(strategy)
        if best_model is not None:
            self.prev_model = copy.deepcopy(best_model) #copy.deepcopy(strategy.model)

        freeze_everything(self.prev_model)

        last_fc_name, last_fc = get_last_fc_layer(strategy.model)

        for param in last_fc.parameters():
            param.requires_grad = False

    def before_training(self, strategy, **kwargs):
        """
        Check if the model is an instance of base class to ensure get_features()
        is implemented
        """
        if not isinstance(strategy.model, BaseModel):
            raise NotImplementedError(BaseModel.__name__ + ".get_features()")


__all__ = ['DoubleLFLPlugin']