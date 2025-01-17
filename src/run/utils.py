from collections import defaultdict
from avalanche.evaluation.metrics import loss_metrics, timing_metrics
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin, GenerativeReplayPlugin

from ..utils import *


def get_metric_names_list(task):
    if task == 'regression':
        return ['Loss_Exp', 'R2Score_Exp', 'RelativeDistance_Exp', 'Forgetting_Exp']
    else:
        return ['Loss_Exp', 'BinaryAccuracy_Exp']


def get_title_names_list(task):
    if task == 'regression':
        return [
            'Loss over each experience', 'R2 Score over each experience',
            'Relative Distance over each experience', 'Forgetting over each experience'
        ]
    else:
        return [
            'Loss over each experience', 'Binary Accuracy over each experience',
        ]


def get_ylabel_names_list(task):
    if task == 'regression':
        return ['Loss', 'R2 Score', 'Relative Distance', 'Forgetting']
    else:
        return ['Loss', 'BinaryAccuracy']


def make_scheduler(scheduler_config, optimizer):
    if scheduler_config:
        scheduler_class = scheduler_config['class']
        scheduler_parameters = scheduler_config['parameters']
        scheduler_metric = scheduler_config['metric']
        scheduler_first_epoch_only = scheduler_config['first_epoch_only']
        scheduler_first_exp_only = scheduler_config['first_exp_only']
        return LRSchedulerPlugin(
            scheduler_class(optimizer, **scheduler_parameters),
            metric=scheduler_metric,
            first_exp_only=scheduler_first_exp_only,
            first_epoch_only=scheduler_first_epoch_only,
        )
    else:
        return None


def get_metrics(loss_type):
    if loss_type == 'GaussianNLL':
        metrics = \
            timing_metrics(epoch=True, experience=True, stream=True) + \
            gaussian_mse_metrics(epoch=True, experience=True, stream=True) + \
            relative_distance_metrics(epoch=True, experience=True, stream=True) + \
            r2_score_metrics(epoch=True, experience=True, stream=True) + \
            gaussian_variance_metrics(epoch=True, experience=True, stream=True) + \
            renamed_forgetting_metrics(experience=True, stream=True)
    elif loss_type in ['BCE', 'bce', 'BCEWithLogits', 'bce_with_logits']:
        metrics = \
            timing_metrics(epoch=True, experience=True, stream=True) + \
            binary_accuracy_metrics(epoch=True, experience=True, stream=True) + \
            f1_metrics(epoch=True, experience=True, stream=True)
    else:
        metrics = \
            timing_metrics(epoch=True, experience=True, stream=True) + \
            relative_distance_metrics(epoch=True, experience=True, stream=True) + \
            r2_score_metrics(epoch=True, experience=True, stream=True) + \
            renamed_forgetting_metrics(experience=True, stream=True) + \
            renamed_bwt_metrics(experience=True, stream=True)
    return metrics


def process_test_results(final_test_results: dict):
    results = defaultdict(dict)
    for key, value in final_test_results.items():
        info = extract_metric_info(key)
        if info['type'] == 'exp':
            exp_id = info['exp_number']
            name = info['name']
            results[str(exp_id)][name] = value
        else:
            results["stream"][key] = value
    return results


__all__ = [
    'get_metric_names_list',
    'get_title_names_list',
    'get_ylabel_names_list',
    'make_scheduler',
    'get_metrics',
    'process_test_results',
]