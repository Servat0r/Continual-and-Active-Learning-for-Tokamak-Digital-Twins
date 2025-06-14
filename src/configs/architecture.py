from typing import Any
import torch

from .model_saving import MODELS_DIR
from .parser import *
from ..utils import SimpleRegressionMLP, SimpleClassificationMLP, GaussianRegressionMLP, SimpleConv1DModel, TransformerRegressor

__model_classes = {
    'MLP': SimpleRegressionMLP,
    'GaussianMLP': GaussianRegressionMLP,
    'ClassificationMLP': SimpleClassificationMLP,
    'ConvNet': SimpleConv1DModel,
    'Transformer': TransformerRegressor
}


def mlp_config(parameters: dict[str, Any], gaussian=False, task='regression', task_id=0):
    default_config = {
        'hidden_size': 512,
        'hidden_layers': 2,
        'input_size': 15,
        'output_size': 4,
        'drop_rate': 0.5,
        'dtype': 'float32'
    }
    default_config.update(parameters)
    assert isinstance(default_config['hidden_size'], int) and default_config['hidden_size'] > 0
    assert isinstance(default_config['hidden_layers'], int) and default_config['hidden_layers'] > 0
    assert isinstance(default_config['input_size'], int) and default_config['input_size'] > 0
    assert isinstance(default_config['output_size'], int) and default_config['output_size'] > 0
    assert isinstance(default_config['drop_rate'], float) and 0 <= default_config['drop_rate'] <= 1
    assert default_config['dtype'] in ['float16', 'float32', 'float64']
    for key in default_config:
        assert key in [
            'hidden_size', 'hidden_layers', 'input_size', 'output_size',
            'drop_rate', 'dtype', 'include_softplus', 'activation'
        ]
    if task == 'regression':
        if gaussian:
            return GaussianRegressionMLP(**default_config)
        else:
            return SimpleRegressionMLP(**default_config)
    elif task == 'classification':
        # todo add Gaussian Classification?
        return SimpleClassificationMLP(**default_config)
    else:
        raise ValueError(f"Invalid task \"{task}\"")


def transformer_config(parameters: dict[str, Any], gaussian=False, task='regression', task_id=0):
    default_config = {
        'input_size': 15,
        'output_size': 4,
        'd_model': 8,
        'nhead': 8,
        'num_layers': 2,
        'dropout': 0.25
    }
    default_config.update(parameters)
    assert isinstance(default_config['input_size'], int) and default_config['input_size'] > 0
    assert isinstance(default_config['output_size'], int) and default_config['output_size'] > 0
    assert isinstance(default_config['d_model'], int) and default_config['d_model'] > 0
    assert isinstance(default_config['nhead'], int) and default_config['nhead'] > 0
    assert isinstance(default_config['num_layers'], int) and default_config['num_layers'] > 0
    assert isinstance(default_config['dropout'], float) and 0 <= default_config['dropout'] <= 1
    for key in default_config:
        assert key in [
            'input_size', 'output_size', 'd_model', 'nhead',
            'num_layers', 'dropout'
        ]
    if task == 'regression':
        if gaussian:
            raise NotImplementedError("Gaussian transformer not implemented yet")
        else:
            return TransformerRegressor(**default_config)
    elif task == 'classification':
        raise NotImplementedError("Classification transformer not implemented yet")
    else:
        raise ValueError(f"Invalid task \"{task}\"")



def saved_model_handler(model_folder: str, model_name: str, model_class_name: str, **kwargs):
    task_id = kwargs.pop('task_id', 0)
    model_class = __model_classes.get(model_class_name, None)
    if not model_class:
        raise ValueError(f"Invalid model class name \"{model_class_name}\"")
    model_path = f'{MODELS_DIR}/{model_folder}/{model_name} task_{task_id}.pt'
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(model_path))
    return model


@ConfigParser.register_handler('architecture')
def architecture_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    if 'task' in kwargs:
        task = kwargs.pop('task')
    elif ('general' in kwargs) and ('task' in kwargs['general']):
        task = kwargs['general']['task']
    else:
        task = 'regression'
    if name == 'saved':
        model_folder = data.get('model_folder', '')
        model_name = data.get('model_name', 'model.pt')
        model_class_name = data.get('model_class_name', 'MLP')
        return saved_model_handler(
            model_folder=model_folder, model_name=model_name,
            model_class_name=model_class_name, **parameters,
            task_id=task_id
        )
    if (name == 'MLP') or (name == 'mlp'):
        return mlp_config(parameters, gaussian=False, task=task, task_id=task_id)
    elif (name == 'GaussianMLP') or (name == 'gaussian_mlp'):
        return mlp_config(parameters, gaussian=True, task=task, task_id=task_id)
    elif (name == 'Transformer') or (name == 'transformer'):
        return transformer_config(parameters, gaussian=False, task=task, task_id=task_id)
    else:
        raise ValueError(f"Invalid architecture name \"{name}\"")


__all__ = ['mlp_config', 'transformer_config', 'saved_model_handler', 'architecture_handler']
