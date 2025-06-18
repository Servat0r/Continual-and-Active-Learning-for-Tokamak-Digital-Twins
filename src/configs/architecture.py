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


# Il grosso problema è QUI! "saved_model_handler" = ??? Come si sistema??
@ConfigParser.standardizer('architecture')
def architecture_std(config: dict[str, Any], key: str):
    data = config[key]
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    if 'task' in config:
        task = config['task'] # NOTICE: Before it was config.pop('task')
    elif ('general' in config) and ('task' in config['general']):
        task = config['general']['task']
    else:
        task = 'regression'
    if name == 'saved':
        result = {
            "model_type": data["model_class_name"],
            "model_folder": data["model_folder"],
            "parameters": parameters,
            "@extra": {
                "saved": True,
                "model_name": data["model_name"],
                "task": task
            }
        }
    else:
        result = {
            "model_type": name,
            "model_folder": None,
            "parameters": parameters
        }
    return result


@ConfigParser.processor('architecture')
def architecture_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    extra = data.pop("@extra", None)
    is_saved = extra.get('saved', False) if extra else False
    if is_saved:
        model_class_name = data['model_type']
        model_folder = data['model_folder']
        model_name = extra['model_name']
        task = extra['task']
        parameters = data['parameters']
        return saved_model_handler( # TODO FIX!
            model_folder=model_folder, model_name=model_name,
            model_class_name=model_class_name, **parameters,
            task_id=task_id
        )
    else:
        model_type = data['model_type']
        if model_type.upper() == 'MLP':
            return mlp_config(parameters, gaussian=False, task=task, task_id=task_id)
        elif model_type.replace('_', '').upper() == 'GAUSSIANMLP':
            return mlp_config(parameters, gaussian=True, task=task, task_id=task_id)
        elif model_type.upper() == 'TRANSFORMER':
            return transformer_config(parameters, gaussian=False, task=task, task_id=task_id)
        else:
            raise ValueError(f"Invalid architecture name \"{model_type}\"")


__all__ = ['mlp_config', 'transformer_config', 'saved_model_handler', 'architecture_handler']
