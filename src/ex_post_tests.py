# Tests for results after training
import json
import os.path
from pathlib import Path

import torch
import pandas as pd

from src.utils import *


def get_normalization_transforms(
        data, input_columns, output_columns, transform=None, target_transform=None, dtype=float32,
        normalize_inputs: bool = True, normalize_outputs: bool = False,
):
    first_exp_data = data[data.campaign == 0]
    if normalize_inputs:
        norm_transform, (mean, std) = build_normalization_transforms(first_exp_data, input_columns, dtype)
        transform = transforms.Compose([norm_transform, transform]) if transform else norm_transform
    if normalize_outputs:
        norm_target_transform, (mean, std) = build_normalization_transforms(first_exp_data, output_columns, dtype)
        target_transform = transforms.Compose([norm_target_transform, target_transform]) \
            if target_transform else norm_target_transform
    return transform, target_transform


def load_baseline_csv_data(
        pow_type: str, cluster_type: str, dataset_type: str = 'not_null',
        raw_or_final: str = 'final', train: bool = True,
        validation: bool = True, test: bool = True, task: str = 'regression',
):
    data_folder = f'data/baseline/cleaned/{pow_type}_cluster/{cluster_type}'
    train_filename = f'{raw_or_final}_train_data_{task}.csv'
    eval_filename = f'{raw_or_final}_eval_data_{task}.csv'
    test_filename = f'{raw_or_final}_test_data_{task}.csv'
    train_data, eval_data, test_data = None, None, None
    if train:
        train_data = pd.read_csv(f'{data_folder}/{train_filename}')
    if validation:
        eval_data = pd.read_csv(f'{data_folder}/{eval_filename}')
    if test:
        test_data = pd.read_csv(f'{data_folder}/{test_filename}')
    return train_data, eval_data, test_data


def get_baseline_tensor_data(
        data: pd.DataFrame | dict[str, pd.DataFrame],
        input_columns: list[str], output_columns: list[str],
):
    if not isinstance(data, dict):
        data = {'default': data}
    tensor_data = {}
    for key, df in data.items():
        ...


def load_models(
        pow_type: str, cluster_type: str, dataset_type: str = 'not_null',
        task: str = 'regression', outputs: str = 'efe_efi_pfe_pfi',
        strategy: str = 'Naive', extra_log_folder: str = 'Base',
        task_ids: int | list[int] = 0,
):
    base_log_folder = \
        f'logs/{pow_type}/{cluster_type}/{task}/{dataset_type}/{outputs}/{strategy}/{extra_log_folder}'
    if not isinstance(task_ids, list):
        task_ids = [task_ids]
    task_ids.sort()
    path = Path(base_log_folder)
    directories = [
        os.path.join(base_log_folder, d.name) for d in path.iterdir() if d.is_dir()
    ]
    # Order as task_0, task_1, task_2 etc
    directories = sorted(directories, key=lambda x: int(x[-1]))
    models = {}
    for task_id in task_ids:
        state_dict = torch.load(os.path.join(directories[task_id], 'model.pt'))
        config_filename = os.path.join(directories[task_id], 'config.json')
        config = json.load(open(config_filename))
        model_parameters = config['architecture']['parameters']
        model_class = SimpleRegressionMLP if task == 'regression' else SimpleClassificationMLP # TODO REWORK TO INCLUDE OTHER MODELS
        model = model_class(**model_parameters)
        model.load_state_dict(state_dict)
        models[task_id] = model
    return models


def build_full_datasets(
    train_data: pd.DataFrame, eval_data: pd.DataFrame, test_data: pd.DataFrame,
    input_columns: list[str] = BASELINE_HIGHPOW_INPUTS, output_columns: list[str] = BASELINE_HIGHPOW_OUTPUTS,
    transform = None, target_transform = None, float_precision: str = 'float32', device: str = 'cpu',
    normalize_inputs: bool = True, normalize_outputs: bool = False,
):
    dtype = get_dtype_from_str(float_precision)
    transform, target_transform = get_normalization_transforms(
        train_data, input_columns, output_columns, transform=transform, target_transform=target_transform,
        dtype=dtype, normalize_inputs=normalize_inputs, normalize_outputs=normalize_outputs,
    )
    full_train_dataset = CSVRegressionDataset(
        train_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
        target_transform=target_transform, filter_by=None, float_precision=float_precision, device=device,
    )
    full_eval_dataset = CSVRegressionDataset(
        eval_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
        target_transform=target_transform, filter_by=None, float_precision=float_precision, device=device,
    )
    full_test_dataset = CSVRegressionDataset(
        test_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
        target_transform=target_transform, filter_by=None, float_precision=float_precision, device=device,
    )
    return full_train_dataset, full_eval_dataset, full_test_dataset


def build_experience_datasets(
    train_data: pd.DataFrame, eval_data: pd.DataFrame, test_data: pd.DataFrame,
    input_columns: list[str] = BASELINE_HIGHPOW_INPUTS, output_columns: list[str] = BASELINE_HIGHPOW_OUTPUTS,
    transform = None, target_transform = None, float_precision: str = 'float32', device: str = 'cpu',
    num_campaigns: int = 10, normalize_inputs: bool = True, normalize_outputs: bool = False,
):
    dtype = get_dtype_from_str(float_precision)
    transform, target_transform = get_normalization_transforms(
        train_data, input_columns, output_columns, transform=transform, target_transform=target_transform,
        dtype=dtype, normalize_inputs=normalize_inputs, normalize_outputs=normalize_outputs,
    )
    experience_train_datasets = {}
    experience_eval_datasets = {}
    experience_test_datasets = {}
    for campaign in range(num_campaigns):
        train_dataset = CSVRegressionDataset(
            train_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
            target_transform=target_transform, filter_by={'campaign': [campaign]}, float_precision=float_precision,
            device=device,
        )
        eval_dataset = CSVRegressionDataset(
            eval_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
            target_transform=target_transform, filter_by={'campaign': [campaign]}, float_precision=float_precision,
            device=device,
        )
        test_dataset = CSVRegressionDataset(
            test_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
            target_transform=target_transform, filter_by={'campaign': [campaign]}, float_precision=float_precision,
            device=device,
        )
        experience_train_datasets[campaign] = train_dataset
        experience_eval_datasets[campaign] = eval_dataset
        experience_test_datasets[campaign] = test_dataset
    return experience_train_datasets, experience_eval_datasets, experience_test_datasets


__all__ = [
    'load_models', 'load_baseline_csv_data',
    'build_full_datasets', 'build_experience_datasets',
]