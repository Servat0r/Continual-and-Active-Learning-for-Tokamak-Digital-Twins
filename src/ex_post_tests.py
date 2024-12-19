# Tests for results after training
from typing import *
import json
import os.path
from pathlib import Path

import torch
import pandas as pd

from src.utils import *


def get_normalization_transforms(
        data, input_columns, output_columns, transform=None, target_transform=None, dtype=float32,
        normalize_inputs: bool = True, normalize_outputs: bool = False,
) -> tuple[Any, Any]:
    """
    Given (transform, target_transform) and the other parameters, returns the compositions
    - CustomNormalize(mean(input_0), std(input_0)) ° {transform if normalize_inputs else identity}
    - CustomNormalize(mean(output_0), std(output_0)) ° {target_transform if normalize_outputs else identity}
    """
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
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Returns the triple (train_data, eval_data, test_data) as pandas DataFrames according to the
    specific baseline dataset selected.
    :param pow_type: One of {"highpow", "lowpow"}
    :param cluster_type: One of {"Ip_Pin_based", "tau_based", "pca_based"}
    :param dataset_type: One of {"complete", "not_null"}
    :param raw_or_final: One of {"raw", "final"}. If "raw", loads unprocessed data just right after
    train-eval-test split. If "final", loads already processed data after subsampling and zero-drops,
    but BEFORE normalization (this is done for allowing training with non-normalized data).
    :param train: If True, loads train data.
    :param validation: If True, loads validation data.
    :param test: If True, loads test data.
    :param task: Either "classification" or "regression".
    :return: The triple (train_data, eval_data, test_data) as pandas DataFrames, with each value being
    None if the corresponding train/validation/test input parameter is False.
    """
    data_folder = f'data/baseline/cleaned/{pow_type}_cluster/{cluster_type}'
    train_filename = f'{raw_or_final}_train_data_{task}_{dataset_type}.csv'
    eval_filename = f'{raw_or_final}_eval_data_{task}_{dataset_type}.csv'
    test_filename = f'{raw_or_final}_test_data_{task}_{dataset_type}.csv'
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
) -> dict[int, torch.nn.Module]:
    """
    Loads saved models according to experiment data.
    :param pow_type: One of {"highpow", "lowpow"}
    :param cluster_type: One of {"Ip_Pin_based", "tau_based", "pca_based"}
    :param dataset_type: One of {"complete", "not_null"}
    :param task: One of {"classification", "regression"}
    :param outputs: Output columns string, e.g. "efe_efi_pfe_pfi"
    :param strategy: Strategy class name, e.g. "Naive" or "Replay"
    :param extra_log_folder: Extra log folder name (see README), e.g. "Base" or "Buffer 2000"
    :param task_ids: For which run(s) we want to load final model(s).
    :return: A dictionary of the form `task_id -> model`.
    """
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
) -> tuple[CSVRegressionDataset, CSVRegressionDataset, CSVRegressionDataset]:
    """
    Retrieves full train/validation/test datasets, i.e. with all experiences together.
    :param train_data: DataFrame with (all) train data.
    :param eval_data: DataFrame with (all) validation data.
    :param test_data: DataFrame with (all) test data.
    :param input_columns: Input columns, defaults to BASELINE_HIGHPOW_INPUTS (15 columns for high-power experiments).
    :param output_columns: Output columns, defaults to BASELINE_HIGHPOW_OUTPUTS (4 columns for high-power experiments).
    :param transform: Transform to apply to input data (excluding mean-std normalization), defaults to None.
    :param target_transform: Transform to apply to output data (excluding mean-std normalization), defaults to None.
    :param float_precision: Floating point precision, one of {'float32', 'float16', 'float64'}. Defaults to 'float32'.
    :param device: Device to use, one of {'cpu', 'gpu', 'cuda:<id>'}. Defaults to 'cpu'.
    :param normalize_inputs: If True, CustomNormalize(mean(input_exp0), std(input_exp0)) is applied
    to inputs. Defaults to True.
    :param normalize_outputs: If True, CustomNormalize(mean(output_exp0), std(output_exp0)) is applied
    to outputs. Defaults to False.
    :return: The triple (full_train_dataset, full_eval_dataset, full_test_dataset).
    """
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
) -> tuple[dict[int, CSVRegressionDataset], dict[int, CSVRegressionDataset], dict[int, CSVRegressionDataset]]:
    """
    Builds train/eval/test data for each experience from full pandas DataFrames.
    :param train_data: DataFrame with (all) train data.
    :param eval_data: DataFrame with (all) validation data.
    :param test_data: DataFrame with (all) test data.
    :param input_columns: Input columns, defaults to BASELINE_HIGHPOW_INPUTS (15 columns for high-power experiments).
    :param output_columns: Output columns, defaults to BASELINE_HIGHPOW_OUTPUTS (4 columns for high-power experiments).
    :param transform: Transform to apply to input data (excluding mean-std normalization), defaults to None.
    :param target_transform: Transform to apply to output data (excluding mean-std normalization), defaults to None.
    :param float_precision: Floating point precision, one of {'float32', 'float16', 'float64'}. Defaults to 'float32'.
    :param device: Device to use, one of {'cpu', 'gpu', 'cuda:<id>'}. Defaults to 'cpu'.
    :param num_campaigns: Number of experiences to build, defaults to 10.
    :param normalize_inputs: If True, CustomNormalize(mean(input_exp0), std(input_exp0)) is applied
    to inputs. Defaults to True.
    :param normalize_outputs: If True, CustomNormalize(mean(output_exp0), std(output_exp0)) is applied
    to outputs. Defaults to False.
    :return: The triple (experience_train_dataset, experience_eval_dataset, experience_test_dataset), with each of
    them being a dict of the form experience -> CSVRegressionDataset.
    """
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