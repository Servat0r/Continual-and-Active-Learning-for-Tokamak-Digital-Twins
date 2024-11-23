# Tests for results after training
import json
import os.path
import pandas as pd
from src.utils import *
from sklearn.model_selection import train_test_split
import torch


def load_models_and_raw_data(log_folder: str, load_intermediate_models: bool = False, test_size=0.2):
    with open(os.path.join(log_folder, 'config.json'), 'r') as fp:
        config = json.load(fp)
    dtype = config['dtype']
    pow_type = config['pow_type']
    cluster_type = config['cluster_type']
    dataset_type = config['dataset_type']
    task = config['task']
    num_campaigns = config['num_campaigns']
    normalize_inputs = config['normalize_inputs']
    normalize_outputs = config['normalize_outputs']

    model = torch.load(os.path.join(log_folder, 'model.pt'))
    intermediate_models = {}
    if load_intermediate_models:
        for i in range(config['num_campaigns']):
            intermediate_models[i] = torch.load(os.path.join(log_folder, f'model_after_exp_{i}.pt'))
    csv_file = f'data/baseline/cleaned/{pow_type}_cluster/{cluster_type}/{dataset_type}_dataset.csv'
    float_precision = dtype
    dtype = get_dtype_from_str(dtype)
    OUTPUTS = BASELINE_HIGHPOW_OUTPUTS if task == 'regression' else ['has_turbulence']
    data = pd.read_csv(csv_file)
    print(len(data))
    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    transform = None
    target_transform = None
    if normalize_inputs:
        inputs_mean = torch.load(os.path.join(log_folder, "input_mean.pt"))
        inputs_std = torch.load(os.path.join(log_folder, "input_std.pt"))
        transform = CustomNormalize(inputs_mean, inputs_std)
    if normalize_outputs:
        outputs_mean = torch.load(os.path.join(log_folder, "output_mean.pt"))
        outputs_std = torch.load(os.path.join(log_folder, "output_std.pt"))
        target_transform = CustomNormalize(outputs_mean, outputs_std)
    stuff = {
        'model': model,
        'intermediate_models': intermediate_models,
        'train_data': train_data,
        'test_data': test_data,
        'dtype': dtype,
        'float_precision': float_precision,
        'OUTPUTS': OUTPUTS,
        'transform': transform,
        'target_transform': target_transform,
        'num_campaigns': num_campaigns,
    }
    return stuff

def build_full_datasets(
    train_data, test_data, input_columns, output_columns, transform, target_transform, float_precision, device
):
    full_train_dataset = CSVRegressionDataset(
        train_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
        target_transform=target_transform, filter_by=None, float_precision=float_precision, device=device,
    )
    full_test_dataset = CSVRegressionDataset(
        test_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
        target_transform=target_transform, filter_by=None, float_precision=float_precision, device=device,
    )
    return full_train_dataset, full_test_dataset

def build_experience_datasets(
    train_data, test_data, input_columns, output_columns, transform, target_transform, float_precision,
    device, num_campaigns,
):
    experience_train_datasets = {}
    experience_test_datasets = {}
    for campaign in range(num_campaigns):
        train_dataset = CSVRegressionDataset(
            train_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
            target_transform=target_transform, filter_by={'campaign': [campaign]}, float_precision=float_precision,
            device=device,
        )
        test_dataset = CSVRegressionDataset(
            test_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
            target_transform=target_transform, filter_by={'campaign': [campaign]}, float_precision=float_precision,
            device=device,
        )
        experience_train_datasets[campaign] = train_dataset
        experience_test_datasets[campaign] = test_dataset
    return experience_train_datasets, experience_test_datasets


#### Main workload functions
def full_models_and_datasets_load(log_folder: str, load_intermediate_models: bool = False, test_size=0.2):
    stuff = load_models_and_raw_data(log_folder, load_intermediate_models, test_size)
    model = stuff['model']
    intermediate_models = stuff['intermediate_models']
    train_data = stuff['train_data']
    test_data = stuff['test_data']
    float_precision = stuff['float_precision']
    OUTPUTS = stuff['OUTPUTS']
    transform = stuff['transform']
    target_transform = stuff['target_transform']
    num_campaigns = stuff['num_campaigns']
    device = 'cpu'
    full_train_dataset, full_test_dataset = build_full_datasets(
        train_data, test_data, BASELINE_HIGHPOW_INPUTS, OUTPUTS, transform, target_transform,
        float_precision, device,
    )
    return model, intermediate_models, full_train_dataset, full_test_dataset