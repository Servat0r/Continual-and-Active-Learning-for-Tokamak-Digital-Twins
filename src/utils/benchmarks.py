import os
from rich import print

import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from avalanche.benchmarks import *
from torchvision.transforms import transforms

from .misc import *
from .datasets import *
from .transforms import *


def _build_normalization_transforms(data, columns, dtype, transform=None):
    mean = torch.tensor(data[columns].mean(axis=0), dtype=dtype)
    std = torch.tensor(data[columns].std(axis=0), dtype=dtype)
    print(f"[green]mean = {mean}, std = {std} ...[/green]")
    if transform is not None:
        transform = transforms.Compose([CustomNormalize(mean, std), transform])
    else:
        transform = CustomNormalize(mean, std)
    return transform, (mean, std)


def make_benchmark(
        csv_file, train_datasets, eval_datasets, test_datasets, task='regression',
        input_columns=BASELINE_HIGHPOW_INPUTS, output_columns=BASELINE_HIGHPOW_OUTPUTS,
        NUM_CAMPAIGNS=5, dtype='float64', *, test_size=0.2, eval_size=0.25,
        normalize_inputs=False, normalize_outputs=False, log_folder=None,
        dataset_type='complete', filter_by_leq: dict[str, int | float] = None,
        filter_by_geq: dict[str, int | float] = None,
        transform=None, target_transform=None,
):
    float_precision = dtype
    dtype = get_dtype_from_str(dtype)
    if task == 'classification':
        output_columns = ['has_turbulence']
    elif output_columns is None:
        output_columns = BASELINE_HIGHPOW_OUTPUTS
    data = pd.read_csv(csv_file)
    debug_print(f"There are {len(data)} items in the dataset in {csv_file}.")
    if dataset_type == 'not_null':
        filter_function = lambda x: any([x[column] for column in output_columns])
        data = data[data.apply(filter_function, axis=1)]
        debug_print(f"After filtering, there are {len(data)} items.")
    # Split the data into train and test sets
    dev_data, test_data = train_test_split(data, test_size=test_size, random_state=42, shuffle=True)
    train_data, eval_data = train_test_split(dev_data, test_size=eval_size, random_state=42, shuffle=True)
    if normalize_inputs:
        norm_transform, (mean, std) = _build_normalization_transforms(train_data, input_columns, dtype)
        if log_folder:
            torch.save(mean, os.path.join(log_folder, "input_mean.pt"))
            torch.save(std, os.path.join(log_folder, "input_std.pt"))
        transform = transforms.Compose([norm_transform, transform]) if transform else norm_transform
    if normalize_outputs:
        norm_target_transform, (mean, std) = _build_normalization_transforms(train_data, output_columns, dtype)
        if log_folder:
            torch.save(mean, os.path.join(log_folder, "output_mean.pt"))
            torch.save(std, os.path.join(log_folder, "output_std.pt"))
        target_transform = transforms.Compose([norm_target_transform, target_transform]) \
            if target_transform else norm_target_transform
        # todo be careful on the fact that normalization is not included for inverse()-based preprocess_* methods
    for campaign in range(NUM_CAMPAIGNS):
        print(f"[yellow]Loading data for campaign {campaign} ...[/yellow]")
        train_dataset, eval_dataset, test_dataset = get_avalanche_csv_regression_datasets(
            train_data, eval_data, test_data, input_columns=input_columns, output_columns=output_columns,
            filter_by={'campaign': [campaign]}, float_precision=float_precision,
            device='cpu', transform=transform, target_transform=target_transform,
            filter_by_leq=filter_by_leq, filter_by_geq=filter_by_geq,
        )
        X, y = train_dataset[0]
        print(
            f"[red]Input Shape = {X.shape}[/red]",
            f"[red]Output Shape = {y.shape}[/red]",
            f"[red]Length of Train Dataset = {len(train_dataset)}[/red]",
            f"[red]Length of Validation Dataset = {len(eval_dataset)}[/red]",
            f"[red]Length of Test Dataset = {len(test_dataset)}[/red]"
        )
        train_datasets.append(train_dataset)
        eval_datasets.append(eval_dataset)
        test_datasets.append(test_dataset)
    benchmark = benchmark_from_datasets(train=train_datasets, eval=eval_datasets, test=test_datasets)
    return benchmark
