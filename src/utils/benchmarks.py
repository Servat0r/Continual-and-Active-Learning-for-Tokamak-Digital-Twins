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


def subsample(data, column='has_turbulence'):
    positives = data[data[column] == 1]
    negatives = data[data[column] == 0]
    positives_len = len(positives)
    negatives_len = len(negatives)
    print(positives_len, negatives_len)
    min_len = min(positives_len, negatives_len)
    positive_sample = positives.sample(n=min_len, random_state=0)
    negative_sample = negatives.sample(n=min_len, random_state=0)
    data = pd.concat([positive_sample, negative_sample]).sample(frac=1, random_state=42)  # Shuffle the resulting dataset
    return data


def make_benchmark(
        csv_file, train_datasets, eval_datasets, test_datasets, task='regression',
        input_columns=BASELINE_HIGHPOW_INPUTS, output_columns=BASELINE_HIGHPOW_OUTPUTS,
        NUM_CAMPAIGNS=5, dtype='float64', *, test_size=0.2, eval_size=0.25,
        normalize_inputs=False, normalize_outputs=False, log_folder=None,
        dataset_type='complete', filter_by_leq: dict[str, int | float] = None,
        filter_by_geq: dict[str, int | float] = None,
        transform=None, target_transform=None, apply_subsampling=False,
):
    float_precision = dtype
    dtype = get_dtype_from_str(dtype)
    if output_columns is None:
        output_columns = BASELINE_HIGHPOW_OUTPUTS
    debug_print(output_columns)
    data = pd.read_csv(csv_file)
    debug_print(f"There are {len(data)} items in the dataset in {csv_file}.")
    filter_function = lambda x: 1 if any([x[column] for column in output_columns]) else 0
    data['has_turbulence'] = data.apply(filter_function, axis=1)
    def stratification_function(x):
        mult = 1 if x['has_turbulence'] == 1 else -1
        return x['campaign'] * mult
    data['stratify'] = data.apply(stratification_function, axis=1)
    # Split the data into train and test sets with stratification
    dev_data, test_data = train_test_split(
        data, test_size=test_size, random_state=42, shuffle=True, stratify=data.stratify
    )
    train_data, eval_data = train_test_split(
        dev_data, test_size=eval_size, random_state=42, shuffle=True, stratify=dev_data.stratify
    )
    dirname = os.path.dirname(csv_file)
    train_data.to_csv(f'{dirname}/raw_train_data_{task}.csv', index=False)
    eval_data.to_csv(f'{dirname}/raw_eval_data_{task}.csv', index=False)
    test_data.to_csv(f'{dirname}/raw_test_data_{task}.csv', index=False)
    train_data = train_data.drop(columns=['stratify'])
    eval_data = eval_data.drop(columns=['stratify'])
    test_data = test_data.drop(columns=['stratify'])
    print(
        f"After stratification: ",
        f"train_data has {len(train_data[train_data.has_turbulence == 1])} positives and {len(train_data[train_data.has_turbulence == 0])} negatives.",
        f"eval_data has {len(eval_data[eval_data.has_turbulence == 1])} positives and {len(eval_data[eval_data.has_turbulence == 0])} negatives.",
        f"test_data has {len(test_data[test_data.has_turbulence == 1])} positives and {len(test_data[test_data.has_turbulence == 0])} negatives.",
        sep='\n', end='\n', flush=True,
    )
    # Subsampling and Stratification
    if apply_subsampling:
        train_data = subsample(train_data, column='has_turbulence')
        eval_data = subsample(eval_data, column='has_turbulence')
        test_data = subsample(test_data, column='has_turbulence')
        total_data = len(train_data) + len(eval_data) + len(test_data)
        debug_print(
            f"There are {total_data} items in the dataset after subsampling:"
            f"{len(train_data)} train, {len(eval_data)} eval, and {len(test_data)} test items."
        )
    if dataset_type == 'not_null':
        train_data = train_data[train_data.has_turbulence != 0]
        eval_data = eval_data[eval_data.has_turbulence != 0]
        test_data = test_data[test_data.has_turbulence != 0]
        total_data = len(train_data) + len(eval_data) + len(test_data)
        debug_print(
            f"There are {total_data} items in the dataset after filtering:"
            f"{len(train_data)} train, {len(eval_data)} eval, and {len(test_data)} test items."
        )
    if task == 'classification':
        output_columns = ['has_turbulence']
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
    train_data.to_csv(f'{dirname}/final_train_data_{task}.csv', index=False)
    eval_data.to_csv(f'{dirname}/final_eval_data_{task}.csv', index=False)
    test_data.to_csv(f'{dirname}/final_test_data_{task}.csv', index=False)
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
