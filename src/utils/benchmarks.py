import os
from rich import print

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from avalanche.benchmarks import *
from torchvision.transforms import transforms

from .misc import *
from .datasets import *
from .transforms import *


def build_normalization_transforms(data, columns, dtype, transform=None):
    mean = torch.tensor(data[columns].mean(axis=0), dtype=dtype)
    std = torch.tensor(data[columns].std(axis=0), dtype=dtype)
    if transform is not None:
        transform = transforms.Compose([CustomNormalize(mean, std), transform])
    else:
        transform = CustomNormalize(mean, std)
    return transform, (mean, std)


def subsample(data, column='has_turbulence'):
    """
    Subsampling utility, in particular it performs down-sampling to the minimum subset size across
    all column values.
    :param data: DataFrame to subsample.
    :param column: Column to use for subsampling.
    :return: Down-sampled DataFrame.
    """
    positives = data[data[column] == 1]
    negatives = data[data[column] == 0]
    positives_len = len(positives)
    negatives_len = len(negatives)
    min_len = min(positives_len, negatives_len)
    positive_sample = positives.sample(n=min_len, random_state=0)
    negative_sample = negatives.sample(n=min_len, random_state=0)
    data = pd.concat([positive_sample, negative_sample]).sample(frac=1, random_state=42)  # Shuffle the resulting dataset
    return data


def stratified_split_df(df, input_cols, output_cols, n_bins=10,
                          train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                          random_state=None):
    """
    Stratifies a DataFrame for regression by binning each input and output column
    into n_bins (using quantile-based binning) and splitting the rows such that each 
    unique combination of binned values is proportionally represented in train, validation,
    and test sets.

    Parameters:
      - df: pandas DataFrame containing all data.
      - input_cols: list of column names corresponding to input features.
      - output_cols: list of column names corresponding to output features.
      - n_bins: int, number of bins to use for each column.
      - train_ratio, val_ratio, test_ratio: floats that add up to 1.0, representing the 
        desired proportions for each split.
      - random_state: int or None, for reproducible shuffling.

    Returns:
      - train_df: DataFrame for training.
      - val_df: DataFrame for validation.
      - test_df: DataFrame for testing.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    df = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    binned_features = {}
    
    # Bin each column in both input and output using quantile-based binning.
    for col in input_cols + output_cols:
        col_values = df[col].values
        bin_edges = np.percentile(col_values, np.linspace(0, 100, n_bins + 1))
        # np.digitize returns bins starting at 1; adjust right boundary.
        bins = np.digitize(col_values, bin_edges, right=True)
        bins[bins > n_bins] = n_bins  # Ensure maximum values fall into the last bin.
        binned_features[col] = bins

    # Create a combined stratification key (tuple of binned labels for all input and output columns)
    combined_bins = [
        tuple(binned_features[col][i] for col in input_cols + output_cols)
        for i in range(len(df))
    ]
    
    # Group row indices by their combined bin tuple.
    group_indices = {}
    for idx, grp in enumerate(combined_bins):
        group_indices.setdefault(grp, []).append(idx)
    
    # Prepare lists for the indices for each split.
    train_indices, val_indices, test_indices = [], [], []
    
    # For each group, shuffle and allocate indices according to the given ratios.
    for group, indices in group_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        n = len(indices)
        # For very small groups, use a custom rule.
        if n < 3:
            if n >= 1:
                train_indices.extend(indices[:1])
            if n >= 2:
                val_indices.extend(indices[1:2])
            continue
        
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        n_test  = n - n_train - n_val  # Remaining indices go to test.

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # Create the final DataFrames using the row indices.
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df   = df.iloc[val_indices].reset_index(drop=True)
    test_df  = df.iloc[test_indices].reset_index(drop=True)
    
    return train_df, val_df, test_df


def make_benchmark(
        csv_file: str | os.PathLike, train_datasets: list[CSVRegressionDataset | AvalancheDataset],
        eval_datasets: list[CSVRegressionDataset | AvalancheDataset],
        test_datasets: list[CSVRegressionDataset | AvalancheDataset],
        task='regression', input_columns=QUALIKIZ_HIGHPOW_INPUTS,
        output_columns=QUALIKIZ_HIGHPOW_OUTPUTS, NUM_CAMPAIGNS=5,
        dtype='float32', *, test_size=0.2, eval_size=0.25,
        normalize_inputs=True, normalize_outputs=False, log_folder=None,
        dataset_type='complete', filter_by_leq: dict[str, int | float] = None,
        filter_by_geq: dict[str, int | float] = None,
        transform=None, target_transform=None, apply_subsampling=False,
        load_saved_final_data: bool = False,
) -> CLScenario:
    """
    Utility to build Continual Learning benchmark. It can be used either with dynamic building of
    train/validation/test sets from an original dataset, or by loading already saved datasets. The
    requirement of train_datasets, eval_datasets and test_datasets is for allowing internal inspections
    after the benchmark has been built, mainly for debugging purposes.
    :param csv_file: CSV file to use for loading data.
    :param train_datasets: List to be used for appending train datasets.
    :param eval_datasets: List to be used for appending validation datasets.
    :param test_datasets: List to be used for appending test datasets.
    :param task: One of {"classification", "regression"}.
    :param input_columns: Input columns, defaults to QUALIKIZ_HIGHPOW_INPUTS (15 columns for high-power case).
    :param output_columns: Output columns, defaults to QUALIKIZ_HIGHPOW_OUTPUTS (4 columns for high-power case).
    :param NUM_CAMPAIGNS: Number of experiences (called "campaigns").
    :param dtype: Datatype to be used for loading floating-point data. Defaults to 'float32'.
    :param test_size: Test size w.r.t. the whole dataset. Defaults to 0.2.
    :param eval_size: Validation size w.r.t. the couple (train, validation). Defaults to 0.25 (i.e., it will be
    0.25 * 0.8 = 20% of whole dataset).
    :param normalize_inputs: If True, normalizes inputs. Defaults to True.
    :param normalize_outputs: If True, normalizes outputs. Defaults to False.
    :param log_folder: Log folder to be used for saving (mean, std) couples.
    :param dataset_type: One of {"complete", "not_null"}.
    :param filter_by_leq: A dictionary used to filter out values in the dataset according to given parameters.
    Specifically, for each couple (column, value), every row such that row[column] > value will be excluded.
    :param filter_by_geq: A dictionary used to filter out values in the dataset according to given parameters.
    Specifically, for each couple (column, value), every row such that row[column] < value will be excluded.
    :param transform: Extra input transform(s).
    :param target_transform: Extra output transform(s).
    :param apply_subsampling: If True, down-samples data. Defaults to False.
    :param load_saved_final_data: If True, loads already saved and preprocessed datasets. Defaults to False.
    :return: A CLScenario object representing Continual Learning benchmark.
    """
    float_precision = dtype
    dtype = get_dtype_from_str(dtype)
    if output_columns is None:
        output_columns = QUALIKIZ_HIGHPOW_OUTPUTS
    dirname = os.path.dirname(csv_file)
    if not load_saved_final_data:
        data = pd.read_csv(csv_file)
        filter_function = lambda x: 1 if any([x[column] for column in output_columns]) else 0
        data['has_turbulence'] = data.apply(filter_function, axis=1)
        def stratification_function(x):
            mult = 1 if x['has_turbulence'] == 1 else -1
            return x['campaign'] * mult
        data['stratify'] = data.apply(stratification_function, axis=1)
        # Now ensure that no class has just one sample
        counts = data['stratify'].value_counts()
        valid_classes = counts[counts > 1].index
        stdout_debug_print(f"Before filtering out classes with less than 2 samples: {len(data)} items", color='red')
        data = data[data['stratify'].isin(valid_classes)]
        stdout_debug_print(f"After having filtered out classes with less than 2 samples: {len(data)} items", color='red')

        """
        train_ratio, eval_ratio, test_ratio = (1 - eval_size) * (1 - test_size), eval_size * (1 - test_size), test_size
        stdout_debug_print(f"train_ratio = {train_ratio}, eval_ratio = {eval_ratio}, test_ration = {test_ratio}", color='red')
        train_data, eval_data, test_data = stratified_split_df(
            data, input_columns + ['stratify'], output_columns, n_bins=5, train_ratio=train_ratio,
            val_ratio=eval_ratio, test_ratio=test_ratio, random_state=42
        )
        """
        # Split the data into train and test sets with stratification
        dev_data, test_data = train_test_split(
            data, test_size=test_size, random_state=42, shuffle=True, stratify=data.stratify # OCCHIO!
        )
        train_data, eval_data = train_test_split(
            dev_data, test_size=eval_size, random_state=42, shuffle=True, stratify=dev_data.stratify
        )
        #"""
        train_data.to_csv(f'{dirname}/raw_train_data_{task}_{dataset_type}.csv', index=False)
        eval_data.to_csv(f'{dirname}/raw_eval_data_{task}_{dataset_type}.csv', index=False)
        test_data.to_csv(f'{dirname}/raw_test_data_{task}_{dataset_type}.csv', index=False)
        train_data = train_data.drop(columns=['stratify'])
        eval_data = eval_data.drop(columns=['stratify'])
        test_data = test_data.drop(columns=['stratify'])
        debug_print(
            f"After stratification: ",
            f"train_data has {len(train_data[train_data.has_turbulence == 1])} positives and {len(train_data[train_data.has_turbulence == 0])} negatives.",
            f"eval_data has {len(eval_data[eval_data.has_turbulence == 1])} positives and {len(eval_data[eval_data.has_turbulence == 0])} negatives.",
            f"test_data has {len(test_data[test_data.has_turbulence == 1])} positives and {len(test_data[test_data.has_turbulence == 0])} negatives.",
            file=STDOUT, sep='\n', end='\n', flush=True,
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
                f"{len(train_data)} train, {len(eval_data)} eval, and {len(test_data)} test items.",
                file=STDOUT,
            )
        train_data.to_csv(f'{dirname}/final_train_data_{task}_{dataset_type}.csv', index=False)
        eval_data.to_csv(f'{dirname}/final_eval_data_{task}_{dataset_type}.csv', index=False)
        test_data.to_csv(f'{dirname}/final_test_data_{task}_{dataset_type}.csv', index=False)
    else:
        train_data = pd.read_csv(f'{dirname}/final_train_data_{task}_{dataset_type}.csv')
        eval_data = pd.read_csv(f'{dirname}/final_eval_data_{task}_{dataset_type}.csv')
        test_data = pd.read_csv(f'{dirname}/final_test_data_{task}_{dataset_type}.csv')
        #eval_data = pd.read_csv(f'{dirname}/final_test_data_{task}_{dataset_type}.csv')
        #test_data = pd.read_csv(f'{dirname}/final_eval_data_{task}_{dataset_type}.csv')

    if task == 'classification':
        output_columns = ['has_turbulence']

    # Handling normalizations
    first_exp_train_data = train_data[train_data.campaign == 0]
    if normalize_inputs:
        norm_transform, (mean, std) = build_normalization_transforms(first_exp_train_data, input_columns, dtype)
        if log_folder:
            torch.save(mean, os.path.join(log_folder, "input_mean.pt"))
            torch.save(std, os.path.join(log_folder, "input_std.pt"))
        transform = transforms.Compose([norm_transform, transform]) if transform else norm_transform
    if normalize_outputs:
        norm_target_transform, (mean, std) = build_normalization_transforms(first_exp_train_data, output_columns, dtype)
        if log_folder:
            torch.save(mean, os.path.join(log_folder, "output_mean.pt"))
            torch.save(std, os.path.join(log_folder, "output_std.pt"))
        target_transform = transforms.Compose([norm_target_transform, target_transform]) \
            if target_transform else norm_target_transform
        # todo be careful on the fact that normalization is not included for inverse()-based preprocess_* methods

    for campaign in range(NUM_CAMPAIGNS):
        debug_print(f"[yellow]Loading data for campaign {campaign} ...[/yellow]", file=STDOUT)
        debug_print(f"[yellow]Input Columns = {input_columns}\nOutput Columns = {output_columns}[/yellow]")
        train_dataset, eval_dataset, test_dataset = get_avalanche_csv_regression_datasets(
            train_data, eval_data, test_data, input_columns=input_columns, output_columns=output_columns,
            filter_by={'campaign': [campaign]}, float_precision=float_precision,
            device='cpu', transform=transform, target_transform=target_transform,
            filter_by_leq=filter_by_leq, filter_by_geq=filter_by_geq, task_label=campaign
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


__all__ = [
    'build_normalization_transforms', 'subsample', 'make_benchmark'
]