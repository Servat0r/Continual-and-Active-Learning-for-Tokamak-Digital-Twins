# Tests for results after training
from typing import *
import json
import copy
import os.path
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity

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


def load_complete_dataset(
    pow_type: str, cluster_type: str, dataset_type: str ='not_null', simulator_type: str = 'qualikiz'
):
    """
    Loads the complete dataset for the given parameters. If dataset_type == "not_null", drops the entries
    with "has_turbulence = False".
    """
    data_folder = f'data/{simulator_type}/cleaned/{pow_type}_cluster/{cluster_type}'
    df = pd.read_csv(f"{data_folder}/complete_dataset.csv")
    if simulator_type == 'tglf':
        for column in TGLF_HIGHPOW_OUTPUTS:
            df = df[df[column] <= 500.0]
        df = df[df['efe'] >= 0.0]
        df = df[df['efi'] >= 0.0]
    print(f"Complete dataset has {len(df)} items")
    if dataset_type == 'not_null':
        df = df[df['has_turbulence'] == True]
        print(f"Filtered dataset has {len(df)} items")
    return df


def load_baseline_csv_data(
        pow_type: str, cluster_type: str, dataset_type: str = 'not_null',
        raw_or_final: str = 'final', train: bool = True,
        validation: bool = True, test: bool = True, task: str = 'regression',
        simulator_type: str = 'qualikiz'
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
    data_folder = f'data/{simulator_type}/cleaned/{pow_type}_cluster/{cluster_type}'
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


# TODO: Not very meaningful, see if to remove it!
def load_baseline_csv_data_from_config(
    config: LoggingConfiguration, raw_or_final: str = 'final',
    train: bool = True, validation: bool = True, test: bool = True
):
    simulator_type, pow_type, cluster_type, dataset_type, task = config.get_common_params(end=5)
    return load_baseline_csv_data(
        pow_type, cluster_type, dataset_type, raw_or_final, train, validation, test, task, simulator_type
    )


def extract_dataset_weights(
    simulator_type: str, pow_type: str, cluster_type: str, dataset_type: str = 'not_null',
    raw_or_final: str = 'final', task: str = 'regression', folder_path='.', weights_source: str = 'train'
) -> np.ndarray:
    """
    Loads a dataset and saves the length of each campaign to be used for weighted evaluations.
    """
    train_data, eval_data, test_data = load_baseline_csv_data(
        pow_type, cluster_type, dataset_type, raw_or_final, task=task, simulator_type=simulator_type
    )
    file_path = os.path.join(folder_path, f"{weights_source}_weights_{simulator_type}_{pow_type}_{cluster_type}_{dataset_type}_{raw_or_final}_{task}.txt")
    if weights_source == 'train':
        absolute_weights = train_data.groupby('campaign').size().to_numpy()
    elif weights_source == 'eval':
        absolute_weights = eval_data.groupby('campaign').size().to_numpy()
    elif weights_source == 'test':
        absolute_weights = test_data.groupby('campaign').size().to_numpy()
    print(absolute_weights, absolute_weights.dtype)
    np.savetxt(file_path, absolute_weights, fmt='%d')
    return absolute_weights


def load_dataset_weights(
    simulator_type: str, pow_type: str, cluster_type: str, dataset_type: str = 'not_null',
    raw_or_final: str = 'final', task: str = 'regression', folder_path='.', weights_source: str = 'train'
) -> np.ndarray[np.intp]:
    """
    Loads saved dataset weights from a txt file.
    """
    file_path = os.path.join(folder_path, f"{weights_source}_weights_{simulator_type}_{pow_type}_{cluster_type}_{dataset_type}_{raw_or_final}_{task}.txt")
    absolute_weights = np.loadtxt(file_path, dtype=np.intp)
    return absolute_weights


def load_models(
        pow_type: str, cluster_type: str, dataset_type: str = 'not_null',
        task: str = 'regression', outputs: str = 'efe_efi_pfe_pfi',
        strategy: str = 'Naive', extra_log_folder: str = 'Base',
        task_ids: int | list[int] = 0, params_to_remove: list[str] = None,
) -> dict[int, torch.nn.Module]:
    """
    Loads saved models according to experiment data.
    :param pow_type: One of {"highpow", "lowpow", "mixed"}
    :param cluster_type: One of {"Ip_Pin_based", "tau_based", "wmhd_based", "beta_based"}
    :param dataset_type: One of {"complete", "not_null"}
    :param task: One of {"classification", "regression"}
    :param outputs: Output columns string, e.g. "efe_efi_pfe_pfi"
    :param strategy: Strategy class name, e.g. "Naive" or "Replay"
    :param extra_log_folder: Extra log folder name (see README), e.g. "Base" or "Buffer 2000"
    :param task_ids: For which run(s) we want to load final model(s).
    :param params_to_remove: Parameters to remove from the configuration (for backward
    compatibility).
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
        for param in params_to_remove:
            model_parameters.pop(param, None)
        model_class = SimpleRegressionMLP if task == 'regression' else SimpleClassificationMLP # TODO REWORK TO INCLUDE OTHER MODELS
        model = model_class(**model_parameters)
        model.load_state_dict(state_dict)
        models[task_id] = model
    return models


def build_full_datasets(
    train_data: pd.DataFrame, eval_data: pd.DataFrame, test_data: pd.DataFrame,
    input_columns: list[str] = QUALIKIZ_HIGHPOW_INPUTS, output_columns: list[str] = QUALIKIZ_HIGHPOW_OUTPUTS,
    transform = None, target_transform = None, float_precision: str = 'float32', device: str = 'cpu',
    normalize_inputs: bool = True, normalize_outputs: bool = False,
) -> tuple[CSVRegressionDataset, CSVRegressionDataset, CSVRegressionDataset]:
    """
    Retrieves full train/validation/test datasets, i.e. with all experiences together.
    :param train_data: DataFrame with (all) train data.
    :param eval_data: DataFrame with (all) validation data.
    :param test_data: DataFrame with (all) test data.
    :param input_columns: Input columns, defaults to QUALIKIZ_HIGHPOW_INPUTS (15 columns for high-power experiments).
    :param output_columns: Output columns, defaults to QUALIKIZ_HIGHPOW_OUTPUTS (4 columns for high-power experiments).
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
    input_columns: list[str] = QUALIKIZ_HIGHPOW_INPUTS, output_columns: list[str] = QUALIKIZ_HIGHPOW_OUTPUTS,
    transform = None, target_transform = None, float_precision: str = 'float32', device: str = 'cpu',
    num_campaigns: int = 10, normalize_inputs: bool = True, normalize_outputs: bool = False,
) -> tuple[dict[int, CSVRegressionDataset], dict[int, CSVRegressionDataset], dict[int, CSVRegressionDataset]]:
    """
    Builds train/eval/test data for each experience from full pandas DataFrames.
    :param train_data: DataFrame with (all) train data.
    :param eval_data: DataFrame with (all) validation data.
    :param test_data: DataFrame with (all) test data.
    :param input_columns: Input columns, defaults to QUALIKIZ_HIGHPOW_INPUTS (15 columns for high-power experiments).
    :param output_columns: Output columns, defaults to QUALIKIZ_HIGHPOW_OUTPUTS (4 columns for high-power experiments).
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


def outputs_direction_report(model, inputs, targets, ef_columns=None, mult_factor: float = 1.0, zero_negs: bool = False):
    """
    Quick visualization of predicted and target outputs directions, with cosine similarity and negative outputs count.
    """
    predicted = model(inputs) * mult_factor
    if zero_negs:
        predicted[:, ef_columns][predicted[:, ef_columns] < 0] = 0
    mse = ((predicted - targets)**2).mean()
    cos_sim = cosine_similarity(predicted, targets).mean()
    ef_columns = ef_columns or [0, 1]
    ef_length = len(ef_columns)
    ef_targets, ef_predicted = targets[:, ef_columns], predicted[:, ef_columns]
    negatives = (ef_predicted < 0.0).int().sum(axis=0)
    print(
        f"The Mean Square Error is: {mse}",
        f"The Cosine Similarity is: {cos_sim}",
        f"Negative outputs are: {list(negatives)}",
        f"Negative percentages are: {list(negatives / len(ef_targets) * 100 / ef_length)}%",
        sep='\n', end='\n'
    )


def get_mean_std_metric_values(
    dataset, log_folder, mean_filename='eval_mean_values.csv', std_filename='eval_std_values.csv',
    metric='Forgetting_Exp', num_exp=10, include_future_experiences=False, absolute_weights: np.ndarray = None
):
    """
    Returns mean and std metric values for past and current experimental campaigns.
    """
    mean_file_path = os.path.join(log_folder, mean_filename)
    std_file_path = os.path.join(log_folder, std_filename)
    mean_df = pd.read_csv(mean_file_path)
    std_df = pd.read_csv(std_file_path)
    absolute_weights = absolute_weights if absolute_weights is not None else np.array([
        len(dataset[dataset.campaign == i]) for i in range(num_exp)
    ])
    mean_data = []
    std_data = []
    for i in range(num_exp):
        index = num_exp if include_future_experiences else i+1
        weights = absolute_weights[:index] / absolute_weights[:index].sum()
        exp_mean_series = mean_df[(mean_df['training_exp'] == i) & (mean_df['eval_exp'] < index)][metric].to_numpy()
        exp_std_series = std_df[num_exp*i:num_exp*i+index][metric].to_numpy()
        combined_mean = (exp_mean_series * weights[:index]).sum()
        exp_mean_series = (exp_mean_series - combined_mean)
        exp_std_series = exp_std_series**2 + exp_mean_series**2
        exp_std_series = weights[:index] * exp_std_series
        combined_std = exp_std_series.sum()
        mean_data.append(combined_mean)
        std_data.append(combined_std)
    return pd.DataFrame({
        'Experience': list(range(num_exp)),
        f'Mean {metric}': mean_data,
        f'Std {metric}': std_data
    })


def get_stat_metric_value_last_experience(
    log_folder, mean_filename='eval_mean_values.csv',
    std_filename='eval_std_values.csv', metric='Forgetting_Exp',
    stat: Literal["min", "max"] = "max"
):
    """
    Returns 
    """
    mean_file_path = os.path.join(log_folder, mean_filename)
    std_file_path = os.path.join(log_folder, std_filename)
    mean_df = pd.read_csv(mean_file_path)
    std_df = pd.read_csv(std_file_path)
    num_exp = len(mean_df['training_exp'].unique())
    last_exp = num_exp - 1
    last_exp_df = mean_df[mean_df.training_exp == last_exp]
    last_exp_arr = last_exp_df[metric].to_numpy()
    if stat == "max":
        index = np.argmax(last_exp_arr)
    elif stat == "min":
        index = np.argmin(last_exp_arr)
    else:
        raise ValueError(f"Unknown stat \"{stat}\"")
    return index, last_exp_arr[index].item()


def get_stat_metric_value_per_experience(
    log_folder: str, mean_filename='test_mean_values.csv',
    std_filename='test_std_values.csv', metric='Forgetting_Exp',
    stat: Literal["min", "max"] = "max"
) -> np.ndarray:
    mean_file_path = os.path.join(log_folder, mean_filename)
    std_file_path = os.path.join(log_folder, std_filename)
    mean_df = pd.read_csv(mean_file_path)
    std_df = pd.read_csv(std_file_path)
    mean_df['training_exp'] = mean_df['training_exp'].apply(int)
    std_df['training_exp'] = std_df['training_exp'].apply(int)
    if stat == "max":
        return mean_df.groupby('training_exp')[metric].apply(list).apply(lambda vals: vals.index(max(vals))).to_numpy()
    elif stat == "min":
        return mean_df.groupby('training_exp')[metric].apply(list).apply(lambda vals: vals.index(min(vals))).to_numpy()
    else:
        raise ValueError(f"Unknown stat \"{stat}\"")


def mean_std_df_wrapper(    
    logging_config: LoggingConfiguration, mean_filepath: str, std_filepath: str,
    metric='Forgetting_Exp', count=0, include_future_experiences=False,
):
    try:
        log_folder = logging_config.get_log_folder(count=count, task_id=0)
        pow_type, cluster_type, dataset_type, task, simulator_type = \
            logging_config.pow_type, logging_config.cluster_type, logging_config.dataset_type, \
            logging_config.task, logging_config.simulator_type
        _, eval_data, _ = load_baseline_csv_data(
            pow_type, cluster_type, dataset_type, raw_or_final='final',
            task=task, simulator_type=simulator_type
        )
        mean_std_df = get_mean_std_metric_values(
            eval_data, log_folder, mean_filename=mean_filepath, std_filename=std_filepath,
            metric=metric, include_future_experiences=include_future_experiences
        )
        return mean_std_df
    except Exception as ex:
        print(f"Directory {logging_config.get_log_folder(suffix=False, count=count, task_id=0)} not found")
        return None


# Comparing multiple strategies on a metric
def mean_std_strategy_plots_wrapper(
    logging_config: LoggingConfiguration, strategy_dicts: dict[str, str | tuple[str, str]],
    mean_filename: str, std_filename: str, internal_metric_name: str = 'Forgetting_Exp',
    plot_metric_name: str = 'Forgetting', count: int = 0, title: str = None,
    save: bool = False, savepath: str = None, show: bool = True, grid: bool = True,
    legend: bool = True, colors_and_linestyle_dict: dict[str, tuple[str, str]] = None,
    include_future_experiences: bool = False, include_std: bool = True
):
    """
    strategy_dicts = {Naive: Base}
    strategy_dicts = {Replay (2000): (Replay, Buffer 2000)}
    colors_and_linestyle_dict = {Replay (2000): ('red', '-')}
    """
    # Get mean_std_df for each strategy
    strategy_dfs = {}
    for strategy_metric_name, strategy_data in strategy_dicts.items():
        color, linestyle = colors_and_linestyle_dict[strategy_metric_name]
        if isinstance(strategy_data, str):
            extra_folder = strategy_data
            strategy_name = strategy_metric_name
        else:
            strategy_name = strategy_data[0]
            extra_folder = strategy_data[1]
        logging_config.strategy = strategy_name
        logging_config.extra_log_folder = extra_folder
        mean_std_df = mean_std_df_wrapper(
            logging_config, metric=internal_metric_name, count=count,
            mean_filepath=mean_filename, std_filepath=std_filename,
            include_future_experiences=include_future_experiences
        )
        if mean_std_df is not None:
            strategy_dfs[strategy_metric_name] = (mean_std_df, color, linestyle)
    # Plot metrics across strategies
    if strategy_dfs:
        plot_metric_over_multiple_strategies(
            strategy_dfs, grid=grid, legend=legend,
            show=show, save=save, savepath=savepath, 
            title=f"{plot_metric_name} Over Experiences" or title,
            xlabel="Experience", ylabel=plot_metric_name,
            include_std=include_std
        )


def mean_std_al_plots_wrapper(
    #logging_config: LoggingConfiguration, al_methods_dict: dict[str, tuple[str, str]],
    configs_and_dicts: list[tuple[LoggingConfiguration, dict[str, tuple[str, str]]]],
    mean_filename: str, std_filename: str, internal_metric_name: str = 'Forgetting_Exp',
    plot_metric_name: str = 'Forgetting', count: int = 0, title: str = None,
    save: bool = False, savepath: str = None, show: bool = True, grid: bool = True,
    legend: bool = True, colors_and_linestyle_dict: dict[str, tuple[str, str]] = None,
    pure_cl_strategy: str = None, pure_cl_extra_log_folder: str = None,
    include_future_experiences: bool = False, include_std: bool = True
):
    """
    al_methods_dict = {Random: random_sketch_grad}
    al_methods_dict = {Uncertainty: uncertainty_sketch_grad}
    colors_and_linestyle_dict = {Random: ('red', '-')}
    """
    al_method_dfs = {}
    for (logging_config, al_methods_dict) in configs_and_dicts:
        logging_config.active_learning = True
        # Get mean_std_df for each AL method
        if pure_cl_strategy is not None:
            al = logging_config.active_learning
            elf = logging_config.extra_log_folder
            logging_config.active_learning = False
            logging_config.extra_log_folder = pure_cl_extra_log_folder
            mean_std_df = mean_std_df_wrapper(
                logging_config, metric=internal_metric_name, count=count,
                mean_filepath=mean_filename, std_filepath=std_filename,
                include_future_experiences=include_future_experiences
            )
            if mean_std_df is not None:
                al_method_dfs[f"Pure {pure_cl_strategy}"] = (mean_std_df, 'blue', '-')
            logging_config.active_learning = al
            logging_config.extra_log_folder = elf

        for al_method_metric_name, (al_method, extra_log_folder) in al_methods_dict.items():
            color, linestyle = colors_and_linestyle_dict[al_method_metric_name]
            logging_config.al_method = al_method
            logging_config.extra_log_folder = extra_log_folder
            mean_std_df = mean_std_df_wrapper(
                logging_config, metric=internal_metric_name, count=count,
                mean_filepath=mean_filename, std_filepath=std_filename
            )
            if mean_std_df is not None:
                al_method_dfs[al_method_metric_name] = (mean_std_df, color, linestyle)
    # Plot metrics across AL methods
    if al_method_dfs:
        plot_metric_over_multiple_strategies(
            al_method_dfs, grid=grid, legend=legend,
            show=show, save=save, savepath=savepath,
            title=f"{plot_metric_name} Over Experiences" or title,
            xlabel="Experience", ylabel=plot_metric_name,
            include_std=include_std
        )


def mean_std_params_plots_wrapper(
    logging_config: LoggingConfiguration, param_name: str, params_list: list[int | float],
    strategy_name: str, strategy_label: str, extra_log_folder: str,
    mean_filename: str, std_filename: str, internal_metric_name: str = 'Forgetting_Exp',
    plot_metric_name: str = 'Forgetting', count: int = 0, title: str = None,
    save: bool = False, savepath: str = None, show: bool = True, grid: bool = True,
    legend: bool = True, colors_and_linestyle_list: list[tuple[str, str]] = None,
    include_future_experiences: bool = False, include_std: bool = True
):
    """
    param_name = 'batch_size'
    params_list = [512, 1024, 2048, 4096]
    strategy_name = 'Replay'
    strategy_label = 'Replay (2000)'
    extra_log_folder = 'Buffer 2000'
    strategy_dicts = {Naive: Base}
    strategy_dicts = {Replay (2000): (Replay, Buffer 2000)}
    colors_and_linestyle_list = [('red', '-'), ...]
    """
    # Get mean_std_df for each strategy
    strategy_dfs = {}
    logging_config_bak = copy.deepcopy(logging_config)
    logging_config.strategy = strategy_name
    logging_config.extra_log_folder = extra_log_folder
    for param_value, (color, linestyle) in zip(params_list, colors_and_linestyle_list):
        setattr(logging_config, param_name, param_value)
        mean_std_df = mean_std_df_wrapper(
            logging_config, metric=internal_metric_name, count=count,
            mean_filepath=mean_filename, std_filepath=std_filename,
            include_future_experiences=include_future_experiences
        )
        strategy_metric_name = f"{strategy_label} ({param_value} {param_name.replace('_', ' ')})"
        if mean_std_df is not None:
            strategy_dfs[strategy_metric_name] = (mean_std_df, color, linestyle)
    logging_config = logging_config_bak
    # Plot metrics across strategies
    if strategy_dfs:
        plot_metric_over_multiple_strategies(
            strategy_dfs, grid=grid, legend=legend,
            show=show, save=save, savepath=savepath, 
            title=f"{plot_metric_name} Over Experiences" or title,
            xlabel="Experience", ylabel=plot_metric_name,
            include_std=include_std
        )


def get_datasets_sizes_report(
    simulator_type: str = 'qualikiz', pow_type: str = 'highpow', cluster_type: str = 'tau_based',
    dataset_type: str = 'not_null', task: str = 'regression', verbose: bool = True,
):
    data_folder = os.path.join('data', simulator_type, 'cleaned', f"{pow_type}_cluster", cluster_type)
    # Complete, raw, final
    def get_csv_length(filepath: str) -> int:
        """Get number of rows in CSV without loading into memory"""
        with open(filepath) as f:
            # Subtract 1 to account for header row
            return sum(1 for _ in f) - 1

    train_filename = f'raw_train_data_{task}_{dataset_type}.csv'
    eval_filename = f'raw_eval_data_{task}_{dataset_type}.csv'
    test_filename = f'raw_test_data_{task}_{dataset_type}.csv'
    
    raw_train_size = get_csv_length(os.path.join(data_folder, train_filename))
    raw_eval_size = get_csv_length(os.path.join(data_folder, eval_filename)) 
    raw_test_size = get_csv_length(os.path.join(data_folder, test_filename))

    train_filename = f'final_train_data_{task}_{dataset_type}.csv'
    eval_filename = f'final_eval_data_{task}_{dataset_type}.csv'
    test_filename = f'final_test_data_{task}_{dataset_type}.csv'
    
    final_train_size = get_csv_length(os.path.join(data_folder, train_filename))
    final_eval_size = get_csv_length(os.path.join(data_folder, eval_filename))
    final_test_size = get_csv_length(os.path.join(data_folder, test_filename))

    complete_filename = f'complete_dataset.csv'
    complete_size = get_csv_length(os.path.join(data_folder, complete_filename))

    result = {
        'raw': {
            'train': raw_train_size,
            'eval': raw_eval_size,
            'test': raw_test_size
        },
        'final': {
            'train': final_train_size,
            'eval': final_eval_size,
            'test': final_test_size
        },
        'complete': complete_size
    }
    if verbose:
        print(f"Complete dataset has {complete_size} items.")
        for t, tname in zip(['raw', 'final'], ['Raw', 'Final']):
            for p, pname in zip(['train', 'eval', 'test'], ['Training', 'Validation', 'Test']):
                print(f"{tname} {pname} dataset has {result[t][p]} items.")
    return result


def mean_and_separated_plots(
    logging_config: LoggingConfiguration, internal_metric_name: str = 'Forgetting_Exp', plot_metric_name: str = 'Forgetting',
    count: int = 0, title: str = None, xlabel: str = 'Experimental Campaign', ylabel: str = None,
    save: bool = False, savepath: str = None, show: bool = True, grid: bool = True, legend: bool = True,
    mean_filename: str = 'test_mean_values.csv', std_filename: str = 'test_std_values.csv',
    include_std: bool = True, means: bool = True, weighted_means: bool = True
):
    log_folder = logging_config.get_log_folder(count=count)
    print(log_folder)
    df = pd.read_csv(f"{log_folder}/{mean_filename}")
    num_exp = len(df['training_exp'].unique())
    for i in range(num_exp):
        mask = (df['eval_exp'] == float(i)) & (df['training_exp'] >= i)
        target_df = df[mask][[internal_metric_name, 'training_exp']].reset_index(drop=True)
        plt.plot(target_df['training_exp'].astype(int), target_df[internal_metric_name], marker='o', linestyle='-', label=f"Campaign {i}")
    if means:
        if weighted_means:
            simulator_type, pow_type, cluster_type, dataset_type, task = logging_config.get_common_params(0, 5)
            absolute_weights = load_dataset_weights(
                simulator_type, pow_type, cluster_type, dataset_type, raw_or_final='final', task=task, weights_source='test'
            )
            mean_label = 'Weighted Mean'
        else:
            absolute_weights = np.ones(num_exp, dtype=np.intp)
            mean_label = 'Arithmetic Mean'
        mean_values, std_values = [], []
        for j in range(num_exp): # j = training experience
            mask = (df['training_exp'] == j) & (df['eval_exp'] <= j)
            current_data: np.ndarray[np.float64] = df[mask][internal_metric_name].reset_index(drop=True).to_numpy()
            relative_weights: np.ndarray[np.float64] = absolute_weights[:j+1] / absolute_weights[:j+1].sum()
            combined: np.ndarray[np.float64] = current_data * relative_weights
            mu = combined.sum()
            sigma = np.sqrt((relative_weights * np.square(current_data - mu)).sum())
            mean_values.append(mu)
            std_values.append(sigma)
        print(mean_values, std_values, sep='\n')
        mean_values_arr: np.ndarray[np.float64] = np.array(mean_values)
        std_values_arr: np.ndarray[np.float64] = np.array(std_values)
        plt.plot(np.arange(num_exp), mean_values_arr, marker='o', linestyle='-', label=mean_label, color='black')
        if include_std:
            plt.fill_between(np.arange(num_exp), mean_values_arr - std_values_arr, mean_values_arr + std_values_arr, alpha=0.2, color='black')
    plt.grid(grid)
    if legend: plt.legend(fontsize=10)
    if title: plt.title(title)
    if xlabel: plt.xlabel("Experimental Campaign")
    ylabel = ylabel if ylabel is not None else f"{plot_metric_name} Values"
    plt.ylabel(ylabel)
    if save and savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    if show: plt.show()


def mean_vs_weighted_mean_plots(
    logging_config: LoggingConfiguration, internal_metric_name: str = 'Forgetting_Exp', plot_metric_name: str = 'Forgetting',
    count: int = 0, title: str = None, xlabel: str = 'Experimental Campaign', ylabel: str = None,
    save: bool = False, savepath: str = None, show: bool = True, grid: bool = True, legend: bool = True,
    mean_filename: str = 'test_mean_values.csv', std_filename: str = 'test_std_values.csv', include_std: bool = True,
):
    log_folder = logging_config.get_log_folder(count=count)
    print(log_folder)
    df = pd.read_csv(f"{log_folder}/{mean_filename}")
    num_exp = len(df['training_exp'].unique())
    simulator_type, pow_type, cluster_type, dataset_type, task = logging_config.get_common_params(0, 5)
    
    weighted_absolute_weights = load_dataset_weights(
        simulator_type, pow_type, cluster_type, dataset_type, raw_or_final='final', task=task, weights_source='test'
    )
    arithmetic_absolute_weights = np.ones(num_exp)

    weighted_mean_label = 'Weighted Mean'
    weighted_mean_values, weighted_std_values = [], []
    arithmetic_mean_label = 'Arithmetic Mean'
    arithmetic_mean_values, arithmetic_std_values = [], []

    for j in range(num_exp): # j = training experience
        mask = (df['training_exp'] == j) & (df['eval_exp'] <= j)
        current_data: np.ndarray[np.float64] = df[mask][internal_metric_name].reset_index(drop=True).to_numpy()

        weighted_relative_weights: np.ndarray[np.float64] = weighted_absolute_weights[:j+1] / weighted_absolute_weights[:j+1].sum()
        arithmetic_relative_weights: np.ndarray[np.float64] = arithmetic_absolute_weights[:j+1] / arithmetic_absolute_weights[:j+1].sum()
        
        weighted_combined: np.ndarray[np.float64] = current_data * weighted_relative_weights
        arithmetic_combined: np.ndarray[np.float64] = current_data * arithmetic_relative_weights
        
        weighted_mu, arithmetic_mu = weighted_combined.sum(), arithmetic_combined.sum()
        weighted_sigma = np.sqrt((weighted_relative_weights * np.square(current_data - weighted_mu)).sum())
        arithmetic_sigma = np.sqrt((arithmetic_relative_weights * np.square(current_data - arithmetic_mu)).sum())

        weighted_mean_values.append(weighted_mu)
        weighted_std_values.append(weighted_sigma)

        arithmetic_mean_values.append(arithmetic_mu)
        arithmetic_std_values.append(arithmetic_sigma)
    
    weighted_mean_values_arr: np.ndarray[np.float64] = np.array(weighted_mean_values)
    weighted_std_values_arr: np.ndarray[np.float64] = np.array(weighted_std_values)

    arithmetic_mean_values_arr: np.ndarray[np.float64] = np.array(arithmetic_mean_values)
    arithmetic_std_values_arr: np.ndarray[np.float64] = np.array(arithmetic_std_values)

    plt.plot(np.arange(num_exp), weighted_mean_values_arr, marker='o', linestyle='-', label=weighted_mean_label, color='blue')
    if include_std:
        left = weighted_mean_values_arr - weighted_std_values_arr
        right = weighted_mean_values_arr + weighted_std_values_arr
        plt.fill_between(np.arange(num_exp), left, right, alpha=0.2, color='blue')
    
    plt.plot(np.arange(num_exp), arithmetic_mean_values_arr, marker='o', linestyle='-', label=arithmetic_mean_label, color='red')    
    if include_std:
        left = arithmetic_mean_values_arr - arithmetic_std_values_arr
        right = arithmetic_mean_values_arr + arithmetic_std_values_arr
        plt.fill_between(np.arange(num_exp), left, right, alpha=0.2, color='red')
    
    plt.grid(grid)
    if legend: plt.legend(fontsize=8)
    if title: plt.title(title)
    if xlabel: plt.xlabel("Experimental Campaign")
    ylabel = ylabel if ylabel is not None else f"{plot_metric_name} Values"
    plt.ylabel(ylabel)
    if save and savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    if show: plt.show()


def get_training_times(
    config: LoggingConfiguration, num_tasks: int = 4
) -> tuple[np.ndarray[np.float64], float]:
    all_times = []
    all_sums = []
    for task_id in range(num_tasks):
        log_folder = config.get_log_folder(count=-1, task_id=task_id)
        df = pd.read_csv(os.path.join(log_folder, "training_results_epoch.csv"))
        times_array = df.groupby('training_exp')['Time_Epoch'].apply(lambda g: g.sum()).to_numpy()
        all_times.append(times_array)
        all_sums.append(times_array.sum().item())
    all_means: np.ndarray = np.array(all_times).mean(axis=0)
    final_mean: float = np.array(all_sums).mean().item()
    return all_means, final_mean


def get_num_epochs(config: LoggingConfiguration, num_tasks: int = 4):
    all_times = []
    all_sums = []
    for task_id in range(num_tasks):
        log_folder = config.get_log_folder(count=-1, task_id=task_id)
        df = pd.read_csv(os.path.join(log_folder, "training_results_epoch.csv"))
        times_array = df.groupby('training_exp')['epoch'].apply(lambda g: len(g)).to_numpy()
        all_times.append(times_array)
        all_sums.append(times_array.sum().item())
    all_means: np.ndarray = np.array(all_times).mean(axis=0)
    final_mean: float = np.array(all_sums).mean().item()
    return all_means, final_mean


def computeR_from_config(
    config: LoggingConfiguration, naive_values: pd.DataFrame, cumulative_values: pd.DataFrame,
    mean_filename: str = 'test_mean_values.csv', std_filename: str = 'test_std_values.csv'
) -> np.ndarray:
    metric = "Mean R2Score_Exp"
    simulator_type, pow_type, cluster_type, dataset_type, task = config.get_common_params(end=5)
    absolute_weights: np.ndarray = load_dataset_weights(
        simulator_type, pow_type, cluster_type, dataset_type,
        raw_or_final='final', task=task, weights_source='test'
    )
    strategy_values = get_mean_std_metric_values(
        None, config.get_log_folder(count=-1), mean_filename, std_filename,
        metric[5:], absolute_weights=absolute_weights
    )
    result = (strategy_values[metric] - naive_values[metric]) / (cumulative_values[metric] - naive_values[metric])
    return result.to_numpy()


def computeT_from_config(config: LoggingConfiguration, naive_time: float, num_tasks: int = 4) -> float:
    per_exp_times, total_time = get_training_times(config, num_tasks=num_tasks)
    return total_time / naive_time


__all__ = [
    'load_models', 'load_baseline_csv_data', 'extract_dataset_weights', 'load_dataset_weights',
    'load_complete_dataset', 'build_full_datasets', 'build_experience_datasets',
    'outputs_direction_report', 'get_mean_std_metric_values', 'get_stat_metric_value_last_experience',
    'get_stat_metric_value_per_experience', 'mean_std_df_wrapper', 'mean_std_strategy_plots_wrapper',
    'mean_std_al_plots_wrapper', 'get_datasets_sizes_report', 'mean_std_params_plots_wrapper',
    'mean_and_separated_plots', 'mean_vs_weighted_mean_plots', 'get_training_times',
    'computeR_from_config', 'computeT_from_config', 'get_num_epochs'
]