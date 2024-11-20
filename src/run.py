import json
import sys
import os
import shutil

import pandas as pd
from datetime import datetime

import torch
from sklearn.model_selection import train_test_split
from avalanche.logging import InteractiveLogger
from torch.nn import MSELoss, BCEWithLogitsLoss, HuberLoss
from torch.optim import SGD, Adam

from avalanche.benchmarks import *
from avalanche.evaluation.metrics import loss_metrics, forgetting_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, Replay, LwF, EWC, SynapticIntelligence
from avalanche.training.plugins import EarlyStoppingPlugin as AvalancheEarlyStopping
from avalanche.training.plugins import LRSchedulerPlugin
from torch.optim.lr_scheduler import StepLR

from utils import *


def load_config(filename):
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse '{filename}' as JSON.")
        sys.exit(1)


def _build_normalization_transforms(data, columns, dtype, transform=None):
    mean = torch.tensor(data[columns].mean(axis=0), dtype=dtype)
    std = torch.tensor(data[columns].std(axis=0), dtype=dtype)
    print(f"mean = {mean}, std = {std} ...")
    if transform is not None:
        transform = transforms.Compose([CustomNormalize(mean, std), transform])
    else:
        transform = CustomNormalize(mean, std)
    return transform, (mean, std)


@time_logger()
def make_benchmark(
        csv_file, train_datasets, test_datasets, task='regression', NUM_CAMPAIGNS=5, dtype='float64',
        test_size=0.2, normalize_inputs=False, normalize_outputs=False, log_folder=None,
):
    float_precision = dtype
    dtype = get_dtype_from_str(dtype)
    OUTPUTS = BASELINE_HIGHPOW_OUTPUTS if task == 'regression' else ['has_turbulence'] #Try with single column
    data = pd.read_csv(csv_file)
    debug_print(f"There are {len(data)} items in the dataset in {csv_file}.")
    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    if normalize_inputs:
        transform, (mean, std) = _build_normalization_transforms(train_data, BASELINE_HIGHPOW_INPUTS, dtype)
        if log_folder:
            torch.save(mean, os.path.join(log_folder, "input_mean.pt"))
            torch.save(std, os.path.join(log_folder, "input_std.pt"))
    else:
        transform = None
    if normalize_outputs:
        target_transform, (mean, std) = _build_normalization_transforms(train_data, OUTPUTS, dtype)
        if log_folder:
            torch.save(mean, os.path.join(log_folder, "output_mean.pt"))
            torch.save(std, os.path.join(log_folder, "output_std.pt"))
    else:
        target_transform = None
    for campaign in range(NUM_CAMPAIGNS):
        print(f"Loading data for campaign {campaign} ...")
        train_dataset, test_dataset = get_avalanche_csv_regression_datasets(
            train_data, test_data, BASELINE_HIGHPOW_INPUTS, output_columns=OUTPUTS,
            filter_by={'campaign': [campaign]}, float_precision=float_precision,
            device='cpu', transform=transform, target_transform=target_transform,
        )
        X, y = train_dataset[0]
        print(
            f"Input Shape = {X.shape}",
            f"Output Shape = {y.shape}",
            f"Length of Train Dataset = {len(train_dataset)}",
            f"Length of Test Dataset = {len(test_dataset)}"
        )
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    benchmark = benchmark_from_datasets(train=train_datasets, test=test_datasets)
    return benchmark


def main():
    # Config
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model
    input_size = len(BASELINE_HIGHPOW_INPUTS)
    output_size = len(BASELINE_HIGHPOW_OUTPUTS)
    train_mb_size = 512
    eval_mb_size = 2048
    train_epochs = 30
    num_campaigns = 10
    dtype='float64'
    lr = 5e-4
    momentum = 0.9
    weight_decay = 2e-4
    drop_rate = 0.5
    out_channels1 = 16
    out_channels2 = 32
    hidden_size = 128
    hidden_layers = 2
    kernel_size = 3
    padding = 1
    alpha = 0.5
    temperature = 1
    lambda_value = 1
    decay_factor = 0.8
    si_lambda = 1.0
    si_eps = 1e-5
    pow_type = 'highpow'
    cluster_type = 'Ip_Pin_based'
    optimizer_type = 'SGD'
    dataset_type = 'not_null' #'complete' # complete dataset (for classification or full regression), or not_null (for regression)
    task = 'regression'
    mem_size = 1000
    strategy = 'Naive'
    model_type = 'MLP'
    normalize_inputs = False
    normalize_outputs = False
    early_stopping = False
    early_stopping_patience = 5
    early_stopping_delta = 0.1
    loss_type = 'MSE'
    scheduler_type = None
    scheduler_first_exp_only = False
    config = {
        'input_size': input_size,
        'output_size': output_size,
        'train_mb_size': train_mb_size,
        'eval_mb_size': eval_mb_size,
        'train_epochs': train_epochs,
        'num_campaigns': num_campaigns,
        'dtype': dtype,
        'lr': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'drop_rate': drop_rate,
        'out_channels1': out_channels1,
        'out_channels2': out_channels2,
        'hidden_size': hidden_size,
        'hidden_layers': hidden_layers,
        'kernel_size': kernel_size,
        'padding': padding,
        'alpha': alpha,
        'temperature': temperature,
        'lambda': lambda_value,
        'decay_factor': decay_factor,
        'si_lambda': si_lambda,
        'si_eps': si_eps,
        'pow_type': pow_type,
        'cluster_type': cluster_type,
        'optimizer_type': optimizer_type,
        'dataset_type': dataset_type,
        'task': task,
        'mem_size': mem_size,
        'strategy': strategy,
        'model_type': model_type,
        'normalize_inputs': normalize_inputs,
        'normalize_outputs': normalize_outputs,
        'early_stopping': early_stopping,
        'early_stopping_patience': early_stopping_patience,
        'early_stopping_delta': early_stopping_delta,
        'loss_type': loss_type,
        'scheduler_type': scheduler_type,
        'scheduler_first_epoch_only': scheduler_first_exp_only,
    }
    if len(sys.argv) >= 2:
        config_file = sys.argv[1]
        print(config_file)
        new_config_data = load_config(config_file)
        config.update(new_config_data)
    input_size = config['input_size']
    output_size = config['output_size']
    train_mb_size = config['train_mb_size']
    eval_mb_size = config['eval_mb_size']
    train_epochs = config['train_epochs']
    num_campaigns = config['num_campaigns']
    dtype = config['dtype']
    lr = config['lr']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    drop_rate = config['drop_rate']
    out_channels1 = config['out_channels1']
    out_channels2 = config['out_channels2']
    hidden_size = config['hidden_size']
    hidden_layers = config['hidden_layers']
    kernel_size = config['kernel_size']
    padding = config['padding']
    alpha = config['alpha']
    temperature = config['temperature']
    lambda_value = config['lambda']
    decay_factor = config['decay_factor']
    si_lambda = config['si_lambda']
    si_eps = config['si_eps']
    pow_type = config['pow_type']
    cluster_type = config['cluster_type']
    optimizer_type = config['optimizer_type']
    dataset_type = config['dataset_type']
    task = config['task']
    mem_size = config['mem_size']
    strategy = config['strategy']
    model_type = config['model_type']
    normalize_inputs = config['normalize_inputs']
    normalize_outputs = config['normalize_outputs']
    early_stopping = config['early_stopping']
    early_stopping_patience = config['early_stopping_patience']
    early_stopping_delta = config['early_stopping_delta']
    loss_type = config['loss_type']
    scheduler_type = config['scheduler_type']
    scheduler_first_exp_only = config['scheduler_first_epoch_only']
    try:
        print("Configuration Loaded:")
        print(f"  Input size: {input_size}")
        print(f"  Output size: {output_size}")
        print(f"  Train MB size: {train_mb_size}")
        print(f"  Eval MB size: {eval_mb_size}")
        print(f"  Train epochs: {train_epochs}")
        print(f"  Num campaigns: {num_campaigns}")
        print(f"  Dtype: {dtype}")
        print(f"  LR: {lr}")
        print(f"  Momentum: {momentum}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Drop rate: {drop_rate}")
        print(f"  Out channels 1: {out_channels1}")
        print(f"  Out channels 2: {out_channels2}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  Kernel size: {kernel_size}")
        print(f"  Padding: {padding}")
        print(f"  Alpha: {alpha}")
        print(f"  Temperature: {temperature}")
        print(f"  Lambda: {lambda_value}")
        print(f"  Decay factor: {decay_factor}")
        print(f"  Si-Lambda: {si_lambda}")
        print(f"  Si-Eps: {si_eps}")
        print(f"  Pow type: {pow_type}")
        print(f"  Cluster type: {cluster_type}")
        print(f"  Optimizer type: {optimizer_type}")
        print(f"  Dataset type: {dataset_type}")
        print(f"  Task: {task}")
        print(f"  Mem size: {mem_size}")
        print(f"  Strategy: {strategy}")
        print(f"  Model type: {model_type}")
        print(f"  Normalize inputs: {normalize_inputs}")
        print(f"  Normalize outputs: {normalize_outputs}")
        print(f"  Early stopping: {early_stopping}")
        print(f"  Early stopping_patience: {early_stopping_patience}")
        print(f"  Early stopping delta: {early_stopping_delta}")
        print(f"  Loss Type: {loss_type}")
        print(f"  Scheduler Type: {scheduler_type}")
        print(f"  Scheduler First Epoch Only: {scheduler_first_exp_only}")
    except NameError as e:
        print(f"Error: Missing expected configuration key: \"{e}\"")
        sys.exit(1)

    if task == 'regression':
        if model_type == 'ConvNet':
            model = SimpleConv1DModel(
                in_features=input_size, out_channels1=out_channels1, out_channels2=out_channels2,
                hidden_size=hidden_size, out_features=output_size, kernel_size=kernel_size,
                padding=padding, dtype=dtype,
            )
        elif model_type == 'MLP':
            model = SimpleRegressionMLP(
                output_size=output_size, hidden_size=hidden_size, dtype=dtype,
                hidden_layers=hidden_layers, drop_rate=drop_rate,
            )
    elif task == 'classification':
        model = SimpleClassificationMLP(
            num_classes=1, input_size=input_size, hidden_size=hidden_size,
            hidden_layers=hidden_layers, dtype=dtype, drop_rate=drop_rate,
        )
    else:
        raise ValueError(f"Unknown task type '{task}'")

    # Print Model Size
    trainables, total = get_model_size(model)
    print(
        f"Trainable Parameters = {trainables}"
        f"\nTotal Parameters = {total}"
    )

    # Prepare folders for experiments
    folder_name = \
        (f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} "
         f"({model_type} early_stopping = {early_stopping})")
    log_folder = os.path.join('logs', pow_type, cluster_type, task, dataset_type, strategy, folder_name)
    os.makedirs(os.path.join(log_folder), exist_ok=True)

    # Print model size to experiment directory
    with open(f'{log_folder}/model_size.txt', 'w') as fp:
        print(
            f"Trainable Parameters = {trainables}"
            f"\nTotal Parameters = {total}",
            file=fp
        )

    train_datasets = []
    test_datasets = []
    csv_file = f'data/baseline/cleaned/{pow_type}_cluster/{cluster_type}/{dataset_type}_dataset.csv'
    benchmark = make_benchmark(
        csv_file, train_datasets, test_datasets, task=task, NUM_CAMPAIGNS=num_campaigns, dtype=dtype,
        normalize_inputs=normalize_inputs, normalize_outputs=normalize_outputs, log_folder=log_folder,
    )

    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    if optimizer_type == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print(f"Error: Unsupported optimizer type '{optimizer_type}'.")
        sys.exit(1)
    with open(os.path.join(log_folder, 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=4)
    # Define loss
    if task == 'regression':
        if loss_type == 'MSE':
            criterion = MSELoss(reduction='mean') #if task == 'regression' else BCEWithLogitsLoss()
        elif loss_type == 'Huber':
            criterion = HuberLoss(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type '{loss_type}'")
    else: # classification
        criterion = BCEWithLogitsLoss()
    # forgetting_metrics(experience=True, stream=True) + \
    metrics = \
        loss_metrics(epoch=True, experience=True, stream=True) + \
        relative_distance_metrics(epoch=True, experience=True, stream=True) + \
        r2_score_metrics(epoch=True, experience=True, stream=True)
    # Build logger
    csv_logger = CustomCSVLogger(log_folder=log_folder, metrics=metrics)
    has_interactive_logger = int(os.getenv('INTERACTIVE', '0'))
    loggers = ([InteractiveLogger()] if has_interactive_logger else []) + [csv_logger]
    # Define the evaluation plugin with desired metrics
    if task == 'regression':
        eval_plugin = EvaluationPlugin(*metrics, loggers=loggers)
    else:
        eval_plugin = EvaluationPlugin(*metrics, loggers=loggers)

    # Extra plugins
    plugins = []
    if early_stopping:
        plugins.append(
            EarlyStoppingPlugin(
                patience=early_stopping_patience, delta=early_stopping_delta,
                restore_best_weights=True, metric='Loss', type='min',
            )
        )
    if scheduler_type == 'StepLR':
        plugins.append(
            LRSchedulerPlugin(
                StepLR(optimizer, step_size=30, gamma=0.8),
                first_exp_only=scheduler_first_exp_only,
            )
        )

    # Continual learning strategy
    if strategy == 'Naive':
        cl_strategy = Naive(
            model=model, optimizer=optimizer, criterion=criterion, train_mb_size=train_mb_size,
            train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, evaluator=eval_plugin,
            plugins=plugins,
        )
    elif strategy == 'Replay':
        cl_strategy = Replay(
            model=model, optimizer=optimizer, criterion=criterion, mem_size=mem_size, train_mb_size=train_mb_size,
            train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, evaluator=eval_plugin,
            plugins=plugins,
        )
    elif strategy == 'LwF':
        cl_strategy = LwF(
            model=model, optimizer=optimizer, criterion=criterion, alpha=alpha, temperature=temperature,
            train_mb_size=train_mb_size, train_epochs=train_epochs, eval_mb_size=eval_mb_size,
            device=device, evaluator=eval_plugin,
            plugins=plugins,
        )
    elif strategy == 'EWC':
        if decay_factor >= 0.0:
            cl_strategy = EWC(
                model=model, optimizer=optimizer, criterion=criterion, ewc_lambda=lambda_value,
                decay_factor=decay_factor, train_mb_size=train_mb_size, train_epochs=train_epochs,
                eval_mb_size=eval_mb_size, device=device, evaluator=eval_plugin, mode='online',
                plugins=plugins,
            )
        else:
            cl_strategy = EWC(
                model=model, optimizer=optimizer, criterion=criterion, ewc_lambda=lambda_value,
                train_mb_size=train_mb_size, train_epochs=train_epochs,
                eval_mb_size=eval_mb_size, device=device, evaluator=eval_plugin,
                plugins=plugins,
            )
    elif strategy == 'SI':
        cl_strategy = SynapticIntelligence(
            model=model, optimizer=optimizer, criterion=criterion, si_lambda=si_lambda,
            eps=si_eps, train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, evaluator=eval_plugin,
            plugins=plugins,
        )
    else:
        raise ValueError(f"Unknown strategy type '{strategy}'")

    debug_print(cl_strategy.evaluator)
    debug_print(cl_strategy.plugins)

    @time_logger(log_file=f'{log_folder}/timing.txt')
    def run(train_stream, test_stream, cl_strategy, model, log_folder):
        results = []
        for idx, train_exp in enumerate(train_stream):
            print(f"Starting training experience {idx}: ")
            cl_strategy.train(train_exp)
            print(f"Starting testing experience {idx}: ")
            results.append(cl_strategy.eval(test_stream))
            print(f"Saving model after experience {id}: ")
            model.eval()
            torch.save(model.state_dict(), os.path.join(log_folder, f'model_after_exp_{idx}.pt'))
            model.train()
        return results

    # train and test loop over the stream of experiences
    print("Starting ...")
    try:
        results = run(train_stream, test_stream, cl_strategy, model, log_folder)
        with open(os.path.join(log_folder, 'results.json'), 'w') as fp:
            json.dump(results, fp, indent=4)

        model.eval()
        # Save model for future usage
        torch.save(model.state_dict(), os.path.join(log_folder, 'model.pt'))
        # Plot over first 5 experiences
        for metric, title, ylabel in zip(
            ['Loss_Exp', 'R2Score_Exp'], ['Loss over each experience', 'R2 Score over each experience'],
            ['Loss', 'R2 Score']
        ):
            plot_metric_over_evaluation_experiences(
                os.path.join(log_folder, 'eval_results_experience.csv'), metric,
                title, 'Training Experience', ylabel, show=False, start_exp=0, end_exp=4,
                savepath=os.path.join(log_folder, f'plot_of_first_5_experiences_{metric[:-4]}.png'),
            )
            # Plot over all experiences
            plot_metric_over_evaluation_experiences(
                os.path.join(log_folder, 'eval_results_experience.csv'), metric,
                title, 'Training Experience', ylabel, show=False, start_exp=0, end_exp=9,
                savepath=os.path.join(log_folder, f'plot_of_all_10_experiences_{metric[:-4]}.png'),
            )
    except Exception as ex:
        debug_print("Caught Exception: ", ex)
        shutil.rmtree(log_folder)


if __name__ == '__main__':
    main()
