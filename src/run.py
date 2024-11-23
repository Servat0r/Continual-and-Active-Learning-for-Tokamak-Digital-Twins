import json
import sys
import os
import shutil
from rich import print

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
from avalanche.training.plugins import LRSchedulerPlugin
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from utils import *


def load_config(filename):
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"[red]Error: Configuration file '{filename}' not found.[/red]")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[red]Error: Failed to parse '{filename}' as JSON.[/red]")
        sys.exit(1)


def _build_normalization_transforms(data, columns, dtype, transform=None):
    mean = torch.tensor(data[columns].mean(axis=0), dtype=dtype)
    std = torch.tensor(data[columns].std(axis=0), dtype=dtype)
    print(f"[green]mean = {mean}, std = {std} ...[/green]")
    if transform is not None:
        transform = transforms.Compose([CustomNormalize(mean, std), transform])
    else:
        transform = CustomNormalize(mean, std)
    return transform, (mean, std)


@time_logger()
def make_benchmark(
        csv_file, train_datasets, test_datasets, task='regression',
        input_columns=BASELINE_HIGHPOW_INPUTS, output_columns=BASELINE_HIGHPOW_OUTPUTS,
        NUM_CAMPAIGNS=5, dtype='float64', *, test_size=0.2, eval_size=0.25,
        normalize_inputs=False, normalize_outputs=False, log_folder=None,
):
    float_precision = dtype
    dtype = get_dtype_from_str(dtype)
    if task == 'classification':
        output_columns = ['has_turbulence']
    elif output_columns is None:
        output_columns = BASELINE_HIGHPOW_OUTPUTS
    data = pd.read_csv(csv_file)
    debug_print(f"There are {len(data)} items in the dataset in {csv_file}.")
    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42, shuffle=True)
    if normalize_inputs:
        transform, (mean, std) = _build_normalization_transforms(train_data, input_columns, dtype)
        if log_folder:
            torch.save(mean, os.path.join(log_folder, "input_mean.pt"))
            torch.save(std, os.path.join(log_folder, "input_std.pt"))
    else:
        transform = None
    if normalize_outputs:
        target_transform, (mean, std) = _build_normalization_transforms(train_data, output_columns, dtype)
        if log_folder:
            torch.save(mean, os.path.join(log_folder, "output_mean.pt"))
            torch.save(std, os.path.join(log_folder, "output_std.pt"))
    else:
        target_transform = None
    for campaign in range(NUM_CAMPAIGNS):
        print(f"[yellow]Loading data for campaign {campaign} ...[/yellow]")
        train_dataset, test_dataset = get_avalanche_csv_regression_datasets(
            train_data, test_data, BASELINE_HIGHPOW_INPUTS, output_columns=output_columns,
            filter_by={'campaign': [campaign]}, float_precision=float_precision,
            device='cpu', transform=transform, target_transform=target_transform,
        )
        X, y = train_dataset[0]
        print(
            f"[red]Input Shape = {X.shape}[/red]",
            f"[red]Output Shape = {y.shape}[/red]",
            f"[red]Length of Train Dataset = {len(train_dataset)}[/red]",
            f"[red]Length of Test Dataset = {len(test_dataset)}[/red]"
        )
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    benchmark = benchmark_from_datasets(train=train_datasets, test=test_datasets)
    return benchmark


def main():
    # Config
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model
    input_columns = BASELINE_HIGHPOW_INPUTS
    output_columns = BASELINE_HIGHPOW_OUTPUTS
    input_size = len(input_columns)
    output_size = len(output_columns)
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
    scheduler_step_size = 30
    scheduler_gamma = 0.8
    scheduler_eta_min = 1e-4
    config = {
        'input_columns': input_columns,
        'output_columns': output_columns,
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
        'scheduler_step_size': scheduler_step_size,
        'scheduler_gamma': scheduler_gamma,
        'scheduler_eta_min': scheduler_eta_min,
    }
    if len(sys.argv) >= 2:
        config_file = sys.argv[1]
        print(config_file)
        new_config_data = load_config(config_file)
        config.update(new_config_data)
    input_columns = config['input_columns']
    output_columns = config['output_columns']
    input_size = len(input_columns)
    output_size = len(output_columns)
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
    scheduler_step_size = config['scheduler_step_size']
    scheduler_gamma = config['scheduler_gamma']
    scheduler_eta_min = config['scheduler_eta_min']
    try:
        print("[cyan]Configuration Loaded:[/cyan]")
        print(f"  [cyan]Input columns: {input_columns}[/cyan]")
        print(f"  [cyan]Output columns: {output_columns}[/cyan]")
        print(f"  [cyan]Input size: {input_size}[/cyan]")
        print(f"  [cyan]Output size: {output_size}[/cyan]")
        print(f"  [cyan]Train MB size: {train_mb_size}[/cyan]")
        print(f"  [cyan]Eval MB size: {eval_mb_size}[/cyan]")
        print(f"  [cyan]Train epochs: {train_epochs}[/cyan]")
        print(f"  [cyan]Num campaigns: {num_campaigns}[/cyan]")
        print(f"  [cyan]Dtype: {dtype}[/cyan]")
        print(f"  [cyan]LR: {lr}[/cyan]")
        print(f"  [cyan]Momentum: {momentum}[/cyan]")
        print(f"  [cyan]Weight decay: {weight_decay}[/cyan]")
        print(f"  [cyan]Drop rate: {drop_rate}[/cyan]")
        print(f"  [cyan]Out channels 1: {out_channels1}[/cyan]")
        print(f"  [cyan]Out channels 2: {out_channels2}[/cyan]")
        print(f"  [cyan]Hidden size: {hidden_size}[/cyan]")
        print(f"  [cyan]Hidden layers: {hidden_layers}[/cyan]")
        print(f"  [cyan]Kernel size: {kernel_size}[/cyan]")
        print(f"  [cyan]Padding: {padding}[/cyan]")
        print(f"  [cyan]Alpha: {alpha}[/cyan]")
        print(f"  [cyan]Temperature: {temperature}[/cyan]")
        print(f"  [cyan]Lambda: {lambda_value}[/cyan]")
        print(f"  [cyan]Decay factor: {decay_factor}[/cyan]")
        print(f"  [cyan]Si-Lambda: {si_lambda}[/cyan]")
        print(f"  [cyan]Si-Eps: {si_eps}[/cyan]")
        print(f"  [cyan]Pow type: {pow_type}[/cyan]")
        print(f"  [cyan]Cluster type: {cluster_type}[/cyan]")
        print(f"  [cyan]Optimizer type: {optimizer_type}[/cyan]")
        print(f"  [cyan]Dataset type: {dataset_type}[/cyan]")
        print(f"  [cyan]Task: {task}[/cyan]")
        print(f"  [cyan]Mem size: {mem_size}[/cyan]")
        print(f"  [cyan]Strategy: {strategy}[/cyan]")
        print(f"  [cyan]Model type: {model_type}[/cyan]")
        print(f"  [cyan]Normalize inputs: {normalize_inputs}[/cyan]")
        print(f"  [cyan]Normalize outputs: {normalize_outputs}[/cyan]")
        print(f"  [cyan]Early stopping: {early_stopping}[/cyan]")
        print(f"  [cyan]Early stopping_patience: {early_stopping_patience}[/cyan]")
        print(f"  [cyan]Early stopping delta: {early_stopping_delta}[/cyan]")
        print(f"  [cyan]Loss Type: {loss_type}[/cyan]")
        print(f"  [cyan]Scheduler Type: {scheduler_type}[/cyan]")
        print(f"  [cyan]Scheduler First Epoch Only: {scheduler_first_exp_only}[/cyan]")
        print(f"  [cyan]Scheduler Step Size: {scheduler_step_size}[/cyan]")
        print(f"  [cyan]Scheduler Gamma: {scheduler_gamma}[/cyan]")
        print(f"  [cyan]Scheduler Eta Min: {scheduler_eta_min}[/cyan]")
    except NameError as e:
        print(f"Error: Missing expected configuration key: \"{e}\"")
        sys.exit(1)

    if task == 'regression':
        if model_type == 'ConvNet':
            if loss_type in ['MSE', 'Huber']:
                model = SimpleConv1DModel(
                    in_features=input_size, out_channels1=out_channels1, out_channels2=out_channels2,
                    hidden_size=hidden_size, out_features=output_size, kernel_size=kernel_size,
                    padding=padding, dtype=dtype,
                )
            else:
                raise ValueError(f"Invalid model_type = {model_type} for loss type = {loss_type}")
        elif model_type == 'MLP':
            if loss_type in ['MSE', 'Huber']:
                model = SimpleRegressionMLP(
                    output_size=output_size, hidden_size=hidden_size, dtype=dtype,
                    hidden_layers=hidden_layers, drop_rate=drop_rate,
                )
            elif loss_type == 'GaussianNLL':
                model = GaussianRegressionMLP(
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
        f"Trainable Parameters = [red]{trainables}[/red]"
        f"\nTotal Parameters = [red]{total}[/red]"
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
        csv_file, train_datasets, test_datasets, task=task, NUM_CAMPAIGNS=num_campaigns,
        dtype=dtype, normalize_inputs=normalize_inputs, normalize_outputs=normalize_outputs,
        log_folder=log_folder, input_columns=input_columns, output_columns=output_columns,
    )

    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    if optimizer_type == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print(f"[red]Error: Unsupported optimizer type '{optimizer_type}'.[/red]")
        sys.exit(1)
    with open(os.path.join(log_folder, 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=4)
    # Define loss
    if task == 'regression':
        if loss_type == 'MSE':
            criterion = MSELoss(reduction='mean') #if task == 'regression' else BCEWithLogitsLoss()
        elif loss_type == 'Huber':
            criterion = HuberLoss(reduction='mean')
        elif loss_type == 'GaussianNLL':
            criterion = GaussianNLLLoss(reduction='mean')
        else:
            raise ValueError(f"[red]Unsupported loss type '{loss_type}'[/red]")
    else: # classification
        criterion = BCEWithLogitsLoss()
    # forgetting_metrics(experience=True, stream=True) + \
    if loss_type == 'GaussianNLL':
        metrics = \
            loss_metrics(epoch=True, experience=True, stream=True) + \
            gaussian_mse_metrics(epoch=True, experience=True, stream=True)
    else:
        metrics = \
            loss_metrics(epoch=True, experience=True, stream=True) + \
            relative_distance_metrics(epoch=True, experience=True, stream=True) + \
            r2_score_metrics(epoch=True, experience=True, stream=True)
    # Build logger
    csv_logger = CustomCSVLogger(log_folder=log_folder, metrics=metrics, val_stream=test_stream)
    has_interactive_logger = int(os.getenv('INTERACTIVE', '0'))
    loggers = ([InteractiveLogger()] if has_interactive_logger else []) + [csv_logger]
    # Define the evaluation plugin with desired metrics
    if task == 'regression':
        eval_plugin = EvaluationPlugin(*metrics, loggers=loggers)
    else:
        eval_plugin = EvaluationPlugin(*metrics, loggers=loggers)

    # Extra plugins
    plugins = [
        ValidationStreamPlugin(val_stream=test_stream),
        TqdmTrainingEpochsPlugin(num_exp=num_campaigns, num_epochs=train_epochs)
    ]

    if early_stopping:
        #plugins.append(
            #AvalancheEarlyStopping(
            #    patience=early_stopping_patience, val_stream_name='test_stream',
            #    metric_name='Loss_Epoch', mode='min', peval_mode='epoch',
            #    margin=early_stopping_delta, verbose=True
            #)
        #)
        plugins.append(
            ValidationEarlyStoppingPlugin(
                patience=early_stopping_patience, val_stream_name='test_stream',
                metric='Loss', type='min', restore_best_weights=True,
                when_below=30, # todo move to parameters
            )
        )
    if scheduler_type == 'StepLR':
        plugins.append(
            LRSchedulerPlugin(
                StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma),
                first_exp_only=scheduler_first_exp_only,
            )
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        plugins.append(
            LRSchedulerPlugin(
                ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_gamma, patience=5),
                first_exp_only=scheduler_first_exp_only, metric='train_loss'
            )
        )
    elif scheduler_type == 'CosineAnnealingLR':
        plugins.append(
            LRSchedulerPlugin(
                CosineAnnealingLR(optimizer, T_max=scheduler_step_size, eta_min=scheduler_eta_min),
            )
        )
    else:
        raise ValueError(f"[red]Unknown scheduler type '{scheduler_type}'[/red]")

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
        raise ValueError(f"[red]Unknown strategy type '{strategy}'[/red]")

    debug_print(cl_strategy.evaluator)
    debug_print(cl_strategy.plugins)

    @time_logger(log_file=f'{log_folder}/timing.txt')
    def run(train_stream, test_stream, cl_strategy, model, log_folder):
        results = []
        for idx, train_exp in enumerate(train_stream):
            print(f"Starting training experience [red]{idx}[/red]: ")
            cl_strategy.train(train_exp)
            print(f"Starting testing experience [red]{idx}[/red]: ")
            results.append(cl_strategy.eval(test_stream))
            print(f"Saving model after experience [red]{idx}[/red]: ")
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
            ['Loss_Exp', 'R2Score_Exp', 'RelativeDistance_Exp'],
            ['Loss over each experience', 'R2 Score over each experience', 'Relative Distance over each experience'],
            ['Loss', 'R2 Score', 'Relative Distance']
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
        try:
            shutil.rmtree(log_folder)
        except:
            debug_print(f"[red]Failed to remove {log_folder}[/red]")
        finally:
            raise ex


if __name__ == '__main__':
    main()
