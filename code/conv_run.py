import json
import sys
import os
import torch
from torch.nn import MSELoss
from torch.optim import SGD, Adam

from avalanche.benchmarks import *
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive

from datetime import datetime
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


@time_logger
def make_benchmark(csv_file, train_datasets, test_datasets, NUM_CAMPAIGNS=5, dtype='float64'):
    for campaign in range(NUM_CAMPAIGNS):
        print(f"Loading data for campaign {campaign} ...")
        train_dataset, test_dataset = get_avalanche_csv_regression_datasets(
            csv_file, BASELINE_HIGHPOW_INPUTS, BASELINE_HIGHPOW_OUTPUTS, # Only efe and efi
            filter_by={'campaign': [campaign]}, float_precision=dtype,
            device='cpu',
        )
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    benchmark = benchmark_from_datasets(train=train_datasets, test=test_datasets)
    return benchmark


@time_logger
def run(train_stream, test_stream, cl_strategy):
    results = []
    for idx, train_exp in enumerate(train_stream):
        print(f"Starting training experience {idx}: ")
        cl_strategy.train(train_exp)
        print(f"Starting testing experience {idx}: ")
        results.append(cl_strategy.eval(test_stream))
    return results


def main():

    print(os.getcwd())
    # Config
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model
    input_size = len(BASELINE_HIGHPOW_INPUTS)
    output_size = len(BASELINE_HIGHPOW_OUTPUTS)
    train_mb_size = 1024
    eval_mb_size = 2048
    train_epochs = 30
    num_campaigns = 10
    dtype='float64'
    lr = 5e-3
    momentum = 0.9
    weight_decay = 2e-4
    hidden_size = 1024
    hidden_layers = 2
    pow_type = 'highpow'
    cluster_type = 'Ip_Pin_based'
    optimizer_type = 'SGD'
    if len(sys.argv) >= 2:
        config_file = sys.argv[1]
        config = load_config(config_file)
    else:
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
            'hidden_size': hidden_size,
            'hidden_layers': hidden_layers,
            'pow_type': pow_type,
            'cluster_type': cluster_type,
            'optimizer_type': optimizer_type,
        }
    for key, value in config.items():
        globals()[key] = value
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
        print(f"  Hidden size: {hidden_size}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  Pow type: {pow_type}")
        print(f"  Cluster type: {cluster_type}")
        print(f"  Optimizer type: {optimizer_type}")
    except NameError as e:
        print(f"Error: Missing expected configuration key - {e}")
        sys.exit(1)

    model = SimpleRegressionMLP(
        input_size=input_size, hidden_size=hidden_size, hidden_layers=hidden_layers,
        output_size=output_size, dtype=dtype,
    )

    train_datasets = []
    test_datasets = []
    csv_file = f'../data/baseline/campaigns_lumped/{pow_type}_cluster/{cluster_type}/all_campaigns.csv'
    benchmark = make_benchmark(csv_file, train_datasets, test_datasets, num_campaigns, dtype=dtype)

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
    log_folder = os.path.join('logs', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    csv_logger = CustomCSVLogger(log_folder=log_folder)
    with open(os.path.join(log_folder, 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=4)
    # Define loss
    criterion = MSELoss(reduction='mean')
    # Define the evaluation plugin with desired metrics
    eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[csv_logger]
    )

    # Continual learning strategy
    cl_strategy = Naive(
        model=model, optimizer=optimizer, criterion=criterion, train_mb_size=train_mb_size,
        train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, evaluator=eval_plugin,
    )

    # train and test loop over the stream of experiences
    print("Starting ...")
    results = run(train_stream, test_stream, cl_strategy)
    print(results)


if __name__ == '__main__':
    main()
