import json
import sys
import os
import shutil
from datetime import datetime

from avalanche.logging import InteractiveLogger

from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin

sys.path.append(os.path.dirname(__file__))  # Add src directory to sys.path

from .utils import *
from .configs import *


def main():
    # Config
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if len(sys.argv) < 2:
        print(f"[red]Usage[/red]: [cyan](python) {sys.argv[0]} configuration_file_name.json [/cyan]")
        sys.exit(1)
    config_parser = ConfigParser(sys.argv[1])
    debug_print(config_parser.parsing_dict)
    config_parser.load_config()

    # Relevant config names
    config = config_parser.get_config()
    model_type = config['architecture']['name']
    optimizer_type = config['optimizer']['name']
    loss_type = config['loss']['name']
    strategy_type = config['strategy']['name']

    # Config Processing
    config_parser.process_config()
    debug_print(config_parser.get_config())
    # General
    train_mb_size = config_parser['train_mb_size']
    eval_mb_size = config_parser['eval_mb_size']
    train_epochs = config_parser['train_epochs']
    num_campaigns = config_parser['num_campaigns']
    dtype = config_parser['dtype']
    task = config_parser['task']
    # Dataset
    input_columns = config_parser['input_columns']
    output_columns = config_parser['output_columns']
    input_size = len(input_columns)
    output_size = len(output_columns)
    pow_type = config_parser['pow_type']
    cluster_type = config_parser['cluster_type']
    dataset_type = config_parser['dataset_type']
    normalize_inputs = config_parser['normalize_inputs']
    normalize_outputs = config_parser['normalize_outputs']
    # Architecture
    model = config_parser['architecture']
    # Loss
    criterion = config_parser['loss']
    # Optimizer
    optimizer_config = config_parser['optimizer']
    optimizer_class = optimizer_config['class']
    optimizer_parameters = optimizer_config['parameters']
    optimizer = optimizer_class(model.parameters(), **optimizer_parameters)
    # Early Stopping
    early_stopping = config_parser.get_config().get('early_stopping', None)
    # Scheduler
    scheduler_config = config_parser.get_config().get('scheduler', None)
    scheduler = None
    if scheduler_config:
        scheduler_class = scheduler_config['class']
        scheduler_parameters = scheduler_config['parameters']
        scheduler_metric = scheduler_config['metric']
        scheduler_first_epoch_only = scheduler_config['first_epoch_only']
        scheduler_first_exp_only = scheduler_config['first_exp_only']
        scheduler = LRSchedulerPlugin(
            scheduler_class(optimizer, **scheduler_parameters),
            metric=scheduler_metric,
            first_exp_only=scheduler_first_exp_only,
            first_epoch_only=scheduler_first_epoch_only,
        )
    # CL Strategy Config
    cl_strategy_config = config_parser['strategy']
    cl_strategy_class = cl_strategy_config['class']
    cl_strategy_parameters = cl_strategy_config['parameters']
    # Filters
    filters_by_geq = None
    filters_by_leq = None
    filters = config_parser.get_config().get('filters', None)
    if filters:
        filters_by_geq = filters.get('by_geq', None)
        filters_by_leq = filters.get('by_leq', None)

    print("[green]Configuration Loaded[/green]:")
    print(f"  Device: [cyan]{device}[/cyan]")
    for field_name, field_value in config_parser.get_config().items():
        print(f"  {field_name}: [cyan]{field_value}[/cyan]")

    # Print Model Size
    trainables, total = get_model_size(model)
    print(
        f"[green]Trainable Parameters[/green] = [red]{trainables}[/red]"
        f"\n[green]Total Parameters[/green] = [red]{total}[/red]"
    )

    # Prepare folders for experiments
    folder_name = f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} {model_type}"
    log_folder = os.path.join(
        'logs', pow_type, cluster_type, task, dataset_type, strategy_type, folder_name
    )
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
    csv_file = f'data/baseline/cleaned/{pow_type}_cluster/{cluster_type}/complete_dataset.csv'
    print(
        f"Input columns = {input_columns}"
        f"\nOutput columns = {output_columns}"
    )
    benchmark = make_benchmark(
        csv_file, train_datasets, test_datasets, task=task, NUM_CAMPAIGNS=num_campaigns,
        dtype=dtype, normalize_inputs=normalize_inputs, normalize_outputs=normalize_outputs,
        log_folder=log_folder, input_columns=input_columns, output_columns=output_columns,
        dataset_type=dataset_type, filter_by_geq=filters_by_geq, filter_by_leq=filters_by_leq,
    )

    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    with open(os.path.join(log_folder, 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=4)
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
        plugins.append(early_stopping)
    if scheduler:
        plugins.append(scheduler)

    # Continual learning strategy
    cl_strategy = cl_strategy_class(
        model=model, optimizer=optimizer, criterion=criterion, train_mb_size=train_mb_size,
        train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, evaluator=eval_plugin,
        plugins=plugins, **cl_strategy_parameters
    )

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
    print("[#aa0000]Starting ...[/#aa0000]")
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
