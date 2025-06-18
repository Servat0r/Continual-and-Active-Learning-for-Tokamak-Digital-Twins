# Process config data into the database
from typing import *
from src.utils import *
from src.configs import *
from src.database import *


def __get_record_id(db: SecureMLExperimentDB, orm_class, conditions, field, results):
    record = db.read_record_where(orm_class, conditions=conditions, as_dict=True)
    print(f"Got {record} for class = {orm_class}")
    if record is not None:
        results[field] = record['id']
    else:
        record = orm_class(**conditions)
        results[field] = db.create_record(record)


def __get_records_id(db: SecureMLExperimentDB, orm_class, conditions, field, results, parameters: dict):
    records = db.read_records_where(orm_class, conditions=conditions, as_dict=True)
    print(f"Got {len(records)} records for class = {orm_class}")
    record_to_create = False
    if records is not None:
        filtered_records = []
        for record in records:
            record_parameters = record['parameters']
            condition = all([
                record_parameters.get(param_name, None) == param_value for param_name, param_value in parameters.items()
            ])
            if condition:
                filtered_records.append(record)
        if filtered_records:
            print(f"Filtered {len(filtered_records)} for class = {orm_class}")
            record = filtered_records[0]
            results[field] = record['id']
        else:
            record_to_create = True
    else:
        record_to_create = True
    if record_to_create:
        support = conditions.copy()
        support['parameters'] = parameters
        record = orm_class(**support)
        results[field] = db.create_record(record)


def config2db(
    config: dict[str, Any], db: SecureMLExperimentDB,
    num_tasks: int, is_test: bool
) -> Tuple[int, str]:
    config = config.copy() # Make a safe copy to operate with
    results = {
        'id_general': None,
        'id_scenario': None,
        'id_architecture': None,
        'id_loss': None,
        'id_optimizer': None,
        'id_scheduler': None,
        'id_early_stopping': None,
        'id_strategy': None,
        'id_active_learning': None
    }
    # General
    general = config['general']
    if 'task' in general:
        task = general.pop('task')
    else:
        task = None
    general.pop('dtype', None)
    __get_record_id(db, General, general, 'id_general', results)
    # Scenario
    scenario = config['dataset']
    for field in {'input_size', 'output_size', 'load_saved_final_data'}:
        scenario.pop(field, None)
    scenario['task'] = task
    scenario['input_columns'] = '+'.join([w.lower().strip() for w in scenario['input_columns']])
    scenario['output_columns'] = '+'.join([w.lower().strip() for w in scenario['output_columns']])
    __get_record_id(db, Scenario, scenario, 'id_scenario', results)
    # Architecture
    architecture = config['architecture']
    architecture['model_type'] = architecture['model_class_name']
    for field in {'name', 'model_name', 'model_class_name'}:
        architecture.pop(field, None)
    arch_params = architecture.pop('parameters')
    __get_records_id(db, Architecture, architecture, 'id_architecture', results, arch_params)
    # Loss
    loss = config['loss']
    loss['name'] = loss['name'].lower()
    loss_parameters = loss.pop('parameters')
    __get_records_id(db, Loss, loss, 'id_loss', results, loss_parameters)
    # Optimizer
    optimizer = config['optimizer']
    optim_params = optimizer.pop('parameters')
    __get_records_id(db, Optimizer, optimizer, 'id_optimizer', results, optim_params)
    # Scheduler
    scheduler = config['scheduler']
    scheduler_parameters = scheduler.pop('parameters')
    __get_records_id(db, Scheduler, scheduler, 'id_scheduler', results, scheduler_parameters)
    # Early Stopping
    early_stopping = config['early_stopping']
    early_stopping.pop('val_stream_name', None)
    __get_record_id(db, EarlyStopping, early_stopping, 'id_early_stopping', results)
    # Strategy
    strategy = config['strategy']
    for field in {'ignore', 'extra_log_folder'}:
        strategy.pop(field)
    strategy_params = strategy.pop('parameters')
    __get_records_id(db, Strategy, strategy, 'id_strategy', results, strategy_params)
    # Experiment
    experiment = Experiment(
        **results, # All foreign keys
        name='Experiment', # Will be completed to a unique-generated name
        num_tasks=num_tasks,
        status='invalid',
        is_test=is_test
    )
    experiment_id, experiment_name = db.create_experiment(experiment, max_attempts=10)
    print(f"Successfully created experiment with id = {experiment_id}, name = {experiment_name}")
    return experiment_id, experiment_name


__all__ = ['config2db']
