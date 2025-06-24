# Process config data into the database
from typing import *
from src.utils import *
from src.configs import *
from src.database import *


def __check_eq_dicts(dict1: dict, dict2: dict):
    l1, l2 = len(dict1), len(dict2)
    if l1 != l2: return False
    cond = all([field in dict2 for field in dict1])
    if not cond: return False
    cond = all([type(dict1[field]) == type(dict2[field]) for field in dict1])
    if not cond: return False
    dict_fields = sorted([field for field in dict1 if isinstance(dict1[field], dict)])
    cond = all([__check_eq_dicts(dict1[f], dict2[f]) for f in dict_fields])
    if not cond: return False
    return True


def __get_records_id(db: SecureMLExperimentDB, orm_class, conditions, field, results):
    json_fields = orm_class.json_fields()
    cond_copy = {k: v for k, v in conditions.items() if '@' not in k}
    json_conditions = {field: cond_copy.pop(field) for field in json_fields if field in conditions}
    if (orm_class == ActiveLearning) and ('kernel_transforms' in cond_copy):
        cond_copy['kernel_transforms'] = str(cond_copy['kernel_transforms'])
    records = db.get(orm_class, conditions=cond_copy, )
    record_to_create = False
    if records is not None:
        filtered_records = []
        for record in records:
            record_dict = {k: record[k] for k in json_conditions}
            condition = all([
                record_dict.get(k, None) == value for k, value in record_dict.items()
            ])
            if condition:
                filtered_records.append(record)
        if filtered_records:
            record = filtered_records[0]
            results[field] = record['id']
        else:
            record_to_create = True
    else:
        record_to_create = True
    if record_to_create:
        cond_copy = {k: v for k, v in conditions.items() if '@' not in k}
        record = orm_class(**cond_copy)
        results[field] = db.create(record)


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
    __get_records_id(db, General, general, 'id_general', results)
    # Scenario
    scenario = config['dataset']
    for field in {'input_size', 'output_size', 'load_saved_final_data'}:
        scenario.pop(field, None)
    scenario['task'] = task
    scenario['input_columns'] = '+'.join([w.lower().strip() for w in scenario['input_columns']])
    scenario['output_columns'] = '+'.join([w.lower().strip() for w in scenario['output_columns']])
    __get_records_id(db, Scenario, scenario, 'id_scenario', results)
    # Architecture
    architecture = config['architecture']
    for field in {'name', 'model_name', 'model_class_name'}:
        architecture.pop(field, None)
    __get_records_id(db, Architecture, architecture, 'id_architecture', results)
    # Loss
    loss = config['loss']
    loss['name'] = loss['name'].lower()
    __get_records_id(db, Loss, loss, 'id_loss', results)
    # Optimizer
    optimizer = config['optimizer']
    __get_records_id(db, Optimizer, optimizer, 'id_optimizer', results)
    # Scheduler
    scheduler = config['scheduler']
    __get_records_id(db, Scheduler, scheduler, 'id_scheduler', results)
    # Early Stopping
    early_stopping = config['early_stopping']
    early_stopping.pop('val_stream_name', None)
    #import sys; sys.exit(0)
    __get_records_id(db, EarlyStopping, early_stopping, 'id_early_stopping', results)
    # Strategy
    strategy = config['strategy']
    for field in {'ignore', 'extra_log_folder'}:
        strategy.pop(field)
    __get_records_id(db, Strategy, strategy, 'id_strategy', results)
    # Active Learning
    active_learning = config.get('active_learning', None)
    if active_learning: # Should be "not None" and have AT LEAST ONE element
        __get_records_id(db, ActiveLearning, active_learning, 'id_active_learning', results)
    # Experiment
    experiment = Experiment(
        **results, # All foreign keys
        name='Exp', # Will be completed to a unique-generated name
        num_tasks=num_tasks,
        status='invalid',
        is_test=is_test
    )
    experiment_id, experiment_name = db.create_experiment(experiment, max_attempts=10)
    print(f"Successfully created experiment with id = {experiment_id}, name = {experiment_name}")
    return experiment_id, experiment_name


__all__ = ['config2db']
