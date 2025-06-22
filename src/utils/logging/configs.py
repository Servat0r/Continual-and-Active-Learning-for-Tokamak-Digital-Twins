from typing import Optional
import re
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..models.utils import get_model_log_descriptor
from ..scenarios import *
from ...database import SecureMLExperimentDB, Strategy, General, ActiveLearning, Experiment, get_db


simulator_prefixes: dict[str, str] = {
    'qualikiz': '',
    'tglf': 'TGLF/' 
}


@dataclass
class LoggingConfiguration:
    """
    Configuration objects for logging.
    Standard logging folder format is:
        - for pure CL: "logs/pow_type/cluster_type/task/dataset_type/outputs/strategy/extra_log_folder " + 
            "(batch size) (hidden size) (hidden layers)"
        - for AL(CL): "logs/pow_type/cluster_type/task/dataset_type/outputs/strategy/AL(CL)/Continual/ " +
            "al_method/Batches al_batch_size al_max_batch_size full_first_set reload_weights downsampling/ " +
            "extra_log_folder (batch size) (hidden size) (hidden layers)"
    :param pow_type: One of {"highpow", "lowpow"}.
    :param cluster_type: One of {"Ip_Pin_based", "tau_based", "pca_based"}.
    :param task: One of {"classification", "regression"}.
    :param dataset_type: One of {"complete", "not_null"}.
    :param outputs: Either a string or a list of strings, each per output columns.
    If a string, it must be of the form of the output of "_".join(outputs_list).
    :param strategy: Strategy name, e.g. "Naive" or "Replay".
    :param extra_log_folder: Extra log folder path (see README).
    :param count: The ordinal - chronologically - at which the folder appears in the
    ordered sequence (e.g., 1 for retrieving 2nd string that ends with f"task_{task_id}").
    :param task_id: Run id, in {0, ..., N-1}.
    """
    scenario: ScenarioConfig
    strategy: str = 'Naive'
    extra_log_folder: Optional[str] = 'Base'
    hidden_size: Optional[int] = 1024
    hidden_layers: Optional[int] = 2
    batch_size: int = 4096
    active_learning: bool = False
    al_config: ActiveLearningConfig = None
    experiment_name: Optional[str] = None # If given, returns exactly that experiment

    @staticmethod
    def from_experiment(exp_dict: dict, db: SecureMLExperimentDB):
        data = {}
        scenario: ScenarioConfig = ScenarioConfig.from_experiment(exp_dict, db)
        data['scenario'] = scenario
        data['experiment_name'] = exp_dict['name']
        # Strategy
        id_strategy: Optional[int] = exp_dict.get('id_strategy', None)
        if id_strategy is None:
            raise ValueError(f"Cannot build LoggingConfiguration from experiment dictionary since it does not contain \"id_strategy\"")
        else:
            if 'strategy_name' in exp_dict:
                data['strategy'] = exp_dict['strategy_name']
            else:
                # Retrieve data from database
                strategy_data = db.read_record(Strategy, id_strategy, as_dict=True)
                data['strategy'] = strategy_data['name']
        # General (batch size)
        id_general: Optional[int] = exp_dict.get('id_general', None)
        if id_general is None:
            raise ValueError(f"Cannot build LoggingConfiguration from experiment dictionary since it does not contain \"id_general\"")
        else:
            if 'general_train_mb_size' in exp_dict:
                data['batch_size'] = exp_dict['general_train_mb_size']
            else:
                general_data = db.read_record(General, id_general, as_dict=True)
                data['batch_size'] = general_data['train_mb_size']
        # Active Learning data
        if exp_dict.get('id_active_learning', None) is not None:
            id_al = exp_dict['id_active_learning']
            data['active_learning'] = True
            start_str = 'active_learning_'
            start_len = len(start_str)
            al_data = {k[start_len:]: v for k, v in exp_dict.items() if k.startswith(start_str)}
            if al_data:
                data['al_config'] = al_data
            else:
                # Retrieve from database
                al_data = db.read_record(ActiveLearning, id_al, as_dict=True)
                data['al_config'] = al_data
        else:
            data['active_learning'] = False
        data.pop('id', None)
        return LoggingConfiguration(**data)

    def __base_log_folder(self, mode: str = 'old') -> str:
        """
        Old mode is the one used during Master Thesis (e.g.: highpow/tau_based/regression/not_null/<outputs>/Cumulative/Base (...)).
        New mode is like: qualikiz/highpow/tau_based/not_null/regression/<outputs>/CL or CLAEA/[...]
        """
        raw_outputs = self.scenario.outputs
        outputs_string = raw_outputs if isinstance(raw_outputs, str) else '_'.join(raw_outputs)
        simulator_prefix = simulator_prefixes[self.scenario.simulator_type]
        base_extra_name = f'{self.extra_log_folder} ({self.batch_size} batch size) '.lstrip() + \
            f"({self.hidden_size} hidden size)"
        if (self.scenario.simulator_type == 'tglf') or (self.hidden_layers != 2):
            base_extra_name = base_extra_name + f' ({self.hidden_layers} hidden layers)'
        if mode == 'old':
            if self.active_learning:
                al_base_extra_name = self.get_al_log_folder(mode=mode)
                base_extra_name = f'{al_base_extra_name}/{base_extra_name}'
            base_extra_name = f'{simulator_prefix}{base_extra_name}'
            index_dir = os.path.join(
                'old_logs', self.scenario.pow_type, self.scenario.cluster_type,
                self.scenario.task, self.scenario.dataset_type,
                outputs_string, self.strategy, base_extra_name
            )
        else:
            class_folder = 'CL' if not self.active_learning else 'CLAEA'
            assert len(self.experiment_name) > 0, f"Error: <LoggingConfiguration object>.experiment_name is not set"
            index_dir = os.path.join(
                'logs', self.scenario.simulator_type, self.scenario.pow_type, self.scenario.cluster_type,
                self.scenario.dataset_type, self.scenario.task, outputs_string, class_folder,
                self.strategy, self.experiment_name
            )
        return index_dir
    
    @staticmethod
    def parse_al_log_folder(folder: str) -> dict:
        splits = folder.split('Batches ', 1)[1:].split(' ', 2)
        batch_size, max_batch_size = int(splits[0]), int(splits[1])
        splits = splits[2].split('full first set', 1)
        match splits[0]:
            case '':
                full_first_set = True
            case 'non-':
                full_first_set = False
            case _:
                raise ValueError(f"Unknown prefix for 'full first set': {splits[0]}")
        splits = splits[1].split('reload weights', 1)
        match splits[0]:
            case '':
                reload_initial_weights = True
            case 'no ':
                reload_initial_weights = False
            case _:
                raise ValueError(f"Unknown prefix for 'reload weights': {splits[0]}")
        splits = splits[1].split('downsampling ')
        if int(splits[1]) == float(splits[1]):
            factor = int(splits[1])
        else:
            factor = float(splits[1])
        return {
            'batch_size': batch_size,
            'max_batch_size': max_batch_size,
            'full_first_set': full_first_set,
            'reload_initial_weights': reload_initial_weights,
            'downsampling_factor': factor
        }
    
    def get_al_log_folder(self, mode: str = 'old') -> str:
        full_first_set_str = ('' if self.al_config.full_first_set else 'non-') + 'full first set'
        reload_weights_str = ('' if self.al_config.reload_initial_weights else 'no ') + 'reload weights'
        downsampling_factor_str = f'downsampling {float(self.al_config.downsampling_factor)}'
        if mode == 'old':
            return os.path.join(
                "AL(CL)", "Continual", self.al_config.standard_method,
                f"Batches {self.al_config.batch_size} {self.al_config.max_batch_size} " + \
                f"{full_first_set_str} {reload_weights_str} {downsampling_factor_str}"
            )
        else:
            return os.path.join(
                self.al_config.standard_method, f"Batches {self.al_config.batch_size} " + \
                f"{self.al_config.max_batch_size} {full_first_set_str} {reload_weights_str} {downsampling_factor_str}"
            )
    
    def make_log_folder(self, mode: str = 'old') -> str:
        index_dir = self.__base_log_folder(mode=mode)
        os.makedirs(index_dir, exist_ok=True)
        return index_dir
    
    def get_log_folder(self, count: int = -1, task_id: int = 0, suffix: bool = True, mode: str = 'new') -> str:
        index_dir = self.__base_log_folder(mode=mode)
        if suffix:
            current_count = 0
            last_dirname = None
            for dirname in os.listdir(index_dir):
                if dirname.endswith(f"task_{task_id}"):
                    if (count >= 0) and (current_count >= count):
                        return os.path.join(index_dir, dirname)
                    else:
                        current_count += 1
                        last_dirname = dirname[:]
            if (count == -1) and (last_dirname is not None):
                return os.path.join(index_dir, last_dirname)
            raise ValueError(f"Not found any directory in \"{index_dir}\" ending with \"task_{task_id}\"")
        else:
            return index_dir
    
    def get_common_params(self, start: int = 0, end: int = 8):
        data = list(self.scenario.get_common_params()) + [self.strategy, self.extra_log_folder]
        return tuple(data[start:end])


def get_logging_config_from_filepath(file_path: str):
    # file_path = logs/<sim_type>/<pow_type>/<cluster_type>/<dataset_type>/<task>/<outputs>/<class>/<strategy>/<name>
    folders: list[str] = re.split(r'[\\/]', file_path)
    #print(f"FOLDERS: {folders}")
    (simulator_type, pow_type, cluster_type, dataset_type, task, outputs, class_folder) = tuple(folders[1:8])
    extra_folders: list[str] = folders[8:]
    scenario = ScenarioConfig(simulator_type, pow_type, cluster_type, dataset_type, task, outputs)
    strategy, experiment_name = extra_folders[0], extra_folders[1]
    is_active_learning = True if class_folder == 'CLAEA' else False
    result = LoggingConfiguration(
        scenario=scenario,
        strategy=strategy,
        active_learning=is_active_learning,
        experiment_name=experiment_name
    )
    return result


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


__all__ = [
    "LoggingConfiguration", "simulator_prefixes", "get_logging_config_from_filepath",
    "get_training_times", "get_num_epochs"
]