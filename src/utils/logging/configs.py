import os
from dataclasses import dataclass
from ..models.utils import get_model_log_descriptor
from ..scenarios import *


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
    extra_log_folder: str = 'Base'
    model_type: str = 'MLP' # NEW
    hidden_size: int = 1024
    hidden_layers: int = 2
    batch_size: int = 4096
    active_learning: bool = False
    al_config: ActiveLearningConfig = None

    def __base_log_folder(self) -> str:
        raw_outputs = self.scenario.outputs
        outputs_string = raw_outputs if isinstance(raw_outputs, str) else '_'.join(raw_outputs)
        simulator_prefix = simulator_prefixes[self.scenario.simulator_type]
        # TODO Compatibility break for previous version (we now have other model classes + hidden_layers should always be included for MLPs)
        log_descriptor_parameters: list[str] = get_model_log_descriptor(self.model_type)
        #for ...
        base_extra_name = f'{self.extra_log_folder} ({self.batch_size} batch size) '.lstrip() + \
            f"({self.hidden_size} hidden size)"
        if (self.scenario.simulator_type == 'tglf') or (self.hidden_layers != 2):
            base_extra_name = base_extra_name + f' ({self.hidden_layers} hidden layers)'
        if self.active_learning:
            al_base_extra_name = self.get_al_log_folder()
            base_extra_name = f'{al_base_extra_name}/{base_extra_name}'
        base_extra_name = f'{simulator_prefix}{base_extra_name}'
        index_dir = os.path.join(
            'logs', self.scenario.pow_type, self.scenario.cluster_type,
            self.scenario.task, self.scenario.dataset_type,
            outputs_string, self.strategy, base_extra_name
        )
        return index_dir
    
    def get_al_log_folder(self) -> str:
        full_first_set_str = ('' if self.al_config.full_first_set else 'non-') + 'full first set'
        reload_weights_str = ('' if self.al_config.reload_initial_weights else 'no ') + 'reload weights'
        downsampling_factor_str = f'downsampling {float(self.al_config.downsampling_factor)}'
        return os.path.join(
            "AL(CL)", "Continual", self.al_config.standard_method,
            f"Batches {self.al_config.batch_size} {self.al_config.max_batch_size} " + \
            f"{full_first_set_str} {reload_weights_str} {downsampling_factor_str}"
        )
    
    def make_log_folder(self) -> str:
        index_dir = self.__base_log_folder()
        os.makedirs(index_dir, exist_ok=True)
        return index_dir
    
    def get_log_folder(self, count: int = -1, task_id: int = 0, suffix: bool = True) -> str:
        index_dir = self.__base_log_folder()
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


def get_CL_logging_config_from_filepath(file_path: str):
    # file_path = logs/<pow_type>/<cluster_type>/<task>/<dataset_type>/<outputs>/<strategy>/<tglf if present><base_extra_name>
    values: list[str] = file_path.split('/')
    assert len(values) >= 8, f"len(file_path split by '/') == {len(values)}"
    pow_type: str = values[1]
    cluster_type: str = values[2]
    task: str = values[3]
    dataset_type: str = values[4]
    outputs_string: str = values[5]
    strategy: str = values[6]
    if len(values) == 8:
        simulator_type: str = 'qualikiz'
        base_extra_name: str = values[7]
    else:
        simulator_type: str = values[7]
        base_extra_name: str = values[8]
    # Extract extra_log_folder, hidden_size and hidden_layers from base_extra_name
    # Split by last two parentheses to handle extra_log_folder containing parentheses
    parts = base_extra_name.rsplit('(', 3)
    extra_log_folder = parts[0].strip()
    batch_size = int(parts[1].split(' batch size')[0])
    hidden_size = int(parts[2].split(' hidden size')[0])
    if len(parts) > 3:
        hidden_layers = int(parts[3].split(' hidden layers')[0])
    else:
        hidden_layers = 2

    scenario_config = ScenarioConfig(
        simulator_type=simulator_type,
        pow_type=pow_type,
        cluster_type=cluster_type,
        dataset_type=dataset_type,
        task=task, outputs=outputs_string.split('_')
    )

    logging_config = LoggingConfiguration(
        active_learning=False, scenario=scenario_config, strategy=strategy,
        extra_log_folder=extra_log_folder, batch_size=batch_size,
        hidden_size=hidden_size, hidden_layers=hidden_layers
    )

    return logging_config


__all__ = [
    "LoggingConfiguration", "simulator_prefixes", "get_CL_logging_config_from_filepath"
]