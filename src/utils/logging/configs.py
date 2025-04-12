import os
from dataclasses import dataclass


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
    pow_type: str = 'highpow'
    cluster_type: str = 'tau_based'
    dataset_type: str = 'not_null'
    task: str = 'regression'
    outputs: str | list[str] = 'efe_efi_pfe_pfi'
    strategy: str = 'Naive'
    extra_log_folder: str = 'Base'
    simulator_type: str = 'qualikiz'
    hidden_size: int = 1024
    hidden_layers: int = 2
    batch_size: int = 4096
    active_learning: bool = False
    al_method: str = 'random_sketch_grad'
    al_batch_size: int = 128
    al_max_batch_size: int = 2048
    al_full_first_set: bool = False
    al_reload_weights: bool = False
    al_downsampling_factor: int | float = 0.5

    def __base_log_folder(self) -> str:
        outputs_string = self.outputs if isinstance(self.outputs, str) else '_'.join(self.outputs)
        simulator_prefix = simulator_prefixes[self.simulator_type]
        base_extra_name = f'{self.extra_log_folder} ({self.batch_size} batch size) '.lstrip() + \
            f"({self.hidden_size} hidden size)"
        if (self.simulator_type == 'tglf') or (self.hidden_layers != 2):
            base_extra_name = base_extra_name + f' ({self.hidden_layers} hidden layers)'
        if self.active_learning:
            al_base_extra_name = self.get_al_log_folder()
            base_extra_name = f'{al_base_extra_name}/{base_extra_name}'
        base_extra_name = f'{simulator_prefix}{base_extra_name}'
        index_dir = os.path.join(
            'logs', self.pow_type, self.cluster_type, self.task, self.dataset_type,
            outputs_string, self.strategy, base_extra_name
        )
        return index_dir
    
    def get_al_log_folder(self) -> str:
        full_first_set_str = ('' if self.al_full_first_set else 'non-') + 'full first set'
        reload_weights_str = ('' if self.al_reload_weights else 'no ') + 'reload weights'
        downsampling_factor_str = f'downsampling {float(self.al_downsampling_factor)}'
        return os.path.join(
            "AL(CL)", "Continual", self.al_method,
            f"Batches {self.al_batch_size} {self.al_max_batch_size} " + \
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


__all__ = [
    "LoggingConfiguration", "simulator_prefixes"
]