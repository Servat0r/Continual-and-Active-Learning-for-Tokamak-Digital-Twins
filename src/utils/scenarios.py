# Generic Scenarios Config
from typing import Literal, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ScenarioConfig:
    """
    Basic configuration wrapper for a scenario defined by (simulator_type, pow_type, cluster_type, dataset_type, task).
    """
    simulator_type: Literal["qualikiz", "tglf"]
    pow_type: Literal["highpow", "lowpow", "mixed"]
    cluster_type: Literal["tau_based", "Ip_Pin_based", "wmhd_based", "beta_based"]
    dataset_type: Literal["complete", "not_null"]
    task: Literal["classification", "regression"]
    outputs: str = "efe_efi_pfe_pfi"

    def get_common_params(self, start: int = 0, end: int = float('inf')):
        end = min(end, 6)
        outputs_list = self.outputs.split('_') if isinstance(self.outputs, str) else self.outputs
        data = [self.simulator_type, self.pow_type, self.cluster_type, self.dataset_type, self.task, outputs_list]
        return tuple(data[start:end])
    
    def to_dict(self):
        return {
            "simulator_type": self.simulator_type,
            "pow_type": self.pow_type,
            "cluster_type": self.cluster_type,
            "dataset_type": self.dataset_type,
            "task": self.task,
            "outputs": self.outputs,
        }
    

@dataclass
class ActiveLearningConfig:
    """
    Basic configuration wrapper for AL(CL) settings.
    """
    framework: str = 'bmdal'
    batch_size: int = 256
    max_batch_size: int = 1024
    reload_initial_weights: bool = False
    standard_method: Optional[Literal[
        "random_sketch_grad", "random_sketch_ll", "bald", "batchbald",
        "badge", "bait", "coreset", "lcmd_sketch_grad", 
    ]] = "random_sketch_grad" # If None, means that selection method is described below
    selection_method: Optional[Literal["random", "maxdiag", "maxdist", "maxdet", "kmeanspp"]] = None
    initial_selection_method: Optional[Literal["random", "maxdiag", "maxdist", "maxdet", "kmeanspp"]] = None
    base_kernel: Optional[Literal["grad", "ll"]] = None
    kernel_transforms: Optional[list] = None
    sel_with_train: bool = False
    sigma: float = 0.01 # It actually represents variance (sigma**2)
    full_first_set: bool = False
    first_set_size: int = 5120
    downsampling_factor: int | float = 0.5

    def to_dict(self):
        return asdict(self)


def get_paper_scenario(cluster_type: Literal["wmhd_based", "beta_based"]):
    return ScenarioConfig(
        simulator_type='qualikiz',
        pow_type='mixed',
        cluster_type=cluster_type,
        dataset_type='not_null',
        task='regression'
    )


def get_qlk_regression_scenario(
    pow_type: Literal["highpow", "lowpow", "mixed"],
    cluster_type: Literal["tau_based", "Ip_Pin_based", "wmhd_based", "beta_based"],
):
    return ScenarioConfig(
        simulator_type='qualikiz',
        pow_type=pow_type,
        cluster_type=cluster_type,
        dataset_type='not_null',
        task='regression'
    )


def get_tglf_regression_scenario(
    pow_type: Literal["highpow", "lowpow", "mixed"],
    cluster_type: Literal["tau_based", "Ip_Pin_based", "wmhd_based", "beta_based"],
):
    return ScenarioConfig(
        simulator_type='tglf',
        pow_type=pow_type,
        cluster_type=cluster_type,
        dataset_type='not_null',
        task='regression'
    )


__all__ = [
    'ScenarioConfig', 'ActiveLearningConfig', 'get_paper_scenario',
    'get_qlk_regression_scenario', 'get_tglf_regression_scenario'
]
