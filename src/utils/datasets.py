from typing import Any, Callable
import os
import pandas as pd
import torch
from avalanche.benchmarks import AvalancheDataset
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import TransformGroups, DataAttribute


class CSVRegressionDataset(Dataset):
    """
    Dataset class for handling regression data contained in CSV files.
    """
    def __init__(
            self, data, input_columns: list[str], output_columns: list[str],
            transform=None, target_transform=None, filter_by: dict[str, list] = None,
            float_precision: str = 'float32', device=None,
            filter_by_leq: dict[str, int | float] = None,
            filter_by_geq: dict[str, int | float] = None,
            inputs: torch.Tensor = None, outputs: torch.Tensor = None,
    ):
        """
        :param data: pandas DataFrame.
        :param input_columns: Columns to be used as inputs.
        :param output_columns: Columns to be used as outputs.
        :param transform: Extra inputs transform.
        :param target_transform: Extra outputs transform.
        :param filter_by: A dictionary for filtering data according to columns values.
        Specifically, for each (column -> values), each row such that row[column] is not in values
        will be filtered out.
        :param float_precision: Floating-point precision. Defaults to 'float32'.
        :param device: One of {"cpu", "gpu", "cuda", "cuda:<id>"}. Defaults to "cpu".
        :param filter_by_leq: A dictionary for filtering data according to columns values.
        Specifically, for each (column -> value), each row such that row[column] > value
        will be filtered out.
        :param filter_by_geq: A dictionary for filtering data according to columns values.
        Specifically, for each (column -> value), each row such that row[column] < value
        will be filtered out.
        """
        device = str(device) if device is not None else None
        if device is None:
            device = "cpu"
        elif device == 'gpu':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        elif not device.startswith('cuda'):
            device = 'cpu'
        if torch.cuda.is_available() and (device != 'cpu'):
            device_index = torch.cuda.current_device()
            print(f"Using CUDA device {device}: ({torch.cuda.get_device_name(device_index)}) ...")
        else:
            print(f"Using cpu ...")
        # Load the CSV file
        if filter_by:
            for column, values in filter_by.items():
                data = data[data[column].isin(values)]
        if filter_by_leq:
            for column, value in filter_by_leq.items():
                data = data[data[column] <= value]
        if filter_by_geq:
            for column, value in filter_by_geq.items():
                data = data[data[column] >= value]
        self.transform = transform
        self.target_transform = target_transform
        self.inputs = None
        self.targets = None
        if data is not None:
            self.inputs = torch.tensor(data[input_columns].values.astype(float_precision)).to(device)
            self.targets = torch.tensor(data[output_columns].values.astype(float_precision)).to(device)
        else:
            if inputs is not None:
                self.inputs = inputs.to(device)
            if outputs is not None:
                self.targets = outputs.to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x, y = self.inputs[idx], self.targets[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def set_inputs_and_outputs(self, inputs=None, targets=None):
        if inputs is not None:
            self.inputs = inputs
        if targets is not None:
            self.targets = targets

    @property
    def device(self):
        return self.inputs.device

    def set_device(self, device):
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)


def _is_not_null(row, output_columns):
    return any([row[column] != 0.0 for column in output_columns])


def make_complete_dataset(
        pow_type, cluster_type, output_columns: list[str],
        base_src_folder='qualikiz/campaigns_lumped',
        base_dest_folder='qualikiz/cleaned',
):
    src_filename = f'data/{base_src_folder}/{pow_type}_cluster/{cluster_type}/all_campaigns.csv'
    dest_folder = f'data/{base_dest_folder}/{pow_type}_cluster/{cluster_type}'
    dest_filename = f'{dest_folder}/complete_dataset.csv'
    df = pd.read_csv(src_filename)
    df['has_turbulence'] = df.apply(lambda row: _is_not_null(row, output_columns), axis=1)
    os.makedirs(dest_folder, exist_ok=True)
    df.to_csv(dest_filename, index=False)


def get_avalanche_csv_regression_datasets(
        train_data, eval_data, test_data, input_columns: list[str], output_columns: list[str], # todo modify later
        transform=None, target_transform=None, filter_by: dict[str, list] = None,
        float_precision: str = 'float32', device=None, *, indices: list[int] | None = None,
        data_attributes: list[DataAttribute] | None = None, transform_groups: TransformGroups | None = None,
        frozen_transform_groups: TransformGroups | None = None, collate_fn: Callable[[list], Any] | None = None,
        filter_by_leq: dict[str, int | float] = None, filter_by_geq: dict[str, int | float] = None,
) -> tuple[AvalancheDataset, AvalancheDataset, AvalancheDataset]:
    """
    Builds AvalancheDataset objects on top of CSVRegressionDataset ones.
    :param train_data: Train data.
    :param eval_data: Eval data.
    :param test_data: Test data.
    :param input_columns: Input columns.
    :param output_columns: Output columns.
    :param transform: Extra inputs transform.
    :param target_transform: Extra outputs transform.
    :param filter_by: A dictionary for filtering data according to columns values.
    Specifically, for each (column -> values), each row such that row[column] is not in values
    will be filtered out.
    :param float_precision: Floating-point precision. Defaults to 'float32'.
    :param device: One of {"cpu", "gpu", "gpu:<id>"}. Defaults to "cpu".
    :param indices: See AvalancheDataset.__init__() for more information.
    :param data_attributes: See AvalancheDataset.__init__() for more information.
    :param transform_groups: See AvalancheDataset.__init__() for more information.
    :param frozen_transform_groups: See AvalancheDataset.__init__() for more information.
    :param collate_fn: See AvalancheDataset.__init__() for more information.
    :param filter_by_leq: A dictionary for filtering data according to columns values.
    Specifically, for each (column -> value), each row such that row[column] > value
    will be filtered out.
    :param filter_by_geq: A dictionary for filtering data according to columns values.
    Specifically, for each (column -> value), each row such that row[column] < value
    will be filtered out.
    :return: The triple (train_dataset, eval_dataset, test_dataset).
    """
    base_train_dataset = CSVRegressionDataset(
        train_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
        target_transform=target_transform, filter_by=filter_by, float_precision=float_precision,
        device=device, filter_by_geq=filter_by_geq, filter_by_leq=filter_by_leq,
    )
    base_eval_dataset = CSVRegressionDataset(
        eval_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
        target_transform=target_transform, filter_by=filter_by, float_precision=float_precision,
        device=device, filter_by_geq=filter_by_geq, filter_by_leq=filter_by_leq,
    )
    base_test_dataset = CSVRegressionDataset(
        test_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
        target_transform=target_transform, filter_by=filter_by, float_precision=float_precision,
        device=device, filter_by_geq=filter_by_geq, filter_by_leq=filter_by_leq,
    )

    train_dataset = AvalancheDataset(
        [base_train_dataset], indices=indices, data_attributes=data_attributes,
        transform_groups=transform_groups, frozen_transform_groups=frozen_transform_groups,
        collate_fn=collate_fn
    )
    eval_dataset = AvalancheDataset(
        [base_eval_dataset], indices=indices, data_attributes=data_attributes,
        transform_groups=transform_groups, frozen_transform_groups=frozen_transform_groups,
        collate_fn=collate_fn
    )
    test_dataset = AvalancheDataset(
        [base_test_dataset], indices=indices, data_attributes=data_attributes,
        transform_groups=transform_groups, frozen_transform_groups=frozen_transform_groups,
        collate_fn=collate_fn
    )
    return train_dataset, eval_dataset, test_dataset


# Constants for loading qualikiz fields
# High Power
QUALIKIZ_HIGHPOW_INPUTS = [
    'ane', 'ate', 'autor', 'machtor', 'x', 'zeff', 'gammae', 'q', 'smag',
    'alpha', 'ani1', 'ati0', 'normni1', 'ti_te0', 'lognustar'
]
QUALIKIZ_HIGHPOW_OUTPUTS = ['efe', 'efi', 'pfe', 'pfi']
# Low Power
QUALIKIZ_LOWPOW_INPUTS = [
    'ane', 'ate', 'x', 'q', 'smag', 'alpha', 'ani1', 'ati0', 'normni1', 'zeff', 'lognustar'
]
QUALIKIZ_LOWPOW_OUTPUTS = ['efe', 'efi', 'pfe', 'pfi']
# Mixed
QUALIKIZ_MIXED_INPUTS = [
    'ane', 'ate', 'autor', 'machtor', 'x', 'zeff', 'gammae', 'q', 'smag',
    'alpha', 'ani1', 'ati0', 'normni1', 'ti_te0', 'lognustar'
]
QUALIKIZ_MIXED_OUTPUTS = ['efe', 'efi', 'pfe', 'pfi']

# TGLF values
# High Power
TGLF_HIGHPOW_INPUTS = [
    "rmin_loc", "drmajdx_loc", "kappa_loc", "delta_loc", "q_loc",
    "q_prime_loc", "vexb_shear", "rlns_1", "rlts_1", "rlns_2",
    "rlts_2", "taus_2", "as_2", "rlns_3", "zeff", "betae", "xnue"
]
TGLF_HIGHPOW_OUTPUTS = ['efe', 'efi', 'pfe', 'pfi']
# Low Power
TGLF_LOWPOW_INPUTS = [
    "rmin_loc", "drmajdx_loc", "kappa_loc", "delta_loc", "q_loc",
    "q_prime_loc", "vexb_shear", "rlns_1", "rlts_1", "rlns_2",
    "rlts_2", "taus_2", "as_2", "rlns_3", "zeff", "betae", "xnue"
]
TGLF_LOWPOW_OUTPUTS = ['efe', 'efi', 'pfe', 'pfi']
# Mixed Power
TGLF_MIXED_INPUTS = [
    "rmin_loc", "drmajdx_loc", "kappa_loc", "delta_loc", "q_loc",
    "q_prime_loc", "vexb_shear", "rlns_1", "rlts_1", "rlns_2",
    "rlts_2", "taus_2", "as_2", "rlns_3", "zeff", "betae", "xnue"
]
TGLF_MIXED_OUTPUTS = ['efe', 'efi', 'pfe', 'pfi']


__all__ = [
    'CSVRegressionDataset', 'get_avalanche_csv_regression_datasets',
    'QUALIKIZ_HIGHPOW_INPUTS', 'QUALIKIZ_HIGHPOW_OUTPUTS',
    'QUALIKIZ_LOWPOW_INPUTS', 'QUALIKIZ_LOWPOW_OUTPUTS',
    'QUALIKIZ_MIXED_INPUTS', 'QUALIKIZ_MIXED_OUTPUTS',
    'TGLF_HIGHPOW_INPUTS', 'TGLF_HIGHPOW_OUTPUTS',
    'TGLF_LOWPOW_INPUTS', 'TGLF_LOWPOW_OUTPUTS',
    'TGLF_MIXED_INPUTS', 'TGLF_MIXED_OUTPUTS',
    'make_complete_dataset',
]