from typing import Any, Callable
import pandas as pd
import torch
from avalanche.benchmarks import AvalancheDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import TransformGroups, DataAttribute


class CSVRegressionDataset(Dataset):
    def __init__(
            self, data, input_columns: list[str], output_columns: list[str],
            transform=None, target_transform=None, filter_by: dict[str, list] = None,
            float_precision: str = 'float32', device=None,
    ):
        if (device is None) or (device == 'gpu'):
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        elif not device.startswith('cuda'):
            device = 'cpu'
        if torch.cuda.is_available() and (device != 'cpu'):
            device_index = torch.cuda.current_device()
            print(f"Using CUDA device {device}: ({torch.cuda.get_device_name(device_index)}) ...")
        else:
            print(f"Using cpu ...")
        # Load the CSV file
        self.data = data
        if filter_by:
            for column, values in filter_by.items():
                self.data = self.data[self.data[column].isin(values)]
        self.transform = transform
        self.target_transform = target_transform
        self.inputs = torch.tensor(self.data[input_columns].values.astype(float_precision)).to(device)
        self.targets = torch.tensor(self.data[output_columns].values.astype(float_precision)).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.inputs[idx], self.targets[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def get_data(self, copy=False):
        if copy:
            return self.data.clone()
        else:
            return self.data


def get_avalanche_csv_regression_datasets(
        csv_file, input_columns: list[str], output_columns: list[str], test_size: float = 0.2, # todo modify later
        transform=None, target_transform=None, filter_by: dict[str, list] = None,
        float_precision: str = 'float32', device=None, *, indices: list[int] | None = None,
        data_attributes: list[DataAttribute] | None = None, transform_groups: TransformGroups | None = None,
        frozen_transform_groups: TransformGroups | None = None, collate_fn: Callable[[list], Any] | None = None
):
    data = pd.read_csv(csv_file)
    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    base_train_dataset = CSVRegressionDataset(
        train_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
        target_transform=target_transform, filter_by=filter_by, float_precision=float_precision, device=device,
    )
    base_test_dataset = CSVRegressionDataset(
        test_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
        target_transform=target_transform, filter_by=filter_by, float_precision=float_precision, device=device,
    )
    train_dataset = AvalancheDataset(
        [base_train_dataset], indices=indices, data_attributes=data_attributes,
        transform_groups=transform_groups, frozen_transform_groups=frozen_transform_groups,
        collate_fn=collate_fn
    )
    test_dataset = AvalancheDataset(
        [base_test_dataset], indices=indices, data_attributes=data_attributes,
        transform_groups=transform_groups, frozen_transform_groups=frozen_transform_groups,
        collate_fn=collate_fn
    )
    return train_dataset, test_dataset


# Constants for loading baseline fields
# High Power
BASELINE_HIGHPOW_INPUTS = [
    'ane', 'ate', 'autor', 'machtor', 'x', 'zeff', 'gammae', 'q', 'smag',
    'alpha', 'ani1', 'ati0', 'normni1', 'ti_te0', 'lognustar'
]
BASELINE_HIGHPOW_OUTPUTS = ['efi', 'efe', 'pfi', 'pfe']
# Low Power
BASELINE_LOWPOW_INPUTS = [
    'ane', 'ate', 'x', 'q', 'smag', 'alpha', 'ani1', 'ati0', 'normni1', 'zeff', 'lognustar'
]
BASELINE_LOWPOW_OUTPUTS = ['efi', 'efe', 'pfi', 'pfe']


__all__ = [
    'CSVRegressionDataset', 'get_avalanche_csv_regression_datasets',
    'BASELINE_HIGHPOW_INPUTS', 'BASELINE_HIGHPOW_OUTPUTS',
    'BASELINE_LOWPOW_INPUTS', 'BASELINE_LOWPOW_OUTPUTS',
]