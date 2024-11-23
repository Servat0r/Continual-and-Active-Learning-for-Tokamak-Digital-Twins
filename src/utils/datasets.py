from typing import Any, Callable
import os
import pandas as pd
import torch
from avalanche.benchmarks import AvalancheDataset
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import TransformGroups, DataAttribute


class CSVRegressionDataset(Dataset):
    def __init__(
            self, data, input_columns: list[str], output_columns: list[str],
            transform=None, target_transform=None, filter_by: dict[str, list] = None,
            float_precision: str = 'float32', device=None,
            filter_by_leq: dict[str, int | float] = None,
            filter_by_geq: dict[str, int | float] = None,
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
        if filter_by_leq:
            for column, value in filter_by_leq.items():
                self.data = self.data[self.data[column] <= value]
        if filter_by_geq:
            for column, value in filter_by_geq.items():
                self.data = self.data[self.data[column] >= value]
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


def _is_not_null(row, output_columns):
    return any([row[column] != 0.0 for column in output_columns])


def make_datasets(src_path, src_filename, dest_path, output_columns):
    df = pd.read_csv(src_path + '/' + src_filename)
    df['has_turbulence'] = df.apply(lambda row: _is_not_null(row, output_columns), axis=1)
    complete_df = df.copy()
    not_null_df = df[df['has_turbulence'] == True]
    not_null_df = not_null_df.drop(columns=['has_turbulence'])
    classification_df = df.drop(columns=output_columns)
    print(
        f"Complete dataset has {len(complete_df)} elements",
        f"Not Null dataset has {len(not_null_df)} elements",
        f"Classification dataset has {len(classification_df)} elements",
        sep='\n',
    )
    os.makedirs(dest_path, exist_ok=True)
    complete_df.to_csv(path_or_buf=f'{dest_path}/complete_dataset.csv', index=False)
    not_null_df.to_csv(path_or_buf=f'{dest_path}/not_null_dataset.csv', index=False)
    classification_df.to_csv(path_or_buf=f'{dest_path}/classification_dataset.csv', index=False)


def get_avalanche_csv_regression_datasets(
        train_data, test_data, input_columns: list[str], output_columns: list[str], # todo modify later
        transform=None, target_transform=None, filter_by: dict[str, list] = None,
        float_precision: str = 'float32', device=None, *, indices: list[int] | None = None,
        data_attributes: list[DataAttribute] | None = None, transform_groups: TransformGroups | None = None,
        frozen_transform_groups: TransformGroups | None = None, collate_fn: Callable[[list], Any] | None = None,
        filter_by_leq: dict[str, int | float] = None, filter_by_geq: dict[str, int | float] = None,
):
    base_train_dataset = CSVRegressionDataset(
        train_data, input_columns=input_columns, output_columns=output_columns, transform=transform,
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
    'make_datasets',
]