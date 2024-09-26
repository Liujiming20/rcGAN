import numpy as np

import torch
from torch.utils import data

from utils.data_process import process_source_data


class DatasetFromSourceData(data.Dataset):
    def __init__(self, source_data, label_processor, value_processor):
        x_norm, y_norm, self.len = process_source_data(source_data, label_processor, value_processor)

        self.values = torch.from_numpy(y_norm[:, :])

        self.labels = torch.from_numpy(x_norm[:, :])

    def __getitem__(self, item):
        value = self.values[item]
        label = self.labels[item]

        return value, label

    def __len__(self):
        return self.len


class DatasetFromCSV(data.Dataset):
    def __init__(self, train_data_x_filepath, train_data_y_filepath):
        x_input = np.loadtxt(train_data_x_filepath, delimiter=',', dtype=np.float32)
        y_norm = np.loadtxt(train_data_y_filepath, delimiter=',', dtype=np.float32)

        self.len = x_input.shape[0]
        self.values = torch.from_numpy(y_norm[:, :])
        self.labels = torch.from_numpy(x_input[:, :])

    def __getitem__(self, item):
        value = self.values[item]
        label = self.labels[item]

        return value, label

    def __len__(self):
        return self.len


class DatasetFromPop(data.Dataset):
    def __init__(self, pop_data_normal):
        pop_data_normal = pop_data_normal.astype(np.float32)
        self.len = pop_data_normal.shape[0]
        self.labels = torch.from_numpy(pop_data_normal[:, :])

    def __getitem__(self, item):
        label = self.labels[item]

        return label

    def __len__(self):
        return self.len