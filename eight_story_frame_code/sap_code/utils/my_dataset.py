import numpy as np

import torch
from torch.utils import data


class DatasetFromCSV(data.Dataset):
    def __init__(self, train_data_filepath):
        input = np.loadtxt(train_data_filepath, delimiter=',', dtype=np.float32)

        self.len = input.shape[0]
        self.values = torch.from_numpy(input[:, 32:])
        self.labels = torch.from_numpy(input[:, :32])

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