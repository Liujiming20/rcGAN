import numpy as np
import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.query(x)
        queries = queries.view(len(queries), -1, 1)
        keys = self.key(x)
        keys = keys.view(len(queries), -1, 1)
        values = self.value(x)
        values = values.view(len(queries), -1, 1)
        scores = torch.bmm(queries, keys.transpose(2, 1)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        weighted = weighted.view(len(weighted), -1)
        return weighted
