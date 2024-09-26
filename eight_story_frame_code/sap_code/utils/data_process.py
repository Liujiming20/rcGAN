import numpy as np


class LabelProcessor():
    def __init__(self):
        self.max = 19.0
        self.min = 0.0

    def pre_process(self, data_process):
        return (2.0 * (data_process - self.min) / (self.max - self.min)) - 1.0

    def back_process(self, data_process):
        return ((data_process+1.0) * (self.max - self.min) / 2.0) + self.min


class ValueProcessor():
    def __init__(self):
        self.max = 0.012853373
        self.min = 0.000577814

    def pre_process(self, data_process):
        return (2.0 * (data_process - self.min) / (self.max - self.min)) - 1.0

    def back_process(self, data_process):
        return ((data_process+1.0) * (self.max - self.min) / 2.0) + self.min