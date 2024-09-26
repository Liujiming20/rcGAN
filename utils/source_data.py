import numpy as np


class DataSource():
    def __init__(self, train_data_filepath):
        self.data = np.loadtxt(train_data_filepath, delimiter=',', dtype=np.float32)

        self.x_1 = self.data[:, 0:6]
        self.x_2 = self.data[:, 6:27]
        self.x_3 = self.data[:, 27:29]
        self.y = self.data[:, 29:]
