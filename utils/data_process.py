import csv

import numpy as np


class ProfileProcessor():
    def __init__(self, x_1):
        profile_property_beam = np.genfromtxt("./source_data/input_para/profile_property/beam_profile_property.csv", delimiter=",")
        profile_property_column = np.genfromtxt("./source_data/input_para/profile_property/column_profile_property.csv", delimiter=",")

        profile_vectors = np.zeros((len(x_1), 12))

        profile_index_vectors = x_1.astype(int)

        # x_1的label是profile_property数组的下标+1
        profile_index_vectors[:, :] -= 1

        # 利用numpy的索引机制快速翻译label和截面属性对应关系
        profile_vectors[:, :3] = profile_property_beam[profile_index_vectors[:, :3], 1]  # Top-Bot梁截面面积
        profile_vectors[:, 3:6] = profile_property_beam[profile_index_vectors[:, :3], 2]  # Top-Bot梁截面惯性矩
        profile_vectors[:, 6:9] = profile_property_column[profile_index_vectors[:, 3:6], 1]  # Top-Bot柱截面面积
        profile_vectors[:, 9:] = profile_property_column[profile_index_vectors[:, 3:6], 2]  # Top-Bot柱截面惯性矩

        self.x_1_conversion = profile_vectors


class VectorProcessor():
    def __init__(self, x_2):
        vectors = np.zeros((len(x_2), 21))
        for row in range(len(x_2)):
            vector = x_2[row]
            vector = np.unique(vector)

            for num in vector:
                num_int = int(num)
                if num_int != -1:
                    vectors[row, num_int] = 1
        # print(vectors)

        x_2_encoder = np.zeros((len(x_2), 3))
        for row in range(len(x_2)):
            for i in range(3):
                x_2_encoder[row, i] = 2**0*vectors[row, 0+i*7] + 2**1*vectors[row, 1+i*7] + 2**2*vectors[row, 2+i*7] + 2**3*vectors[row, 3+i*7] + 2**4*vectors[row, 4+i*7] + 2**5*vectors[row, 5+i*7] + 2**6*vectors[row, 6+i*7]  # 将21维one-hot编码的每七个维度通过二进制转化为数据

        self.x_2_vectors = x_2_encoder


class LabelProcessor():
    def __init__(self):
        # self.max = np.array([22.0,23.0,24.0,22.0,23.0,24.0,127.0,127.0,127.0,1890.0,1875.0])
        # self.min = np.array([7.0,8.0,9.0,5.0,7.0,8.0,0.0,0.0,0.0,630.0,625.0])
        self.max = np.array([18387.06, 18387.06, 21032.216, 728405000.0, 861599100.0, 1111337900.0, 29870.908, 34580.58, 38257.99, 156503020.0, 201039780.0, 225597430.0, 127.0,127.0,127.0,1890.0,1875.0])
        self.min = np.array([4909.6676, 5703.2144, 6645.148, 59937325.0, 70759342.0, 86992368.0, 8451.596, 9419.336, 10967.72, 20811571.0, 23433829.0, 44536763.0, 0.0,0.0,0.0,630.0,625.0])

    def pre_process(self, data_process):
        return (2 * (data_process - self.min) / (self.max - self.min)) - 1

    def back_process(self, data_process):
        return ((data_process+1) * (self.max - self.min) / 2) + self.min


class ValueProcessor():
    def __init__(self, source_data_value):
        self.max = np.max(source_data_value, axis=0)
        self.min = np.min(source_data_value, axis=0)

    def pre_process(self, data_process):
        return (2 * (data_process - self.min) / (self.max - self.min)) - 1

    def back_process(self, data_process):
        return ((data_process+1) * (self.max - self.min) / 2) + self.min


def process_source_data(source_data, label_processor, value_processor):
    # 将截面号处理为对应的面积和惯性矩
    x_1_pro = ProfileProcessor(source_data.x_1)
    # 将斜撑向量编码
    x_2 = VectorProcessor(source_data.x_2)
    data_encoder = np.hstack((x_1_pro.x_1_conversion, x_2.x_2_vectors, source_data.x_3))

    # print(data_encoder[-1])
    # print(source_data.y[-1])
    data_norm_label = label_processor.pre_process(data_encoder)
    data_norm_label = data_norm_label.astype(np.float32)

    data_norm_value = value_processor.pre_process(source_data.y)

    return data_norm_label, data_norm_value, data_norm_label.shape[0]


def process_source_data_to_csv(source_data, label_processor, value_processor, x_output_filepath, y_output_filepath):
    x_1_pro = ProfileProcessor(source_data.x_1)
    x_2 = VectorProcessor(source_data.x_2)
    data_encoder = np.hstack((x_1_pro.x_1_conversion, x_2.x_2_vectors, source_data.x_3))

    data_norm_label = label_processor.pre_process(data_encoder)

    data_norm_value = value_processor.pre_process(source_data.y)

    with open(x_output_filepath, "w+", newline='') as x_csvfile:
        writer = csv.writer(x_csvfile)
        writer.writerows(data_norm_label)

    with open(y_output_filepath, "w+", newline='') as y_csvfile:
        writer = csv.writer(y_csvfile)
        writer.writerows(data_norm_value)