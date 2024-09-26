"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: uni_samples.py
@time: 2024/9/9 8:04
"""
import numpy as np
import pandas as pd


def uni_array(array_del, max_value, min_value):
    return (2.0 * (array_del - min_value) / (max_value - min_value)) - 1.0


def uni_input(ori_sam_filepath, profile_info_path):
    df_section = pd.read_csv(profile_info_path)

    area_info = df_section.iloc[:, 5].to_numpy()  # 将面积存入数组
    inertia_moment = df_section.iloc[:, 6].to_numpy()  # 将惯性矩存入数组

    area_info = uni_array(area_info, 286.0, 39.1)
    inertia_moment = uni_array(inertia_moment, 210600.0, 3892.0)

    input_ori = np.genfromtxt(ori_sam_filepath, delimiter=",", dtype=int)

    input_array = np.zeros((input_ori.shape[0], input_ori.shape[1]*2))

    for row in range(input_ori.shape[0]):
        for col in range(input_ori.shape[1]):
            section_label = input_ori[row, col]

            input_array[row, col * 2] = area_info[section_label]
            input_array[row, col * 2 + 1] = inertia_moment[section_label]

    # # 设置打印选项以避免科学计数法
    # np.set_printoptions(suppress=True)
    # np.savetxt("./source_data/eight_story_frame/source_data/uni_source_data/input_uni.csv", input_array, delimiter=',', fmt='%.9f')

    return input_array


def find_output_max_min(ori_output_filepath):
    out_data = np.genfromtxt(ori_output_filepath, delimiter=",")

    max_IDR = np.max(out_data)
    min_IDR = np.min(out_data)

    return max_IDR, min_IDR


def uni_output(ori_output_filepath, max_IDR_value, min_IDR_value):
    out_data = np.genfromtxt(ori_output_filepath, delimiter=",")

    return uni_array(out_data, max_IDR_value, min_IDR_value)


def main():
    # ori_sam_filepath = "./source_data/eight_story_frame/source_data/samples/sample.csv"
    # ori_output_filepath = "./source_data/eight_story_frame/source_data/outputs/output_train.csv"

    ori_sam_filepath = "./source_data/eight_story_frame/source_data/samples/sample_test.csv"
    ori_output_filepath = "./source_data/eight_story_frame/source_data/outputs/output_test.csv"

    profile_info_path = "./source_data/eight_story_frame/sec_info.csv"

    input_array = uni_input(ori_sam_filepath, profile_info_path)

    # max_IDR_value, min_IDR_value = find_output_max_min(ori_output_filepath)  # 0.012853373, 0.000577814
    max_IDR_value, min_IDR_value = 0.012853373, 0.000577814

    output_array = uni_output(ori_output_filepath, max_IDR_value, min_IDR_value)

    surrogate_model_input_uni = np.concatenate((input_array, output_array), axis=1)

    np.set_printoptions(suppress=True)
    # np.savetxt("./source_data/eight_story_frame/source_data/uni_source_data/input_train_uni.csv", surrogate_model_input_uni, delimiter=',', fmt='%.9f')
    np.savetxt("./source_data/eight_story_frame/source_data/uni_source_data/input_test_uni.csv", surrogate_model_input_uni, delimiter=',', fmt='%.9f')


if __name__ == '__main__':
    main()