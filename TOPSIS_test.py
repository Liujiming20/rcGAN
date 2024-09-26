import csv

import numpy as np


def normalize_decision_matrix(matrix):
    norm_matrix = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        min_val = np.min(matrix[:, j])
        max_val = np.max(matrix[:, j])
        norm_matrix[:, j] = (max_val - matrix[:, j]) / (max_val - min_val)  # 适用于最小化目标函数
    return norm_matrix


def get_ideal_solutions(weighted_matrix):
    pis = np.max(weighted_matrix, axis=0)  # 正理想解
    nis = np.min(weighted_matrix, axis=0)  # 负理想解
    return pis, nis


def calculate_distances(weighted_matrix, pis, nis):
    distance_to_pis = np.sqrt(np.sum((weighted_matrix - pis) ** 2, axis=1))  # 到正理想解的距离
    distance_to_nis = np.sqrt(np.sum((weighted_matrix - nis) ** 2, axis=1))  # 到负理想解的距离
    return distance_to_pis, distance_to_nis


def calculate_relative_closeness(distance_to_pis, distance_to_nis):
    siw = distance_to_nis / (distance_to_pis + distance_to_nis)  # 相对接近度
    best_index = np.argmax(siw)  # 最优设计的索引
    return siw, best_index


def main():
    fea_filepath = "./NSGA/opt_result/LHS_{}/gen500_front0_feature.csv"
    obj_filepath = "./NSGA/opt_result/LHS_{}/gen500_front0_objective.csv"

    out_filepath = "./NSGA/opt_result/schema.csv"

    weights = [1/5, 1/5, 3/5]

    # for seed_num in [10, 11, 12, 13, 14, 37, 38, 39]:
    for seed_num in [21]:
    # for seed_num in range(25,30):
        # 读取方案
        all_schemes = np.genfromtxt(fea_filepath.format(str(seed_num)), delimiter=",")

        # 读取Pareto前沿的目标值，生成决策矩阵
        pareto_matrix = np.genfromtxt(obj_filepath.format(str(seed_num)), delimiter=",")

        # 归一化决策矩阵
        normalized_matrix = normalize_decision_matrix(pareto_matrix)

        # 加权矩阵
        weighted_decision_matrix = normalized_matrix * weights

        # 计算正理想解和负理想解
        pis, nis = get_ideal_solutions(weighted_decision_matrix)

        # 计算各方案到理想解的距离
        distance_to_pis, distance_to_nis = calculate_distances(weighted_decision_matrix, pis, nis)

        # 计算相对接近度SI_w，并选出最优设计
        siw, best_index = calculate_relative_closeness(distance_to_pis, distance_to_nis)

        print("最优方案的索引为{};".format(best_index), "该方案各目标值为：", pareto_matrix[best_index])

        with open(out_filepath, "a+", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(all_schemes[best_index])


if __name__ == '__main__':
    main()