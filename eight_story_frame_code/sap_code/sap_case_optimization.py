import random

from matplotlib import pyplot as plt
from smt.applications.mixed_integer import MixedIntegerSamplingMethod
from smt.utils.design_space import DesignSpace, IntegerVariable
from smt.sampling_methods import LHS

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

import torch
from torch.utils import data

from eight_story_frame_code.sap_code.utils.data_process import ValueProcessor
from eight_story_frame_code.sap_code.utils.generator import Generator
from eight_story_frame_code.sap_code.utils.my_dataset import DatasetFromPop
from utils.regression_model import Regression03


def initialize_population(samples_filepath, LHS_seed, pop_num):
    group1_section = IntegerVariable(10, 19)
    group2_section = IntegerVariable(10, 19)
    group3_section = IntegerVariable(10, 19)
    group4_section = IntegerVariable(9, 19)
    group5_section = IntegerVariable(9, 19)
    group6_section = IntegerVariable(8, 19)
    group7_section = IntegerVariable(8, 19)
    group8_section = IntegerVariable(8, 19)
    group9_section = IntegerVariable(0, 19)
    group10_section = IntegerVariable(0, 19)
    group11_section = IntegerVariable(0, 19)
    group12_section = IntegerVariable(0, 19)
    group13_section = IntegerVariable(0, 19)
    group14_section = IntegerVariable(0, 19)
    group15_section = IntegerVariable(0, 19)
    group16_section = IntegerVariable(0, 19)

    design_space = DesignSpace([group1_section,
                                group2_section,
                                group3_section,
                                group4_section,
                                group5_section,
                                group6_section,
                                group7_section,
                                group8_section,
                                group9_section,
                                group10_section,
                                group11_section,
                                group12_section,
                                group13_section,
                                group14_section,
                                group15_section,
                                group16_section])

    sampling = MixedIntegerSamplingMethod(LHS, design_space, criterion="ese", random_state=LHS_seed)
    n_doe = pop_num
    Xt = sampling(n_doe)

    samples = np.array(design_space.decode_values(Xt)).astype(int)

    np.savetxt(samples_filepath, samples, delimiter = ",")

    return samples


def uni_array(array_del, max_value, min_value):
    return (2.0 * (array_del - min_value) / (max_value - min_value)) - 1.0


def converse_pop_to_model_input(pop_ori):
    input_ori = pop_ori

    profile_info_path = "./source_data/eight_story_frame/sec_info.csv"
    df_section = pd.read_csv(profile_info_path)

    area_info = df_section.iloc[:, 5].to_numpy()  # 将面积存入数组
    inertia_moment = df_section.iloc[:, 6].to_numpy()  # 将惯性矩存入数组

    area_info_uni = uni_array(area_info, 286.0, 39.1)
    inertia_moment_uni = uni_array(inertia_moment, 210600.0, 3892.0)

    input_array = np.zeros((input_ori.shape[0], input_ori.shape[1] * 2))
    area_array = np.zeros_like(input_ori)

    for row in range(input_ori.shape[0]):
        for col in range(input_ori.shape[1]):
            section_label = input_ori[row, col]

            input_array[row, col * 2] = area_info_uni[section_label]
            input_array[row, col * 2 + 1] = inertia_moment_uni[section_label]

            area_array[row, col] = area_info[section_label]  # 面积单位为cm^2

    return input_array, area_array*100.0


def cal_IDR_constraint(pop_input_torch):
    test_dataloader = data.DataLoader(pop_input_torch, batch_size=pop_input_torch.len, shuffle=False)

    regressor = Regression03(8, 32, 1, True, 0.15, net_para_reg[0], net_para_reg[1], net_para_reg[2], net_para_reg[3], net_para_reg[4], net_para_reg[5])
    regressor.load_state_dict(torch.load(reg_pth_path))
    generator = Generator(8, 8, 1, True, 0.025, net_gen_para[0], net_gen_para[1], net_gen_para[2], net_gen_para[3], net_gen_para[4], net_gen_para[5])
    generator.load_state_dict(torch.load(gen_pth_path))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        regressor.to(device)
        generator.to(device)

    # 创建sap案例的归一化处理器
    value_processor = ValueProcessor()

    regressor.eval()
    generator.eval()
    with torch.no_grad():
        for i, x_input in enumerate(test_dataloader):
            x_input = x_input.to(device)

            z_noise_test = torch.zeros((pop_input_torch.len, 8), dtype=torch.float32)
            z_noise = z_noise_test.to(device)

            label_test = regressor(x_input)
            y_predict_norm = generator(z_noise, label_test)
            y_predict_norm = y_predict_norm.clone().detach().to("cpu").numpy()
            y_predict = value_processor.back_process(y_predict_norm)

    # 使用 count_nonzero 计算每一行大于 threshold 的元素个数
    vio_array = np.count_nonzero(y_predict > 0.0045, axis=1)
    fun_penalty = (1 + 1.0 * vio_array)**2.0

    return fun_penalty


def cal_steel_weight(area_array):
    group_total_length_array = np.array([17500.0, 17500.0, 17500.0, 10500.0, 10500.0, 10500.0, 10500.0, 10500.0, 16000.0, 16000.0, 16000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0])

    steel_volume_array = np.sum(area_array * group_total_length_array, axis=1)

    return steel_volume_array*steel_density


# 批量评估函数 - 一次评估整代种群
def evaluate_population(population):
    global gen_count
    gen_count += 1

    # 提取设计变量并形成二维 NumPy 数组
    design_variables = np.array([ind[:] for ind in population])

    # 根据原始种群(截面型号)获取代理模型输入以及截面积
    input_array, area_array = converse_pop_to_model_input(design_variables)

    # 计算层间漂移约束违反的罚函数
    pop_input_torch = DatasetFromPop(input_array)
    fun_penalty = cal_IDR_constraint(pop_input_torch)

    # 计算种群中各个个体对应的结构自重
    steel_weight_non_penalty = cal_steel_weight(area_array)

    # 计算适应度函数
    fitness_values = fun_penalty * steel_weight_non_penalty

    # 将最小的优化结果记录下来
    min_fitness_value = np.min(fitness_values)
    if len(min_weight_each_generation_list) != 0 and min_fitness_value > min_weight_each_generation_list[-1]:
        min_fitness_value = min_weight_each_generation_list[-1]
    min_weight_each_generation_list.append(min_fitness_value)

    # 将适应度值赋给每个个体
    for ind, fitness in zip(population, fitness_values):
        ind.fitness.values = (fitness,)


def check_and_repair_individual(individual):
    # 对前三列应用约束 [10, 19] 并转换为整数
    individual[0:3] = np.clip(individual[0:3], 10, 19).astype(int)

    # 对第4、5列应用约束 [9, 19] 并转换为整数
    individual[3:5] = np.clip(individual[3:5], 9, 19).astype(int)

    # 对第6到8列应用约束 [8, 19] 并转换为整数
    individual[5:8] = np.clip(individual[5:8], 8, 19).astype(int)

    # 对最后8列（第9到16列）应用约束 [0, 19] 并转换为整数
    individual[8:16] = np.clip(individual[8:16], 0, 19).astype(int)

    # 从第八维度向第一维度检查并调整
    for i in range(7, 0, -1):
        if individual[i] > individual[i - 1]:
            individual[i - 1] = individual[i]  # 调整前一维度以满足约束

    return individual


# 校核并修复每个 offspring 中 individual 对象的约束
def check_and_repair_population(offspring):
    for ind in offspring:
        check_and_repair_individual(ind)
    return offspring


# 遗传算法主流程
def genetic_algorithm(file_path, LHS_seed):
    # 定义适应度和个体
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # 创建基础工具箱
    toolbox = base.Toolbox()

    # 初始化种群
    initial_population = initialize_population(file_path, LHS_seed, pop_size)
    population = [creator.Individual(ind) for ind in initial_population]

    # 定义操作：适应度评估、选择、交叉和变异
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    # 评估初始种群
    evaluate_population(population)

    # 开始演化
    for gen in range(generation):
        # 计算需要保留的个体数量
        num_elites = int(len(population) * 0.1)

        # 选择并保留表现最好的个体
        elites = tools.selBest(population, num_elites)

        # 选择下一代个体
        offspring = toolbox.select(population, len(population) - num_elites)
        offspring = list(map(toolbox.clone, offspring))

        # 应用交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < cross_pro:  # 交叉概率0.5
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                # 将交叉后的个体的值转换为整数
                child1[:] = [int(round(x)) for x in child1]
                child2[:] = [int(round(x)) for x in child2]

        for mutant in offspring:
            if np.random.rand() < mut_pro:  # 变异概率0.2
                toolbox.mutate(mutant)
                del mutant.fitness.values
                # 将变异后的个体的值转换为整数
                mutant[:] = [int(round(x)) for x in mutant]

        # 校核新生代个体约束
        offspring = check_and_repair_population(offspring)

        # 评估新一代种群
        evaluate_population(offspring)

        # 替换种群，并保留精英个体
        population[:] = elites + offspring

    # 输出最优解
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is:", best_ind)
    print("With fitness:", best_ind.fitness.values[0])


def main():
    np.random.seed(62)
    random.seed(62)

    global steel_density
    steel_density = 7.7 * 10 ** (-5)  # 单位N/mm^3

    global net_para_reg, reg_pth_path, net_gen_para, gen_pth_path
    net_para_reg = [3, 64, 128, 256, 32, 16]
    reg_pth_path = "./result/eight_story_frame/networks/pre-train_reg/pre-train_reg.pth"
    net_gen_para = [2, 128, 512, 64, 32, 16]
    gen_pth_path = "./result/eight_story_frame/networks/gen/generator_lf_smoothL1_atten1.pth"

    global gen_count, min_weight_each_generation_list
    gen_count = -1
    min_weight_each_generation_list = []

    # 定义交叉率, 基因变异率, 变异发生概率
    global cross_pro, indpb, mut_pro, tournsize
    cross_pro = 0.5
    indpb = 0.5
    mut_pro = 0.2
    tournsize = 3

    # 定义初始种群参数
    LHS_seed = 62
    samples_filepath = "./source_data/eight_story_frame/source_data/optimization_sap/initial_pop/samples.csv"

    # 执行优化
    global pop_size, generation
    pop_size = 100
    generation = 100

    genetic_algorithm(samples_filepath, LHS_seed)

    # 设置打印选项以避免科学计数法
    np.set_printoptions(suppress=True)
    np.savetxt("./result/eight_story_frame/optimization_sap/{}_opt_process.csv".format(LHS_seed),np.array(min_weight_each_generation_list), fmt='%.3f')

    # gen_array = np.arange(generation+1)
    # plt.plot(gen_array, min_weight_each_generation_list, marker='o', linestyle='-', color='pink')
    # plt.grid(True)
    # plt.show()


if __name__ == '__main__':
    main()