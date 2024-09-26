import math
import random

import numpy as np
import torch
from torch.utils import data

from utils.generator import Generator04
from utils.regression_model import Regression03
from utils.data_process import ValueProcessor
from utils.my_dataset import DatasetFromPop
from utils.source_data import DataSource


def cal_steel_cost(input_vector):
    beam_profile_areas = np.loadtxt("./source_data/multi_opti_config/beam_profile_area.csv", delimiter=",", dtype=float)
    column_profile_areas = np.loadtxt("./source_data/multi_opti_config/column_profile_area.csv", delimiter=",", dtype=float)

    beam_area_t = beam_profile_areas[int(input_vector[0]) - 1][1]  # top
    beam_area_m = beam_profile_areas[int(input_vector[1]) - 1][1]  # middle
    beam_area_b = beam_profile_areas[int(input_vector[2]) - 1][1]  # bottom

    column_area_t = column_profile_areas[int(input_vector[3]) - 1][1]
    column_area_m = column_profile_areas[int(input_vector[4]) - 1][1]
    column_area_b = column_profile_areas[int(input_vector[5]) - 1][1]

    beam_volume = (beam_area_t/10**6 * 2 + beam_area_m/10**6 * 3 + beam_area_b/10**6 * 2) * 228  # 2 3 2 represent the num of floors, respectively.
    column_volume = (column_area_t/10**6 * 2 + column_area_m/10**6 * 3 + column_area_b/10**6) * 84 + column_area_b/10**6 * 96

    beam_cost = beam_volume * 7850 * 4.718
    column_cost = column_volume * 7850 * 4.718

    return beam_cost+column_cost


def cal_damper_cost(input_vector):
    damper_position_list = input_vector[6:27]
    damper_position_list = list(set(damper_position_list))
    damper_num = len(damper_position_list)
    if -1 in damper_position_list:
        damper_num = len(damper_position_list) - 1

    damper_F = input_vector[28] * 1**0.38
    stroke = 70

    price_damper = 6295.552 * math.exp(-0.5*((math.log(damper_F*1000, math.e)-14.732)/0.545)**2) + 1.06023*10**47 * math.exp(-0.5*((math.log(stroke, math.e)-464.046)/32.452)**2)

    length = 0
    for p in damper_position_list:
        if p == 0:
            length += 5
        elif 0<p<7:
            length += 4.61
        elif p == 7 or p == 14:
            length += 7.21
        elif 7<p<14 or p>14:
            length += 6.95
    price_braces = input_vector[27]/(200*10**6) * length * 7850 * 4.718 * 5

    return (price_damper*damper_num + price_braces) *4*2


def cal_cost(ini_pop_data):
    cost_array = np.array([])
    for i in range(ini_pop_data.shape[0]):
        input_vector = ini_pop_data[i].tolist()
        cost = cal_steel_cost(input_vector) + cal_damper_cost(input_vector)

        if i == 0:
            cost_array = np.array([cost])
        else:
            cost_array = np.append(cost_array,[cost])

    return cost_array.reshape(-1,1)


def cal_DIR_acce(ini_pop_data, initial_pop_normal, net_para_reg, reg_pth_path, net_para_gen, gen_pth_path, pop, ini_option):
    torch.manual_seed(111)

    value_dim = 3
    label_dim = 3
    atten_para = 2  # 0-3
    drop = True
    # drop_para = False

    source_data = DataSource("./source_data/input_para/input_para.csv")
    value_processor = ValueProcessor(source_data.y)

    test_dataset = DatasetFromPop(initial_pop_normal)
    test_dataloader = data.DataLoader(test_dataset, batch_size=pop, shuffle=False)

    regressor = Regression03(3, 17, 1, True, 0.15, net_para_reg[0], net_para_reg[1], net_para_reg[2], net_para_reg[3], net_para_reg[4], net_para_reg[5])
    regressor.load_state_dict(torch.load(reg_pth_path))
    generator = Generator04(value_dim, label_dim, atten_para, drop, 0.2, net_para_gen[0], net_para_gen[1], net_para_gen[2], net_para_gen[3], net_para_gen[4], net_para_gen[5])
    generator.load_state_dict(torch.load(gen_pth_path))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        regressor.to(device)
        generator.to(device)

    y_predict_list = np.array([[]])  # Store eleven wave calculations

    regressor.eval()
    generator.eval()
    with torch.no_grad():
        for i, x_input in enumerate(test_dataloader):
            x_input = x_input.to(device)

            # z_noise_test = torch.rand(x_input.shape[0], 3, dtype=torch.float32) * 2 - 1
            # z_noise_test = torch.normal(0, 1, (x_input.shape[0], 3), dtype=torch.float32) / 3
            z_noise_test = torch.zeros((x_input.shape[0], 3), dtype=torch.float32)
            z_noise_test = z_noise_test.to(device)

            label_test = regressor(x_input)
            y_predict_norm = generator(z_noise_test, label_test)
            y_predict_norm = y_predict_norm.clone().detach().to("cpu").numpy()
            y_predict = value_processor.back_process(y_predict_norm)

            if i == 0:
                y_predict_list = y_predict
            else:
                y_predict_list = np.concatenate((y_predict_list, y_predict), axis=0)

    # fre_array = y_predict_11[:, 0]
    # DIR_array = y_predict_11[:, 1]
    # acce_array = y_predict_11[:, 2]
    #
    # avg_fre = np.average(fre_array.reshape(-1, 11), axis=1)
    # avg_DIR_no_fla = np.average(DIR_array.reshape(-1, 11), axis=1)
    # avg_DIR = avg_DIR_no_fla.reshape(-1,1)
    # avg_acce = np.average(acce_array.reshape(-1, 11), axis=1).reshape(-1,1)

    avg_fre = y_predict_list[:, 0]
    avg_DIR_no_reshape = y_predict_list[:, 1]
    avg_DIR = avg_DIR_no_reshape.reshape(-1,1)
    avg_acce = y_predict_list[:, 2].reshape(-1,1)

    con_indicator_predict = np.concatenate((avg_DIR, avg_acce), axis=1)
    # max_indicator = np.max(con_indicator_predict, axis=0)

    # check the union constraint of frequency and damper position
    index_list = np.argwhere(avg_fre > 2.3)
    break_constraint_index = []
    for i in range(index_list.shape[0]):
        index = index_list[i][0]
        input_para_list = ini_pop_data[index][6:27].tolist()
        damper_position_list = list(set(input_para_list))

        if 0 in damper_position_list or 7 in damper_position_list or 14 in damper_position_list:
            pass
        else:
            # con_indicator_predict[index] = max_indicator
            break_constraint_index.append(index)
            continue

        if 1 in damper_position_list or 8 in damper_position_list or 15 in damper_position_list:
            pass
        else:
            # con_indicator_predict[index] = max_indicator
            break_constraint_index.append(index)
            continue

        if 2 in damper_position_list or 9 in damper_position_list or 16 in damper_position_list:
            pass
        else:
            # con_indicator_predict[index] = max_indicator
            break_constraint_index.append(index)
            continue

        if 3 in damper_position_list or 10 in damper_position_list or 17 in damper_position_list:
            pass
        else:
            # con_indicator_predict[index] = max_indicator
            break_constraint_index.append(index)
            continue

        if 4 in damper_position_list or 11 in damper_position_list or 18 in damper_position_list:
            pass
        else:
            # con_indicator_predict[index] = max_indicator
            break_constraint_index.append(index)
            continue

        if 5 in damper_position_list or 12 in damper_position_list or 19 in damper_position_list:
            pass
        else:
            # con_indicator_predict[index] = max_indicator
            break_constraint_index.append(index)
            continue

        if 6 in damper_position_list or 13 in damper_position_list or 20 in damper_position_list:
            pass
        else:
            # con_indicator_predict[index] = max_indicator
            break_constraint_index.append(index)
            continue

    # The beam section number shall not exceed the column section number plus 6.
    if ini_option:
        pass
    else:
        beam_profiles = ini_pop_data[:, 2]
        column_profiles = ini_pop_data[:, 5]

        minus_value = np.subtract(beam_profiles, column_profiles)
        error_index_array = np.argwhere(minus_value > 6)
        error_index_list = error_index_array.flatten().tolist()

        break_constraint_index = break_constraint_index + error_index_list

        # for i in range(error_index_array.shape[0]):
        #     error_index = error_index_array[i][0]
        #     con_indicator_predict[error_index] = max_indicator

    # Check the constraint of maximum displacement between layers.
    DIR_con_index_array = np.argwhere(avg_DIR_no_reshape > 0.02)
    DIR_con_index_list = DIR_con_index_array.flatten().tolist()

    break_constraint_index = break_constraint_index + DIR_con_index_list

    break_constraint_list = list(set(break_constraint_index))

    # return con_indicator_predict, break_constraint_index, avg_DIR
    return con_indicator_predict, break_constraint_list


def evaluate_objective(ini_pop_data, pop_normal_data, net_para_reg, reg_pth_path, net_para_gen, gen_pth_path, pop, ini_option):
    # f_1 predict the average DIR of steel frames; f_2 predict the average acce of steel frames
    f_1_2, break_constraint_list = cal_DIR_acce(ini_pop_data, pop_normal_data, net_para_reg, reg_pth_path, net_para_gen, gen_pth_path, pop, ini_option)

    # f_3 predict the cost of steel frames
    f_3 = cal_cost(ini_pop_data)

    f_cal = np.concatenate((f_1_2, f_3), axis=1)

    # f_max = np.max(f_cal, axis=0)
    f_max = np.array([0.3, 10000, 4000000])

    for i in range(len(break_constraint_list)):
        f_cal[break_constraint_list[i]] = f_max

    return f_cal