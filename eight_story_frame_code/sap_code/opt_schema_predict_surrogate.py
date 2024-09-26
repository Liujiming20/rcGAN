import csv

import numpy as np

import torch
from torch.utils import data

from eight_story_frame_code.sap_code.utils.data_process import ValueProcessor
from eight_story_frame_code.sap_code.utils.generator import Generator
from eight_story_frame_code.sap_code.utils.my_dataset import DatasetFromPop
from utils.regression_model import Regression03
from sap_case_optimization import converse_pop_to_model_input


def main():
    net_para_reg = [3, 64, 128, 256, 32, 16]
    reg_pth_path = "./result/eight_story_frame/networks/pre-train_reg/pre-train_reg.pth"
    net_gen_para = [2, 128, 512, 64, 32, 16]
    gen_pth_path = "./result/eight_story_frame/networks/gen/generator_lf_smoothL1_atten1.pth"

    # 读入截面
    design_variables = np.genfromtxt("./result/eight_story_frame/optimization_sap/opt_result.csv", delimiter=",", skip_header=1, dtype=int)[:, 2:]

    # 根据原始种群(截面型号)获取代理模型输入以及截面积
    input_array, area_array = converse_pop_to_model_input(design_variables)

    pop_input_torch = DatasetFromPop(input_array)

    test_dataloader = data.DataLoader(pop_input_torch, batch_size=pop_input_torch.len, shuffle=False)

    regressor = Regression03(8, 32, 1, True, 0.15, net_para_reg[0], net_para_reg[1], net_para_reg[2], net_para_reg[3],
                             net_para_reg[4], net_para_reg[5])
    regressor.load_state_dict(torch.load(reg_pth_path))
    generator = Generator(8, 8, 1, True, 0.025, net_gen_para[0], net_gen_para[1], net_gen_para[2], net_gen_para[3],
                          net_gen_para[4], net_gen_para[5])
    generator.load_state_dict(torch.load(gen_pth_path))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        regressor.to(device)
        generator.to(device)

    # 创建sap案例的归一化处理器
    value_processor = ValueProcessor()

    with open("./result/eight_story_frame/optimization_sap/opt_schema_predict.csv", "w+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["S1_IDR", "S2_IDR", "S3_IDR", "S4_IDR", "S5_IDR", "S6_IDR", "S7_IDR", "S8_IDR"])

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
            y_predict_ini = value_processor.back_process(y_predict_norm)

            with open("./result/eight_story_frame/optimization_sap/opt_schema_predict.csv", "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(y_predict_ini)


if __name__ == '__main__':
    main()