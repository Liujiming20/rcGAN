import numpy as np
import torch

from utils.train_tools import pre_train, test_pre, train, test


def main():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # networks_para_filepath = "./source_data/input_para/net_paras/net_para_sample_8th.csv"
    # networks_para_data = np.loadtxt(networks_para_filepath, delimiter=",", dtype=int)
    para_list = [4, 256,128,64,32,16]
    networks_para_reg = [4,256,128,64,32,16]

    lr = 0.00004
    # lr_list = [0.00001, 0.00002, 0.00005, 0.0001, 0.0005]
    # lr_list = [0.00003, 0.000045, 0.00005, 0.00006]

    drop_out = 0.2
    # drop_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.5]

    lambda_para = 5
    # lambda_list = [1, 7.5, 10, 15, 30, 50]

    k_count = 3  # 实际对应每训练一次generator就训练了4次discriminator
    # k_count_list = [1, 2, 3, 4]

    pretrain_model_path = "./result/networks/pre-train_reg/pre-train_reg.pth"
    gen_pth_path = "./result/networks/gen/generator_lf_smoothL1_atten2.pth"

    # test_pre(networks_para_reg, pretrain_model_path, 0.15)

    # for lr in lr_list:
    #     print("开始测试学习率{}".format(str(lr)))
    #     train(para_list, networks_para_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_out, lr)

    # for count in range(len(networks_para_data)):
    #     # count = 6
    #     for lambda_para in lambda_list:
    #         lr = 0.000025
    #         if count == 2:
    #             lr = 0.00001
    #
    #         print("开始测试第{}个网络架构, 总共{}个网络".format(count + 1, len(networks_para_data)))
    #         para_list = networks_para_data[count]
    #         train(para_list, networks_para_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_out, lr)
    #
    #     # break

    # for lambda_para in lambda_list:
    #     print("测试lambda为{}".format(lambda_para))
    #     train(para_list, networks_para_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_out, lr)

    # for drop_out in drop_list:
    #     print("测试drop为{}".format(drop_para))
    #     train(para_list, networks_para_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_out, lr)

    # train(para_list, networks_para_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_out, lr)
    # test(para_list, networks_para_reg, gen_pth_path, pretrain_model_path, drop_out, lr, lambda_para)

    # 测试噪声
    seed_list_path = "./source_data/input_para/seed_list.csv"
    seed_list = np.loadtxt(seed_list_path, delimiter=",", dtype=int)

    for seed in seed_list:
        torch.manual_seed(seed)
        z_noise_test = torch.normal(0, 1, (55, 3), dtype=torch.float32) / 3

        test(para_list, networks_para_reg, gen_pth_path, pretrain_model_path, drop_out, lr, lambda_para, z_noise_test, seed)


if __name__ == '__main__':
    main()