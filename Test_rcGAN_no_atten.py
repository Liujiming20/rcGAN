import numpy as np
import torch

from utils.train_tools import pre_train, test_pre, train, test


def main():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    para_list = [4, 256, 128, 64, 32, 16]
    networks_para_reg = [4, 256, 128, 64, 32, 16]

    lambda_para = 5
    k_count = 3
    drop_out = 0.2
    lr = 0.00004
    pretrain_model_path = "./result/networks/pre-train_reg/pre-train_reg_atten0.pth"
    gen_pth_path = "./result/networks/gen/generator_lf_smoothL1_atten0.pth"

    # pre_train(networks_para_reg, pretrain_model_path, 0.15, lr=0.00035)
    # test_pre(networks_para_reg, pretrain_model_path, 0.15)

    # train(para_list, networks_para_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_out, lr)
    #
    # test(para_list, networks_para_reg, gen_pth_path, pretrain_model_path,drop_out, lr, lambda_para)

    seed_list_path = "./source_data/input_para/seed_list.csv"
    seed_list = np.loadtxt(seed_list_path, delimiter=",", dtype=int)

    for seed in seed_list:
        torch.manual_seed(seed)
        z_noise_test = torch.normal(0, 1, (55, 3), dtype=torch.float32) / 3

        test(para_list, networks_para_reg, gen_pth_path, pretrain_model_path, drop_out, lr, lambda_para, z_noise_test, seed)


if __name__ == '__main__':
    main()
