import numpy as np

from utils.train_tools import pre_train, test_pre


def main():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # networks_para_filepath = "./source_data/input_para/net_paras/net_para_sample_8th03.csv"
    # networks_para_data = np.loadtxt(networks_para_filepath, delimiter=",", dtype=int)
    networks_para_reg = [4,256,128,64,32,16]

    lr = 0.00035
    # lr_list = [10**-5, 5*10**-5, 10**-4]
    # lr_list = [10 ** -4, 2 * 10 ** -4, 3 * 10 ** -4, 4 * 10 ** -4, 5 * 10 ** -4]
    # lr_list = [1.5*10 ** -4, 2.5 * 10 ** -4]

    drop_out = 0.15
    # drop_out_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]

    pretrain_model_path = "./result/networks/pre-train_reg/pre-train_reg.pth"

    # for lr in lr_list:
    #     print("开始测试学习率{}".format(str(lr)))
    #     pre_train(networks_para_reg, pretrain_model_path, drop_out, lr)

    # for count in range(len(networks_para_data)):
    #     print("开始测试第{}个网络架构, 总共有{}个".format(count + 1, len(networks_para_data)))
    #
    #     networks_para_reg = networks_para_data[count]
    #
    #     pre_train(networks_para_reg, pretrain_model_path, drop_out, lr)

    # for count in range(len(networks_para_data)):
    #     for drop_out in drop_out_list:
    #         print("开始测试第{}个网络架构, 总共有{}个".format(count + 1, len(networks_para_data)))
    #
    #         networks_para_reg = networks_para_data[count]
    #
    #         pre_train(networks_para_reg, pretrain_model_path, drop_out, lr)

    # pre_train(networks_para_reg, pretrain_model_path, drop_out, lr)
    # test_pre(networks_para_reg, pretrain_model_path, drop_out)


if __name__ == '__main__':
    main()
