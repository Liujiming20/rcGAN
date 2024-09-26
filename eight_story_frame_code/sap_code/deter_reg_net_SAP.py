import numpy as np

from eight_story_frame_code.sap_code.utils.train_tools import pre_train, test_pre


def main():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # networks_para_filepath = "./source_data/eight_story_frame/source_data/net_paras/net_para_sample_7th03.csv"
    # networks_para_data = np.loadtxt(networks_para_filepath, delimiter=",", dtype=int)
    networks_prar_reg = [3,64,128,256,32,16]

    drop_out = 0.15

    pretrain_model_path = "./result/eight_story_frame/networks/pre-train_reg/pre-train_reg.pth"

    lr = 0.00025
    # for lr in [0.0001, 0.0005, 0.001]:
    #     print("开始测试lr={}".format(str(lr)))
    #     pre_train(networks_prar_reg, pretrain_model_path, lr, drop_out)

    # for count in range(22, 26):
    #     print("开始测试第{}个网络架构, 总共有{}个".format(count + 1, len(networks_para_data)))
    #     networks_prar_reg = networks_para_data[count]
    #
    #     pre_train(networks_prar_reg, pretrain_model_path, lr, drop_out)

    # for count in range(len(networks_para_data)):
    #     for drop_out in [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    #         print("开始测试第{}个网络架构, 总共有{}个".format(count + 1, len(networks_para_data)))
    #         networks_prar_reg = networks_para_data[count]
    #
    #         pre_train(networks_prar_reg, pretrain_model_path, lr, drop_out)

    # drop_list = [0.15, 0.1, 0.1, 0.05]
    # for count in range(2, 4):
    #     print("开始测试第{}个网络架构, 总共有{}个".format(count + 1, len(networks_para_data)))
    #     networks_prar_reg = networks_para_data[count]
    #
    #     drop_out = drop_list[count]
    #
    #     for lr in [0.0005, 0.00075, 0.001, 0.0015, 0.003]:
    #         pre_train(networks_prar_reg, pretrain_model_path, lr, drop_out)
    #         test_pre(networks_prar_reg, pretrain_model_path, drop_out)

    # pre_train(networks_prar_reg, pretrain_model_path, lr, drop_out)
    test_pre(networks_prar_reg, pretrain_model_path, drop_out)


if __name__ == '__main__':
    main()