import numpy as np

from eight_story_frame_code.sap_code.utils.train_tools import train, test, test_pre


def main():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # networks_para_filepath = "./source_data/eight_story_frame/source_data/net_paras/net_para_sample_9th02.csv"
    # networks_para_data = np.loadtxt(networks_para_filepath, delimiter=",", dtype=int)
    para_list = [2,128,512,64,32,16]
    networks_prar_reg = [3,64,128,256,32,16]

    drop_out = 0.025
    # drop_list = [0.01]
    # drop_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
    # drop_list = [0.01, 0.025, 0.05]

    lambda_para = 5
    # lambda_list = [5]
    # lambda_list = [0.1, 1, 5, 10, 30, 50, 100]
    k_count = 1  # 实际对应每训练一次generator就训练了2次discriminator
    # k_count_list = [1, 2, 3, 4]
    pretrain_model_path = "./result/eight_story_frame/networks/pre-train_reg/pre-train_reg.pth"
    gen_pth_path = "./result/eight_story_frame/networks/gen/generator_lf_smoothL1_atten1.pth"

    lr = 0.0002
    # lr_list = [0.00001, 0.000025, 0.00005, 0.0001, 0.0002]
    # lr_list = [0.000025, 0.00005, 0.000075, 0.0001]
    # test_pre(networks_prar_reg, pretrain_model_path, 0.15)

    # for count in range(2):
    #     # count = 6
    #     for lambda_para in lambda_list:
    #         print("开始测试第{}个网络架构, 总共{}个网络".format(count + 1, len(networks_para_data)))
    #         para_list = networks_para_data[count]
    #         train(para_list, networks_prar_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_out, lr)
    #
    #     # break

    # for lambda_para in [5, 30]:
    #     print("测试lambda为{}".format(lambda_para))
    #     train(para_list, networks_prar_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, 0.3, 0.000025)

    # for drop_para in drop_list:
    #     print("测试drop为{}".format(drop_para))
    #     train(para_list, networks_prar_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_para, lr)

    # train(para_list, networks_prar_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_out, lr)
    test(para_list, networks_prar_reg, gen_pth_path, pretrain_model_path, drop_out, lr, lambda_para)

    # for lr in lr_list:
    #     print("开始测试lr={}".format(str(lr)))
    #     train(para_list, networks_prar_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_out, lr)

    # # 测试噪声
    # seed_list_path = "./source_data/input_para/seed_list.csv"
    # seed_list = np.loadtxt(seed_list_path, delimiter=",", dtype=int)
    #
    # for seed in seed_list:
    #     torch.manual_seed(seed)
    #     z_noise_test = torch.normal(0, 1, (55, 3), dtype=torch.float32) / 3
    #
    #     test(para_list, networks_prar_reg, gen_pth_path, pretrain_model_path, 0.2, 0.000025, lambda_para, z_noise_test, seed)


if __name__ == '__main__':
    main()