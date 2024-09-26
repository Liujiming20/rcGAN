import csv

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
from torcheval.metrics.functional import mean_squared_error, r2_score

from utils.generator import Generator06
from utils.discriminator import Discriminator06

from utils.data_process import LabelProcessor, ValueProcessor
from utils.early_stopping import EarlyStopping
from utils.my_dataset import DatasetFromCSV, DatasetFromSourceData
from utils.source_data import DataSource
from utils.train_tools import get_val_set


def train(net_para, gen_pth_path, k_count, lambda_para, drop_index, lr):
    torch.manual_seed(111)

    epoch_total = 1000
    value_dim = 3
    label_dim = 17
    atten_para = 2  # 0-3
    drop = True
    drop_para = drop_index
    lr = lr

    train_dataset = DatasetFromCSV("./source_data/input_para/input_label.csv",
                                   "./source_data/input_para/input_value.csv")

    train_size = int(0.89 * train_dataset.len)
    val_size = train_dataset.len - train_size
    train_set, val_set = data.random_split(train_dataset, [train_size, val_size])

    dataloader_train = data.DataLoader(train_set, batch_size=32, shuffle=True)
    dataloader_val = data.DataLoader(val_set, batch_size=32, shuffle=True)

    # 损失函数
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.SmoothL1Loss()

    # GPU使用
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        criterion1.to(device)
        criterion2.to(device)

    # 第一维是层数，后面五维是节点数目
    net_para = net_para
    layer_num = net_para[0]
    generator = Generator06(value_dim, label_dim, atten_para, drop, drop_para, layer_num, net_para[1], net_para[2], net_para[3], net_para[4], net_para[5])
    discriminator = Discriminator06(value_dim, label_dim, atten_para, drop_para, 4, net_para[4], net_para[3], net_para[2], net_para[1], net_para[5], 6)
    if layer_num == 3:
        discriminator = Discriminator06(value_dim, label_dim, atten_para, drop_para, 3, net_para[3], net_para[2], net_para[1], net_para[4], net_para[5], 6)
    elif layer_num == 5:
        discriminator = Discriminator06(value_dim, label_dim, atten_para, drop_para, 5, net_para[5], net_para[4], net_para[3], net_para[2], net_para[1], 6)

    # print(generator)
    # print(discriminator)

    if cuda:
        generator.to(device)
        discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=(lr/10))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=(lr/10))

    early_stopping = EarlyStopping(300, True)
    epochs = []
    counts = []
    disc_loss_epochs = []
    disc_loss_epochs_val = []

    gen_L1_loss_epochs = []
    gen_BCE_loss_epochs = []
    gen_L1_loss_epochs_val = []
    gen_BCE_loss_epochs_val = []

    for epoch in range(epoch_total):
        gen_train_option = False

        # train GAN
        generator.train()
        discriminator.train()
        # regression.eval()
        d_loss_total = 0
        # g_loss_total = 0
        g_L1_loss_total = 0
        g_BCE_loss_total = 0

        # 每一个batch损失是求了平均的【损失函数自己求均方根误差】，但一个epoch的多次batch又持续累加误差，应该除以count
        count_train = 0
        for i, (values, inputs) in enumerate(dataloader_train):
            count_train += 1
            values = values.to(device)
            inputs = inputs.to(device)

            optimizer_D.zero_grad()

            # 利用生成器产生预测值
            predict = generator(inputs)

            # 真实输出的交叉熵损失
            validity_real = discriminator(values)
            d_loss_real = criterion1(validity_real, torch.ones_like(validity_real))  # 真实样本的输出应被判别为1

            # 生成样本的交叉熵损失
            validity_fake = discriminator(predict.detach())
            d_loss_fake = criterion1(validity_fake, torch.zeros_like(validity_fake))  # 生成样本的输出应该被判别为0

            dis_loss = (d_loss_real + d_loss_fake) / 2

            dis_loss.backward()
            optimizer_D.step()

            d_loss_total += dis_loss.item()

            # train generator
            if epoch % k_count == 0:
                gen_train_option = True

                optimizer_G.zero_grad()

                # 利用生成器产生预测值
                predict_gen = generator(inputs)
                validity_gen = discriminator(predict_gen)  # 生成器制造的预测值被判别器鉴定的结果

                # gen_loss = -torch.mean(validity_fake)
                g_loss1 = criterion1(validity_gen, torch.ones_like(validity_gen))  # 审视被鉴定为1的距离
                g_loss2 = criterion2(predict_gen, values) * lambda_para  # 审视预测值与真实值的一范数距离

                gen_loss = g_loss1 + g_loss2

                gen_loss.backward()
                optimizer_G.step()

                g_BCE_loss_total += g_loss1.item()
                g_L1_loss_total += g_loss2.item()

        epochs.append(epoch)  # 收集输出损失函数结果的epoch节点
        disc_loss_epochs.append(d_loss_total / count_train)
        # gen_loss_epoaches.append(g_loss_total / count_train)
        if gen_train_option:
            counts.append(epoch)
            gen_L1_loss_epochs.append(g_L1_loss_total / count_train)
            gen_BCE_loss_epochs.append(g_BCE_loss_total / count_train)

        # val GAN
        generator.eval()
        discriminator.eval()
        d_loss_total_val = 0
        g_L1_loss_total_val = 0
        g_BCE_loss_total_val = 0
        # 记录count
        count_val = 0
        with torch.no_grad():
            for i, (values_val, inputs_val) in enumerate(dataloader_val):
                count_val += 1

                values_val = values_val.to(device)
                inputs_val = inputs_val.to(device)
                predict_val = generator(inputs_val)

                validity_real_val = discriminator(values_val)
                d_loss_real_val = criterion1(validity_real_val, torch.ones_like(validity_real_val))

                validity_fake_val = discriminator(predict_val)
                d_loss_fake_val = criterion1(validity_fake_val, torch.zeros_like(validity_fake_val))

                dis_loss_val = (d_loss_real_val + d_loss_fake_val) / 2
                d_loss_total_val += dis_loss_val.item()

                #  val generator
                validity_gen_val = validity_fake_val
                g_loss1_val = criterion1(validity_gen_val, torch.ones_like(validity_gen_val))
                g_loss2_val = criterion2(predict_val, values_val) * lambda_para

                g_BCE_loss_total_val += g_loss1_val.item()
                g_L1_loss_total_val += g_loss2_val.item()

            disc_loss_epochs_val.append(d_loss_total_val / count_val)
            gen_L1_loss_epochs_val.append(g_L1_loss_total_val / count_val)
            gen_BCE_loss_epochs_val.append(g_BCE_loss_total_val / count_val)

            early_stopping(g_L1_loss_total_val, generator, gen_pth_path)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

    # # 计算验证集上R2
    # generator.load_state_dict(torch.load(gen_pth_path))
    # generator.eval()
    #
    # with torch.no_grad():
    #     values_val_all, inputs_val_all = get_val_set(val_set)
    #     values_val_all = values_val_all.to(device)
    #     inputs_val_all = inputs_val_all.to(device)
    #
    #     prediction_val = generator(inputs_val_all)
    #
    #     R2_val = r2_score(prediction_val, values_val_all, multioutput="raw_values").clone().detach().to("cpu").numpy()
    #
    #     with open("./result/R2_net_para/GAN/r2.csv", "a+", newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(R2_val)

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.title(str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(net_para[2]) + "_" + str(net_para[3]) + "_" + str(
        net_para[4]) + "_" + str(net_para[5]))
    axs[0, 0].plot(epochs, disc_loss_epochs, label="disc_loss_epochs", linestyle="-", color="k")
    axs[0, 1].plot(epochs, disc_loss_epochs_val, label="disc_loss_epochs_val", linestyle="-", color="r")
    axs[1, 0].plot(counts, gen_L1_loss_epochs, label="gen_L1_loss_epochs", linestyle="-", color="g")
    axs[0, 0].plot(counts, gen_BCE_loss_epochs, label="gen_BCE_loss_epochs", linestyle="-.", color="pink")
    axs[1, 1].plot(epochs, gen_L1_loss_epochs_val, label="gen_L1_loss_epochs_val", linestyle="-", color="g")
    axs[0, 1].plot(epochs, gen_BCE_loss_epochs_val, label="gen_BCE_loss_epochs_val", linestyle="-.", color="pink")
    axs[0, 0].legend(loc=0)
    axs[0, 1].legend(loc=0)
    axs[1, 0].legend(loc=0)
    axs[1, 1].legend(loc=0)
    # fig_filepath = "./result/loss_func_val/GAN/" + str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(
    #     net_para[2]) + "_" + str(net_para[3]) + "_" + str(net_para[4]) + "_" + str(net_para[5]) +"_" + str(int(lr*100000)) + ".png"
    fig_filepath = "./result/loss_func_val/GAN/" + str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(
        net_para[2]) + "_" + str(net_para[3]) + "_" + str(net_para[4]) + "_" + str(net_para[5]) + "_" + str(
        int(lambda_para)) + "_" + str(int(drop_index * 100)) + ".png"

    plt.savefig(fig_filepath)
    # plt.show()
    plt.close()

    # with open("./result/R2_net_para/disc_loss_epochs.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(disc_loss_epochs)
    #
    # with open("./result/R2_net_para/gen_BCE_loss_epochs.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_BCE_loss_epochs)
    #
    # with open("./result/R2_net_para/gen_L1_loss_epochs.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_L1_loss_epochs)
    #
    # with open("./result/R2_net_para/disc_loss_epochs_val.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(disc_loss_epochs_val)
    #
    # with open("./result/R2_net_para/gen_BCE_loss_epochs_val.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_BCE_loss_epochs_val)
    #
    # with open("./result/R2_net_para/gen_L1_loss_epochs_val.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_L1_loss_epochs_val)


def test(net_para, gen_pth_path, drop_para):
    torch.manual_seed(111)

    value_dim = 3
    label_dim = 17
    atten_para = 2  # 0-3
    drop = True
    drop_para = drop_para

    source_data = DataSource("./source_data/input_para/input_para.csv")
    label_processor = LabelProcessor()
    value_processor = ValueProcessor(source_data.y)

    source_data_test = DataSource("./source_data/input_para/input_para_test.csv")

    test_dataset = DatasetFromSourceData(source_data_test, label_processor, value_processor)
    # x = np.linspace(0, 55, 55)

    test_dataloader = data.DataLoader(test_dataset, batch_size=55, shuffle=False)

    generator = Generator06(value_dim, label_dim, atten_para, drop, drop_para, net_para[0], net_para[1], net_para[2], net_para[3], net_para[4], net_para[5])
    generator.load_state_dict(torch.load(gen_pth_path))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        generator.to(device)

    generator.eval()
    with torch.no_grad():
        for i, (y_test, x_input) in enumerate(test_dataloader):
            y_test = y_test.to(device)
            x_input = x_input.to(device)

            y_predict_norm = generator(x_input)

            y_test = y_test.clone().detach().to("cpu").numpy()
            y_predict_norm = y_predict_norm.clone().detach().to("cpu").numpy()
            y_test_ini = value_processor.back_process(y_test)
            y_predict_ini = value_processor.back_process(y_predict_norm)

            # with open("./result/R2_net_para/GAN/y_test.csv", "a+", newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerows(y_test_ini)
            #
            # with open("./result/R2_net_para/GAN/y_predict.csv", "a+", newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerows(y_predict_ini)

            y_test_ini_torch = torch.from_numpy(y_test_ini[:, :])
            y_predict_ini_torch = torch.from_numpy(y_predict_ini[:, :])

            R2 = r2_score(y_predict_ini_torch, y_test_ini_torch, multioutput="raw_values").clone().detach().to("cpu").numpy()
            with open("./result/R2_net_para/GAN/r2.csv", "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(R2)

            MSE = mean_squared_error(y_predict_ini_torch, y_test_ini_torch, multioutput="raw_values").clone().detach().to("cpu").numpy()
            fre_mse = MSE[0]*55
            fre_y2 = np.inner(y_test_ini[:, 0], y_test_ini[:, 0])
            fre_nmse = fre_mse/fre_y2

            IDR_mse = MSE[1]*55
            IDR_y2 = np.inner(y_test_ini[:, 1], y_test_ini[:, 1])
            IDR_nmse = IDR_mse/IDR_y2

            acce_mse = MSE[2]*55
            acce_y2 = np.inner(y_test_ini[:, 2], y_test_ini[:, 2])
            acce_nmse = acce_mse/acce_y2

            nmse = [fre_nmse, IDR_nmse, acce_nmse]
            nrmse = [fre_nmse ** 0.5, IDR_nmse ** 0.5, acce_nmse ** 0.5]

            with open("./result/R2_net_para/GAN/NMSE.csv", "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(nmse)

            with open("./result/R2_net_para/GAN/NRMSE.csv", "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(nrmse)

            # cal NMAE
            abs_sub_value = np.abs(np.subtract(y_test_ini, y_predict_ini))
            max_value = np.max(abs_sub_value, axis=0)

            max_test_value = np.max(y_test_ini, axis=0)

            fre_nmae = max_value[0] / (max_test_value[0])

            IDR_nmae = max_value[1] / (max_test_value[1])

            acce_nmae = max_value[2] / (max_test_value[2])

            nmae = [fre_nmae, IDR_nmae, acce_nmae]

            with open("./result/R2_net_para/GAN/NMAE.csv", "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(nmae)


def main():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # networks_para_filepath = "./source_data/input_para/net_paras/net_para_sample_9thGAN.csv"
    # networks_para_data = np.loadtxt(networks_para_filepath, delimiter=",", dtype=int)
    net_para = [5,128,128,64,32,16]

    drop_index = 0.1
    # drop_index_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    k_count = 4  # 这里的3与rcGAN的2有着类似效果，这种方式采用了cGAN原始的训练方法

    lambda_para = 7.5
    # lambda_list = [2.5, 5, 7.5, 10, 20, 30]
    # lambda_list = [0.1, 1, 10, 30, 50, 100, 200]

    lr = 0.0001
    # lr_list = [0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.001]
    # lr_list = [0.00005, 0.0001, 0.0002, 0.0003, 0.0005]

    gen_pth_path = "./result/networks/gen/GAN/generator_lf_smoothL1_atten2_GAN.pth"

    # train(net_para, gen_pth_path, k_count, lambda_para, drop_index, lr)

    test(net_para, gen_pth_path, drop_index)

    # for count in range(len(networks_para_data)):
    #     for lambda_para in lambda_list:
    #         lr = 0.0005
    #
    #         print("开始测试第{}个网络架构, 总共{}个网络".format(count + 1, len(networks_para_data)))
    #
    #         net_para = networks_para_data[count]
    #         train(net_para, gen_pth_path, k_count, lambda_para, drop_index, lr)

    # for lr in lr_list:
    #     print("测试学习率为{}".format(str(lr)))
    #     train(net_para, gen_pth_path, k_count, lambda_para, drop_index, lr)

    # for drop_index in drop_index_list:
    #     train(net_para, gen_pth_path, k_count, lambda_para, drop_index, lr)


if __name__ == '__main__':
    main()
