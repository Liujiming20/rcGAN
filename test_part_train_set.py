import csv

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn

from utils.discriminator import Discriminator04
from utils.early_stopping import EarlyStopping
from utils.generator import Generator04
from utils.mixed_up import mixup_fun
from utils.my_dataset import DatasetFromCSV
from utils.train_tools import load_freeze_regression, test


def train_part_train_data(net_para, reg_net_para, lambda_para, gen_pth_path, reg_pre_path, k_count, drop_index, lr, train_datalodar, val_datalodar, index):
    print(lr)

    torch.manual_seed(111)

    epoch_total = 1000
    value_dim = 3
    label_dim = 3
    atten_para = 2  # 0-3
    drop = True
    drop_para = drop_index
    lr = lr

    dataloader_train = train_datalodar
    dataloader_val = val_datalodar

    # 损失函数
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.SmoothL1Loss()

    # GPU使用
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        criterion1.to(device)
        criterion2.to(device)

    regression = load_freeze_regression(reg_net_para, reg_pre_path, device)

    # 第一维是层数，后面五维是节点数目
    net_para = net_para
    layer_num = net_para[0]
    generator = Generator04(value_dim, label_dim, atten_para, drop, drop_para, layer_num, net_para[1], net_para[2],
                            net_para[3], net_para[4], net_para[5])
    discriminator = Discriminator04(value_dim, label_dim, atten_para, drop_para, 4, net_para[4], net_para[3],
                                    net_para[2], net_para[1], net_para[5], 6)
    if layer_num == 3:
        discriminator = Discriminator04(value_dim, label_dim, atten_para, drop_para, 3, net_para[3], net_para[2],
                                        net_para[1], net_para[4], net_para[5], 6)
    elif layer_num == 5:
        discriminator = Discriminator04(value_dim, label_dim, atten_para, drop_para, 5, net_para[5], net_para[4],
                                        net_para[3], net_para[2], net_para[1], 6)

    # print(generator)
    # print(discriminator)

    if cuda:
        generator.to(device)
        discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=(lr / 10))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=(lr / 10))

    early_stopping = EarlyStopping(1000, True)
    epochs = []
    disc_loss_epochs = []
    disc_loss_epochs_val = []

    gen_L1_loss_epochs = []
    gen_BCE_loss_epochs = []
    gen_L1_loss_epochs_val = []
    gen_BCE_loss_epochs_val = []

    for epoch in range(epoch_total):
        # train GAN
        generator.train()
        discriminator.train()
        regression.eval()
        d_loss_total = 0
        g_L1_loss_total = 0
        g_BCE_loss_total = 0

        # 每一个batch损失是求了平均的【损失函数自己求均方根误差】，但一个epoch的多次batch又持续累加误差，应该除以count
        count_train = 0
        for i, (values, inputs) in enumerate(dataloader_train):
            count_train += 1
            values = values.to(device)
            inputs = inputs.to(device)

            inputs01, values01 = mixup_fun(inputs, values)

            z_noise = torch.ones_like(values)
            if values.dim() == 2:
                z_noise = torch.normal(0, 1, (values.shape[0], values.shape[1]), dtype=torch.float32) / 3
            elif values.dim() == 3:
                z_noise = torch.normal(0, 1, (values.shape[0], values.shape[1], values.shape[2]),
                                       dtype=torch.float32) / 3
            else:
                print("=" * 10 + " ! " * 3 + "=" * 10)
                print("values的输入特征异常！")
            z_noise = z_noise.to(device)

            optimizer_D.zero_grad()

            # 利用回归器创建labels
            labels = regression(inputs01)

            # 利用生成器产生预测值
            predict = generator(z_noise, labels)

            # 真实输出的交叉熵损失
            validity_real = discriminator(values01, labels)
            d_loss_real = criterion1(validity_real, torch.ones_like(validity_real))  # 真实样本的输出应被判别为1

            # 生成样本的交叉熵损失
            validity_fake = discriminator(predict.detach(), labels)
            d_loss_fake = criterion1(validity_fake, torch.zeros_like(validity_fake))  # 生成样本的输出应该被判别为0

            dis_loss = (d_loss_real + d_loss_fake) / 2

            dis_loss.backward()
            optimizer_D.step()

            if epoch < 200 or (epoch > 200 and gen_BCE_loss_epochs[epoch - 1] < 1.0):
                for j in range(k_count):
                    inputs02, values02 = mixup_fun(inputs, values)
                    z_noise2 = torch.ones_like(values)
                    if values.dim() == 2:
                        z_noise2 = torch.normal(0, 1, (values.shape[0], values.shape[1]), dtype=torch.float32) / 3
                    elif values.dim() == 3:
                        z_noise2 = torch.normal(0, 1, (values.shape[0], values.shape[1], values.shape[2]),
                                                dtype=torch.float32) / 3
                    else:
                        print("=" * 10 + " ! " * 3 + "=" * 10)
                        print("values的输入特征异常！")
                    z_noise2 = z_noise2.to(device)

                    optimizer_D.zero_grad()

                    labels2 = regression(inputs02)

                    predict2 = generator(z_noise2, labels2)
                    validity_real2 = discriminator(values02, labels2)
                    d_loss_real2 = criterion1(validity_real2, torch.ones_like(validity_real2))
                    validity_fake2 = discriminator(predict2.detach(), labels2)
                    d_loss_fake2 = criterion1(validity_fake2, torch.zeros_like(validity_fake2))
                    dis_loss2 = (d_loss_real2 + d_loss_fake2) / 2

                    dis_loss2.backward()
                    optimizer_D.step()

            d_loss_total += dis_loss.item()

            optimizer_G.zero_grad()
            validity_gen = discriminator(predict, labels)  # 生成器制造的预测值被判别器鉴定的结果
            g_loss1 = criterion1(validity_gen, torch.ones_like(validity_gen))  # 审视被鉴定为1的距离
            g_loss2 = criterion2(predict, values01) * lambda_para  # 审视预测值与真实值的一范数距离

            gen_loss = g_loss1 + g_loss2

            gen_loss.backward()
            optimizer_G.step()

            if epoch > 200 and gen_BCE_loss_epochs[epoch - 1] > 1.0:
                inputs02, values02 = mixup_fun(inputs, values)

                z_noise_gen = torch.ones_like(values)
                if values.dim() == 2:
                    z_noise_gen = torch.normal(0, 1, (values.shape[0], values.shape[1]), dtype=torch.float32) / 3
                elif values.dim() == 3:
                    z_noise_gen = torch.normal(0, 1, (values.shape[0], values.shape[1], values.shape[2]),
                                               dtype=torch.float32) / 3
                else:
                    print("=" * 10 + " ! " * 3 + "=" * 10)
                    print("values的输入特征异常！")
                z_noise_gen = z_noise_gen.to(device)

                optimizer_G.zero_grad()
                labels_gen = regression(inputs02)

                predict_gen = generator(z_noise_gen, labels_gen)

                g_loss3 = criterion2(predict_gen, values02)
                g_loss3.backward()
                optimizer_G.step()

            g_BCE_loss_total += g_loss1.item()
            g_L1_loss_total += g_loss2.item()

        epochs.append(epoch)  # 收集输出损失函数结果的epoch节点
        disc_loss_epochs.append(d_loss_total / count_train)
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

                z_noise_val = torch.ones_like(values_val)
                if values_val.dim() == 2:
                    z_noise_val = torch.normal(0, 1, (values_val.shape[0], values_val.shape[1]),
                                               dtype=torch.float32) / 3
                elif values_val.dim() == 3:
                    z_noise_val = torch.normal(0, 1, (values_val.shape[0], values_val.shape[1], values_val.shape[2]),
                                               dtype=torch.float32) / 3
                else:
                    print("=" * 10 + " ! " * 3 + "=" * 10)
                    print("values的输入特征异常！")
                z_noise_val = z_noise_val.to(device)

                labels_val = regression(inputs_val)
                predict_val = generator(z_noise_val, labels_val)

                validity_real_val = discriminator(values_val, labels_val)
                d_loss_real_val = criterion1(validity_real_val, torch.ones_like(validity_real_val))

                validity_fake_val = discriminator(predict_val, labels_val)
                d_loss_fake_val = criterion1(validity_fake_val, torch.zeros_like(validity_fake_val))

                dis_loss_val = (d_loss_real_val + d_loss_fake_val) / 2
                d_loss_total_val += dis_loss_val.item()

                #  val generator
                validity_gen_val = validity_fake_val
                g_loss1_val = criterion1(validity_gen_val, torch.ones_like(validity_gen_val))
                g_loss2_val = criterion2(predict_val, values_val)

                g_BCE_loss_total_val += g_loss1_val.item()
                g_L1_loss_total_val += g_loss2_val.item()

            disc_loss_epochs_val.append(d_loss_total_val / count_val)
            gen_L1_loss_epochs_val.append(g_L1_loss_total_val / count_val)
            gen_BCE_loss_epochs_val.append(g_BCE_loss_total_val / count_val)

            early_stopping(g_L1_loss_total_val, generator, gen_pth_path)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.title(str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(net_para[2]) + "_" + str(net_para[3]) + "_" + str(
        net_para[4]) + "_" + str(net_para[5]) + "_" + str(int(drop_para * 100)))
    axs[0, 0].plot(epochs, disc_loss_epochs, label="disc_loss_epoaches", linestyle="-", color="k")
    axs[0, 1].plot(epochs, disc_loss_epochs_val, label="disc_loss_epoaches_val", linestyle="-", color="r")
    axs[1, 0].plot(epochs, gen_L1_loss_epochs, label="gen_L1_loss_epoaches", linestyle="-", color="g")
    axs[0, 0].plot(epochs, gen_BCE_loss_epochs, label="gen_BCE_loss_epoaches", linestyle="-.", color="pink")
    axs[1, 1].plot(epochs, gen_L1_loss_epochs_val, label="gen_L1_loss_epoaches_val", linestyle="-", color="g")
    axs[0, 1].plot(epochs, gen_BCE_loss_epochs_val, label="gen_BCE_loss_epoaches_val", linestyle="-.", color="pink")
    axs[0, 0].legend(loc=0)
    axs[0, 1].legend(loc=0)
    axs[1, 0].legend(loc=0)
    axs[1, 1].legend(loc=0)
    fig_filepath = "./result/loss_func_val/part_train_data/" + str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(
        net_para[2]) + "_" + str(net_para[3]) + "_" + str(net_para[4]) + "_" + str(net_para[5]) + "_" + str(index) + ".png"
    plt.savefig(fig_filepath)
    # plt.show()
    plt.close()

    # # 记录损失函数
    # with open("./result/R2_net_para/part_train_data/{}_disc_loss_epochs.csv".format(str(index)), "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(disc_loss_epochs)
    #
    # with open("./result/R2_net_para/part_train_data/{}_gen_BCE_loss_epochs.csv".format(str(index)), "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_BCE_loss_epochs)
    #
    # with open("./result/R2_net_para/part_train_data/{}_gen_L1_loss_epochs.csv".format(str(index)), "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_L1_loss_epochs)
    #
    # with open("./result/R2_net_para/part_train_data/{}_disc_loss_epochs_val.csv".format(str(index)), "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(disc_loss_epochs_val)
    #
    # with open("./result/R2_net_para/part_train_data/{}_gen_BCE_loss_epochs_val.csv".format(str(index)), "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_BCE_loss_epochs_val)
    #
    # with open("./result/R2_net_para/part_train_data/{}_gen_L1_loss_epochs_val.csv".format(str(index)), "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_L1_loss_epochs_val)


def main():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    para_list = [4, 256, 128, 64, 32, 16]
    networks_para_reg = [4, 256, 128, 64, 32, 16]

    drop_para = 0.2

    lambda_para = 5
    k_count = 3

    lr = 0.00004

    pretrain_model_path = "./result/networks/pre-train_reg/pre-train_reg.pth"

    for i in range(1, 6):
        # i = 5
        torch.manual_seed(111)

        print("正在执行{}/5数据集的训练".format(str(i)))

        gen_pth_path = "./result/networks/gen/part_train_data/{}_generator_lf_smoothL1_atten2.pth".format(str(i))

        # train_dataset = DatasetFromCSV("./source_data/input_para/part_of_train_data/input_label{}.csv".format(str(i)),
        #                                "./source_data/input_para/part_of_train_data/input_value{}.csv".format(str(i)))
        #
        # train_size = int(0.89 * train_dataset.len)
        # val_size = train_dataset.len - train_size
        # train_set, val_set = data.random_split(train_dataset, [train_size, val_size])
        #
        # dataloader_train = data.DataLoader(train_set, batch_size=32, shuffle=True)
        # dataloader_val = data.DataLoader(val_set, batch_size=32, shuffle=True)
        #
        # lr = 0.00001 + 0.00003 * i/5
        #
        # train_part_train_data(para_list, networks_para_reg, lambda_para, gen_pth_path, pretrain_model_path, k_count, drop_para, lr, dataloader_train, dataloader_val, i)

        # 以下是噪声测试代码
        seed_list_path = "./source_data/input_para/seed_list.csv"
        seed_list = np.loadtxt(seed_list_path, delimiter=",", dtype=int)

        for seed in seed_list:
            torch.manual_seed(seed)
            z_noise_test = torch.normal(0, 1, (55, 3), dtype=torch.float32) / 3

            test(para_list, networks_para_reg, gen_pth_path, pretrain_model_path, drop_para, lr, lambda_para, z_noise_test, seed)

        # break


if __name__ == '__main__':
    main()