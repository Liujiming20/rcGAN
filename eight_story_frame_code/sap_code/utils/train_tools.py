import csv

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
from torcheval.metrics.functional import mean_squared_error, r2_score

from eight_story_frame_code.sap_code.utils.discriminator import Discriminator
from eight_story_frame_code.sap_code.utils.generator import Generator
from utils.generator import Generator04
from utils.regression_model import Regression03
from utils.discriminator import Discriminator04

from eight_story_frame_code.sap_code.utils.data_process import ValueProcessor
from utils.early_stopping import EarlyStopping
from utils.mixed_up import mixup_fun
from eight_story_frame_code.sap_code.utils.my_dataset import DatasetFromCSV


def get_val_set(dataset_val):
    length = len(dataset_val)

    # 通过__getitem__方法一次性获取所有数据
    all_data = [dataset_val[i] for i in range(length)]

    # 分离出values和labels
    values, labels = zip(*all_data)

    return torch.stack(values),  torch.stack(labels)


def pre_train(net_para, pretrain_model_path, lr, drop_para):
    torch.manual_seed(111)
    epoch_total = 1000
    value_dim = 8  # 8个层间位移
    label_dim = 32  # 16组截面的面积与惯性矩
    atten_para = 1  # 0-3
    drop = True
    drop_para = drop_para
    lr = lr

    train_dataset = DatasetFromCSV("./source_data/eight_story_frame/source_data/uni_source_data/input_train_uni.csv")

    train_size = int(0.89 * train_dataset.len)
    val_size = train_dataset.len - train_size
    train_set, val_set = data.random_split(train_dataset, [train_size, val_size])

    dataloader_train = data.DataLoader(train_set, batch_size=50, shuffle=True)
    dataloader_val = data.DataLoader(val_set, batch_size=50, shuffle=True)

    # 损失函数
    criterion1 = nn.SmoothL1Loss()

    # GPU使用
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        criterion1.to(device)

    layer_num = net_para[0]
    regression = Regression03(value_dim, label_dim, atten_para, drop, drop_para, layer_num, net_para[1], net_para[2], net_para[3], net_para[4], net_para[5])

    # print(regression)

    if cuda:
        regression.to(device)

    optimizer_R = torch.optim.Adam(regression.parameters(), lr=lr, betas=(0.5, 0.999))

    early_stopping = EarlyStopping(300, True)
    epochs = []
    disc_loss_epochs = []
    disc_loss_epochs_val = []
    for epoch in range(epoch_total):
        # if epoch % 100 == 0:
        #     print("正在训练第{}个epoch，总共有{}个epoch".format(epoch, epoch_total))

        regression.train()
        d_loss_total = 0

        count_train = 0
        for i, (values, labels) in enumerate(dataloader_train):
            count_train += 1
            values = values.to(device)
            labels = labels.to(device)

            labels,values = mixup_fun(labels,values)

            optimizer_R.zero_grad()

            pre_labels = regression(labels)

            dis_loss = criterion1(pre_labels, values)

            dis_loss.backward()
            optimizer_R.step()

            d_loss_total += dis_loss.item()

        epochs.append(epoch)  # 收集输出损失函数结果的epoch节点
        disc_loss_epochs.append(d_loss_total / count_train)

        regression.eval()
        d_loss_total_val = 0

        # 记录count
        count_val = 0
        with torch.no_grad():
            for i, (values_val, labels_val) in enumerate(dataloader_val):
                count_val += 1

                values_val = values_val.to(device)
                labels_val = labels_val.to(device)

                pre_labels_val = regression(labels_val)
                dis_loss_val = criterion1(pre_labels_val, values_val)
                d_loss_total_val += dis_loss_val.item()

            disc_loss_epochs_val.append(d_loss_total_val / count_val)
            early_stopping(d_loss_total_val, regression, pretrain_model_path)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

    # 计算验证集上R2
    regression.load_state_dict(torch.load(pretrain_model_path))
    regression.eval()

    with torch.no_grad():
        values_val_all, labels_val_all = get_val_set(val_set)
        values_val_all = values_val_all.to(device)
        labels_val_all = labels_val_all.to(device)

        prediction_val = regression(labels_val_all)

        R2_val = r2_score(prediction_val, values_val_all, multioutput="raw_values").clone().detach().to("cpu").numpy()

        with open("./result/eight_story_frame/R2_net_para/r2.csv", "a+", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(R2_val)

    fig, axs = plt.subplots(1, 2)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.title(str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(net_para[2]) + "_" + str(net_para[3]) + "_" + str(
        net_para[4]) + "_" + str(net_para[5]))
    axs[0].plot(epochs, disc_loss_epochs, label="disc_loss_epoaches", linestyle="-", color="k")
    axs[1].plot(epochs, disc_loss_epochs_val, label="disc_loss_epoaches_val", linestyle="-", color="r")
    fig_filepath = "./result/eight_story_frame/pre-train_val/" + str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(net_para[2]) + "_" + str(net_para[3]) + "_" + str(net_para[4]) + "_" + str(net_para[5]) + "_" + str(int(lr*10**5)) + "_" + str(int(drop_para * 100)) +".png"
    plt.savefig(fig_filepath)
    # plt.show()
    plt.close()


def test_pre(net_para, reg_pth_path, drop_para):
    torch.manual_seed(111)

    value_dim = 8
    label_dim = 32
    atten_para = 1  # 0-3
    drop = True
    drop_para = drop_para
    # drop_para = False

    test_dataset = DatasetFromCSV("./source_data/eight_story_frame/source_data/uni_source_data/input_test_uni.csv")
    x = np.linspace(0, 55, 55)

    test_dataloader = data.DataLoader(test_dataset, batch_size=55, shuffle=False)

    pretraining_weight_path = reg_pth_path
    regression = Regression03(value_dim, label_dim, atten_para, drop, drop_para, net_para[0], net_para[1], net_para[2], net_para[3], net_para[4], net_para[5])
    regression.load_state_dict(torch.load(pretraining_weight_path))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        regression.to(device)

    # 创建sap案例的归一化处理器
    value_processor = ValueProcessor()

    regression.eval()
    with torch.no_grad():
        for i, (y_test, x_input) in enumerate(test_dataloader):
            y_test = y_test.to(device)
            x_input = x_input.to(device)
            y_predict_norm = regression(x_input)

            y_test = y_test.clone().detach().to("cpu").numpy()
            y_predict_norm = y_predict_norm.clone().detach().to("cpu").numpy()
            y_test_ini = value_processor.back_process(y_test)
            y_predict_ini = value_processor.back_process(y_predict_norm)

            y_test_ini_torch = torch.from_numpy(y_test_ini[:, :])
            y_predict_ini_torch = torch.from_numpy(y_predict_ini[:, :])

            R2 = r2_score(y_predict_ini_torch, y_test_ini_torch, multioutput="raw_values").clone().detach().to("cpu").numpy()
            with open("./result/eight_story_frame/R2_net_para/r2.csv", "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(R2)

    y_test = value_processor.back_process(y_test)
    y_predict = value_processor.back_process(y_predict_norm)

    fig, axs = plt.subplots(8, 1)
    fig.set_size_inches(10,25)
    plt.title(str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(net_para[2]) + "_" + str(net_para[3]) + "_" + str(net_para[4]) + "_" + str(net_para[5]))
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    for fig_num in range(8):
        axs[fig_num].ticklabel_format(style='sci', scilimits=(-1, 2), axis="y", useMathText=True)
        axs[fig_num].plot(x, y_test[:, fig_num], label="IDR_FEM", linestyle="-", color="pink")
        axs[fig_num].plot(x, y_predict[:, fig_num], label="IDR_predict", linestyle="--", color="g")

    fig_filepath = "./result/eight_story_frame/test_dataset_val/" + str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(net_para[2]) + "_" + str(net_para[3]) + "_" + str(net_para[4]) + "_" + str(net_para[5]) + "_" + str(int(drop_para * 100))  + "_pre-reg.png"
    plt.savefig(fig_filepath)
    # plt.show()
    plt.close()


def freeze_model(model, to_freeze_dict, keep_step=None):
    for (name, para) in model.named_parameters():
        if name in to_freeze_dict:
            para.requires_grad = False
        else:
            pass

    return model


def load_freeze_regression(net_para_reg, reg_pre_path, device):
    regression = Regression03(8, 32, 1, True, 0.15, net_para_reg[0], net_para_reg[1], net_para_reg[2], net_para_reg[3], net_para_reg[4], net_para_reg[5])

    pre_state_dict = torch.load(reg_pre_path)
    regression.load_state_dict(pre_state_dict, strict=False)

    regression.to(device)

    regression = freeze_model(regression, pre_state_dict)

    return regression


def train(net_para, reg_net_para, lambda_para, gen_pth_path, reg_pre_path, k_count, drop_index, lr):
    torch.manual_seed(111)

    epoch_total = 1000
    value_dim = 8
    label_dim = 8
    atten_para = 1  # 0-3
    drop = True
    drop_para = drop_index
    # lr = 0.00001
    lr = lr

    train_dataset = DatasetFromCSV("./source_data/eight_story_frame/source_data/uni_source_data/input_train_uni.csv")

    train_size = int(0.89 * train_dataset.len)
    val_size = train_dataset.len - train_size
    train_set, val_set = data.random_split(train_dataset, [train_size, val_size])

    dataloader_train = data.DataLoader(train_set, batch_size=64, shuffle=True)
    dataloader_val = data.DataLoader(val_set, batch_size=64, shuffle=True)

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
    # generator = Generator04(value_dim, label_dim, atten_para, drop, drop_para, layer_num, net_para[1], net_para[2], net_para[3], net_para[4], net_para[5])
    # discriminator = Discriminator04(value_dim, label_dim, atten_para, drop_para, 4, net_para[4], net_para[3], net_para[2], net_para[1], net_para[5], 6)
    # if layer_num == 3:
    #     discriminator = Discriminator04(value_dim, label_dim, atten_para, drop_para, 3, net_para[3], net_para[2], net_para[1], net_para[4], net_para[5], 6)
    # elif layer_num == 5:
    #     discriminator = Discriminator04(value_dim, label_dim, atten_para, drop_para, 5, net_para[5], net_para[4], net_para[3], net_para[2], net_para[1], 6)
    generator = Generator(value_dim, label_dim, atten_para, drop, drop_para, layer_num, net_para[1], net_para[2], net_para[3], net_para[4], net_para[5])
    discriminator = Discriminator(value_dim, label_dim, atten_para, drop_para, 4, net_para[4], net_para[3], net_para[2], net_para[1], net_para[5], 8)
    if layer_num == 2:
        discriminator = Discriminator(value_dim, label_dim, atten_para, drop_para, 2, net_para[2], net_para[1], net_para[3], net_para[4], net_para[5], 8)
    elif layer_num == 3:
        discriminator = Discriminator(value_dim, label_dim, atten_para, drop_para, 3, net_para[3], net_para[2], net_para[1], net_para[4], net_para[5], 8)
    elif layer_num == 5:
        discriminator = Discriminator(value_dim, label_dim, atten_para, drop_para, 5, net_para[5], net_para[4], net_para[3], net_para[2], net_para[1], 8)

    # print(generator)
    # print(discriminator)

    if cuda:
        generator.to(device)
        discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=(lr/10))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=(lr/10))

    early_stopping = EarlyStopping(300, True)
    epochs = []
    disc_loss_epochs = []
    disc_loss_epochs_val = []

    gen_L1_loss_epochs = []
    gen_BCE_loss_epochs = []
    gen_L1_loss_epochs_val = []
    gen_BCE_loss_epochs_val = []

    for epoch in range(epoch_total):
        # if epoch % 100 == 0:
        #     print("正在训练第{}个epoch，总共有{}个epoch".format(epoch, epoch_total))

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
                z_noise = torch.normal(0, 1, (values.shape[0], values.shape[1], values.shape[2]), dtype=torch.float32) / 3
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

            if epoch < 200 or (epoch > 200 and gen_BCE_loss_epochs[epoch-1] < 1.0):
                for j in range(k_count):
                    inputs02, values02 = mixup_fun(inputs, values)
                    z_noise2 = torch.ones_like(values)
                    if values.dim() == 2:
                        z_noise2 = torch.normal(0, 1, (values.shape[0], values.shape[1]), dtype=torch.float32) / 3
                    elif values.dim() == 3:
                        z_noise2 = torch.normal(0, 1, (values.shape[0], values.shape[1], values.shape[2]), dtype=torch.float32) / 3
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

            if epoch > 200 and gen_BCE_loss_epochs[epoch-1] > 1.0:
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

    # 计算验证集上R2
    regression.eval()
    generator.load_state_dict(torch.load(gen_pth_path))
    generator.eval()

    with torch.no_grad():
        values_val_all, inputs_val_all = get_val_set(val_set)
        values_val_all = values_val_all.to(device)
        inputs_val_all = inputs_val_all.to(device)

        z_noise_val_all = torch.ones_like(values_val_all)
        if values_val_all.dim() == 2:
            z_noise_val_all = torch.normal(0, 1, (values_val_all.shape[0], values_val_all.shape[1]),dtype=torch.float32) / 3
        elif values_val_all.dim() == 3:
            z_noise_val_all = torch.normal(0, 1, (
            values_val_all.shape[0], values_val_all.shape[1], values_val_all.shape[2]), dtype=torch.float32) / 3
        else:
            print("=" * 10 + " ! " * 3 + "=" * 10)
            print("values的输入特征异常！")
        z_noise_val_all = z_noise_val_all.to(device)

        labels_val_all = regression(inputs_val_all)
        prediction_val = generator(z_noise_val_all, labels_val_all)

        R2_val = r2_score(prediction_val, values_val_all, multioutput="raw_values").clone().detach().to("cpu").numpy()

        with open("./result/eight_story_frame/R2_net_para/r2.csv", "a+", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(R2_val)

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.title(str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(net_para[2]) + "_" + str(net_para[3]) + "_" + str(net_para[4]) + "_" + str(net_para[5])+ "_" + str(int(drop_para * 100)))
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
    fig_filepath = "./result/eight_story_frame/loss_func_val/" + str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(net_para[2]) + "_" + str(net_para[3]) + "_" + str(net_para[4]) + "_" + str(net_para[5]) + "_" + str(k_count) + "_" + str(int(drop_para * 100)) +"_" + str(int(lr*100000)) + "_" + str(int(lambda_para)) + ".png"
    plt.savefig(fig_filepath)
    # plt.show()
    plt.close()

    # with open("./result/eight_story_frame/R2_net_para/disc_loss_epochs.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(disc_loss_epochs)
    #
    # with open("./result/eight_story_frame/R2_net_para/gen_BCE_loss_epochs.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_BCE_loss_epochs)
    #
    # with open("./result/eight_story_frame/R2_net_para/gen_L1_loss_epochs.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_L1_loss_epochs)
    #
    # with open("./result/eight_story_frame/R2_net_para/disc_loss_epochs_val.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(disc_loss_epochs_val)
    #
    # with open("./result/eight_story_frame/R2_net_para/gen_BCE_loss_epochs_val.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_BCE_loss_epochs_val)
    #
    # with open("./result/eight_story_frame/R2_net_para/gen_L1_loss_epochs_val.csv", "a+", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(gen_L1_loss_epochs_val)


def test(net_para, net_para_reg, gen_pth_path, reg_pth_path, drop_para, lr, lambda_para):
    torch.manual_seed(111)

    value_dim = 8
    label_dim = 8
    atten_para = 1  # 0-3
    # atten_para = 0
    drop = True
    drop_para = drop_para

    test_dataset = DatasetFromCSV("./source_data/eight_story_frame/source_data/uni_source_data/input_test_uni.csv")
    x = np.linspace(0, 55, 55)

    test_dataloader = data.DataLoader(test_dataset, batch_size=55, shuffle=False)

    regression = Regression03(8, 32, 1, True, 0.15, net_para_reg[0], net_para_reg[1], net_para_reg[2], net_para_reg[3], net_para_reg[4], net_para_reg[5])
    regression.load_state_dict(torch.load(reg_pth_path))
    # generator = Generator04(value_dim, label_dim, atten_para, drop, drop_para, net_para[0], net_para[1], net_para[2], net_para[3], net_para[4], net_para[5])
    generator = Generator(value_dim, label_dim, atten_para, drop, drop_para, net_para[0], net_para[1], net_para[2], net_para[3], net_para[4], net_para[5])
    generator.load_state_dict(torch.load(gen_pth_path))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        regression.to(device)
        generator.to(device)

    # 创建sap案例的归一化处理器
    value_processor = ValueProcessor()

    regression.eval()
    generator.eval()
    with torch.no_grad():
        for i, (y_test, x_input) in enumerate(test_dataloader):
            y_test = y_test.to(device)
            x_input = x_input.to(device)

            # 采用第二种测试方法时，下面代码要关闭
            z_noise_test = torch.ones_like(y_test)
            if y_test.dim() == 2:
                z_noise_test = torch.normal(0, 1, (y_test.shape[0], y_test.shape[1]), dtype=torch.float32) / 3
            elif y_test.dim() == 3:
                z_noise_test = torch.normal(0, 1, (y_test.shape[0], y_test.shape[1], y_test.shape[2]), dtype=torch.float32) / 3
            else:
                print("=" * 10 + " ! " * 3 + "=" * 10)
                print("values的输入特征异常！")

            z_noise_test = z_noise_test.to(device)

            label_test = regression(x_input)
            y_predict_norm = generator(z_noise_test, label_test)

            y_test = y_test.clone().detach().to("cpu").numpy()
            y_predict_norm = y_predict_norm.clone().detach().to("cpu").numpy()
            y_test_ini = value_processor.back_process(y_test)
            y_predict_ini = value_processor.back_process(y_predict_norm)

            # with open("./result/eight_story_frame/R2_net_para/y_test.csv", "a+", newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerows(y_test_ini)
            #
            # with open("./result/eight_story_frame/R2_net_para/y_predict.csv", "a+", newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerows(y_predict_ini)

            y_test_ini_torch = torch.from_numpy(y_test_ini[:, :])
            y_predict_ini_torch = torch.from_numpy(y_predict_ini[:, :])

            R2 = r2_score(y_predict_ini_torch, y_test_ini_torch, multioutput="raw_values").clone().detach().to("cpu").numpy()
            with open("./result/eight_story_frame/R2_net_para/r2.csv", "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(R2)

    y_test = value_processor.back_process(y_test)
    y_predict = value_processor.back_process(y_predict_norm)

    fig, axs = plt.subplots(8, 1)
    fig.set_size_inches(10,25)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    for fig_num in range(8):
        axs[fig_num].ticklabel_format(style='sci', scilimits=(-1, 2), axis="y", useMathText=True)
        axs[fig_num].plot(x, y_test[:, fig_num], label="IDR_FEM", linestyle="-", color="pink")
        axs[fig_num].plot(x, y_predict[:, fig_num], label="IDR_predict", linestyle="--", color="g")

    fig_filepath = "./result/eight_story_frame/test_dataset_val/" + str(net_para[0]) + "_" + str(net_para[1]) + "_" + str(net_para[2]) + "_" + str(net_para[3]) + "_" + str(net_para[4]) + "_" + str(net_para[5]) + "_" + str(int(drop_para * 100))  +"_" + str(int(lr*100000)) + "_" + str(int(lambda_para)) + ".png"
    plt.savefig(fig_filepath)
    # plt.show()
    plt.close()