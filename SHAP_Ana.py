import random

import torch
import torch.nn as nn
import shap
import numpy as np
import matplotlib.pyplot as plt

from utils.generator import Generator04
from utils.regression_model import Regression03


# 定义组合模型
class CombinedModel(nn.Module):
    def __init__(self, regressor, generator):
        super(CombinedModel, self).__init__()
        self.first_model = regressor
        self.second_model = generator

    def forward(self, x):
        x_label = self.first_model(x)

        z_noise = torch.normal(0, 1, (x_label.shape[0], x_label.shape[1]), dtype=torch.float32) / 3

        final_output = self.second_model(z_noise, x_label)

        return final_output


def draw_bar_fig(explainer_plt, fig_path, x_bound_right):
    # 创建一个新的图形，并设置图幅
    fig, ax = plt.subplots(figsize=(4, 6))  # 设置图幅大小为 10x6 英寸

    fig.subplots_adjust(left=0.28, right=0.99, top=1, bottom=0.16)

    shap.plots.bar(explainer_plt, max_display=17, show_data=False, ax=ax, show=False)

    # 设置标题和轴标签的字体
    ax.set_title(ax.get_title(), fontsize=22, fontweight='bold')
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)

    # 设置 x 轴标签
    ax.set_xlabel('mean(|SHAP value|)\n(global importance)', fontsize=20)
    # 显示右轴和上轴
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    # 设置右轴和上轴的刻度线不可见
    ax.tick_params(axis='y', which='both', right=False)
    ax.tick_params(axis='x', which='both', top=False)

    # 设置 x 轴刻度线朝上
    ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', grid_color='r', grid_alpha=0.5)

    plt.gca().set_xlim(right=x_bound_right)

    # 设置 x 轴和 y 轴刻度标签的字体
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)

    for text in plt.gca().texts:
        text.set_visible(False)

    # plt.show()
    plt.savefig(fig_path, dpi=600)
    plt.close()


def draw_beeswarm_fig(explainer_plt, fig_path):
    shap.plots.beeswarm(explainer_plt, max_display=17, show=False, plot_size=(8, 6), color_bar_label="")

    # 获取当前的图形和轴
    fig = plt.gcf()
    ax = plt.gca()

    fig.subplots_adjust(left=0.208, right=1, top=0.985, bottom=0.16)

    # 获取散点图的 PathCollection 对象，并调整散点大小
    for pathcollection in ax.collections:
        offsets = pathcollection.get_offsets()
        sizes = pathcollection.get_sizes()
        new_sizes = [size * 1.5 for size in sizes]  # 将散点大小放大 1.5 倍
        pathcollection.set_sizes(new_sizes)

    # 获取当前的颜色条对象
    colorbar = plt.gcf().axes[-1]
    # 修改颜色条字体大小
    colorbar.tick_params(labelsize=18)

    # 设置标题和轴标签的字体
    ax.set_title(ax.get_title(), fontsize=22, fontweight='bold')
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)

    # 设置 y 轴标签
    ax.set_ylabel('Features', fontsize=20)
    # 设置 x 轴标签
    ax.set_xlabel('SHAP value\n(impact on rcGAN output)', fontsize=20)

    # 设置 x 轴刻度线朝上
    ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', grid_color='r', grid_alpha=0.5)

    # 设置 x 轴和 y 轴刻度标签的字体
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)

    # plt.show()
    plt.savefig(fig_path, dpi=600)
    plt.close()


def main():
    seed = 111
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # 准备数据
    X_train = np.genfromtxt("./source_data/input_para/input_label.csv", delimiter=",").astype(np.float32)

    # 初始化模型
    net_para_reg = [4, 256, 128, 64, 32, 16]
    net_para = [4, 256, 128, 64, 32, 16]

    reg_pth_path = "./result/networks/pre-train_reg/pre-train_reg.pth"
    gen_pth_path = "./result/networks/gen/generator_lf_smoothL1_atten2.pth"

    regression = Regression03(3, 17, 1, True, 0.15, net_para_reg[0], net_para_reg[1], net_para_reg[2], net_para_reg[3], net_para_reg[4], net_para_reg[5])
    regression.load_state_dict(torch.load(reg_pth_path))
    generator = Generator04(3, 3, 2, True, 0.2, net_para[0], net_para[1], net_para[2], net_para[3], net_para[4], net_para[5])
    generator.load_state_dict(torch.load(gen_pth_path))

    regression.eval()
    generator.eval()

    com_model = CombinedModel(regression, generator)
    com_model.eval()

    # 创建一个SHAP解释器，使用整个训练集（仅输入）
    explainer = shap.GradientExplainer(com_model, torch.tensor(X_train))

    # 计算SHAP值，仍然使用训练集（仅输入）
    # shap_values = explainer.shap_values(torch.tensor(X_train))
    shap_values = explainer(torch.tensor(X_train))

    # 定义特征名称
    feature_names = [r'$A$_B3', r'$A$_B2', r'$A$_B1', r'$I$_B3', r'$I$_B2', r'$I$_B1', r'$A$_C3', r'$A$_C2', r'$A$_C1', r'$I$_C3', r'$I$_C2', r'$I$_C1', r'$p_{\mathrm{d}}$_bay1', r'$p_{\mathrm{d}}$_bay2', r'$p_{\mathrm{d}}$_bay3', r'$k_{\mathrm{e}}$', r'$c_{\mathrm{d}}$']

    shap.initjs()

    # 绘制SHAP分析图-bar
    for i in range(3):
        explainer_plt = shap.Explanation(shap_values[:, :, i], feature_names=feature_names)

        # fig_path_bar = "./result/output_fig/bar_fig{}".format(str(i+1))
        fig_path_warm = "./result/output_fig/beeswarm_fig{}".format(str(i+1))

        # if i != 1:
        #     x_bound_right = 0.22
        # else:
        #     x_bound_right = 0.39

        # draw_bar_fig(explainer_plt, fig_path_bar, x_bound_right)
        draw_beeswarm_fig(explainer_plt, fig_path_warm)


if __name__ == '__main__':
    main()

