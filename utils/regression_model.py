import torch
import torch.nn as nn

from utils.self_attention import SelfAttention


class Regression03(nn.Module):
    def __init__(self, output_dim, x_dim, self_attention, dropout, drop_para, hid_layer_num, first_layer_node_num, hid_2_node_num, hid_3_node_num, hid_4_node_num, hid_5_node_num):
        """
        :param output_dim:  输出特征维数，默认为3
        :param x_dim:  样本的输入特征维数，即label维数，默认为12
        :param self_attention:  添加自注意力机制的选择
        :param dropout:  添加dropout的选择
        :param hid_layer_num:  网络层数，默认为4，至少为3，最多为5
        :param first_layer_node_num:  第一隐藏层节点数，默认为16
        """

        super(Regression03, self).__init__()
        self.output_dim = output_dim
        self.x_dim = x_dim
        self.attention = self_attention
        self.dropout = dropout

        self.hid_layer_num = hid_layer_num
        self.first_layer_node_num = first_layer_node_num

        self.hid_2_node_num = hid_2_node_num
        self.hid_3_node_num = hid_3_node_num
        self.hid_4_node_num = hid_4_node_num
        self.hid_5_node_num = hid_5_node_num

        def block(in_feat, out_feat, dropout_opt, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.1))
            if dropout_opt:
                layers.append(nn.Dropout(drop_para))

            layers.append(nn.ReLU())
            return layers

        self.in_layer = nn.Linear(self.x_dim, self.first_layer_node_num)
        self.relu0 = nn.ReLU()

        self.hidden = nn.Sequential(
            *block(self.first_layer_node_num, self.hid_2_node_num, dropout_opt=self.dropout),
            *block(self.hid_2_node_num, self.hid_3_node_num, dropout_opt=self.dropout),
        )
        atten_node = self.hid_3_node_num

        self.last_hid_layer = nn.Sequential(*block(self.hid_3_node_num, self.hid_4_node_num, dropout_opt=self.dropout))
        out_hid_layer_node = self.hid_4_node_num

        if hid_layer_num == 3:
            self.hidden = nn.Sequential(
                *block(self.first_layer_node_num, self.hid_2_node_num, dropout_opt=self.dropout),
            )
            atten_node = self.hid_2_node_num

            self.last_hid_layer = nn.Sequential(*block(self.hid_2_node_num, self.hid_3_node_num, dropout_opt=self.dropout))
            out_hid_layer_node = self.hid_3_node_num

        elif hid_layer_num == 5:
            self.hidden = nn.Sequential(
                *block(self.first_layer_node_num, self.hid_2_node_num, dropout_opt=self.dropout),
                *block(self.hid_2_node_num, self.hid_3_node_num, dropout_opt=self.dropout),
                *block(self.hid_3_node_num, self.hid_4_node_num, dropout_opt=self.dropout),
            )
            atten_node = self.hid_4_node_num

            self.last_hid_layer = nn.Sequential(*block(self.hid_4_node_num, self.hid_5_node_num, dropout_opt=self.dropout))
            out_hid_layer_node = self.hid_5_node_num

        self.out_layer = nn.Linear(out_hid_layer_node, self.output_dim)

        self.tanh = nn.Tanh()

        self.atten1 = SelfAttention(self.first_layer_node_num)
        self.atten2 = SelfAttention(atten_node)

    def forward(self, x_label):
        y = self.in_layer(x_label)
        y = self.relu0(y)

        if self.attention == 1 or self.attention == 2:
            y = self.atten1(y)

        y = self.hidden(y)

        if self.attention == 2 or self.attention == 3:
            y = self.atten2(y)

        y = self.last_hid_layer(y)

        y = self.out_layer(y)

        y_norm = self.tanh(y)

        return y_norm