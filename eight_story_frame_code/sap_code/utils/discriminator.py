import torch
from torch import nn

from utils.self_attention import SelfAttention


class Discriminator(nn.Module):
    def __init__(self, output_dim, x_dim, self_attention, dropout_para, hid_layer_num, first_layer_node_num, hid_2_node_num, hid_3_node_num, hid_4_node_num, hid_5_node_num, hidden_last_node):
        """
        :param output_dim: 预测值/输出变量的维数
        :param x_dim: 标签/输入特征的维数
        :param self_attention: 注意力层布置
        """
        super(Discriminator, self).__init__()
        self.output_dim = output_dim
        self.x_dim = x_dim
        self.attention = self_attention
        self.dropout_para = dropout_para

        self.hid_layer_num = hid_layer_num
        self.first_layer_node_num = first_layer_node_num

        self.hid_2_node_num = hid_2_node_num
        self.hid_3_node_num = hid_3_node_num
        self.hid_4_node_num = hid_4_node_num
        self.hid_5_node_num = hid_5_node_num

        self.hidden_last_node = hidden_last_node

        def block(in_feat, out_feat):
            layers = [nn.utils.spectral_norm(nn.Linear(in_feat, out_feat)), nn.Dropout(self.dropout_para),
                      nn.LeakyReLU(0.2, inplace=True)]

            return layers

        self.input_layer = nn.Linear(self.output_dim + self.x_dim, self.first_layer_node_num)
        self.leakyRelu0 = nn.LeakyReLU(0.2, inplace=True)

        self.hidden = nn.Sequential(
            *block(self.first_layer_node_num, self.hid_2_node_num),
        )

        self.last_hid_layer_1 = nn.Sequential(*block(self.hid_2_node_num, self.hid_3_node_num))
        atten_node = self.hid_3_node_num

        self.last_hid_layer = nn.Sequential(*block(self.hid_3_node_num, self.hidden_last_node))  # 这一层是为了避免注意力层直接与输出接触
        out_hid_layer_node = self.hidden_last_node

        if hid_layer_num == 2:
            self.hidden = nn.Sequential(
                *block(self.first_layer_node_num, self.hid_2_node_num),
            )
            atten_node = self.hid_2_node_num  # 鉴别器要多一层

            self.last_hid_layer = nn.Sequential(*block(self.hid_2_node_num, self.hidden_last_node))  # 这一层是为了避免注意力层直接与输出接触
            out_hid_layer_node = self.hidden_last_node

        elif hid_layer_num == 4:
            self.hidden = nn.Sequential(
                *block(self.first_layer_node_num, self.hid_2_node_num),
                *block(self.hid_2_node_num, self.hid_3_node_num),
            )

            self.last_hid_layer_1 = nn.Sequential(*block(self.hid_3_node_num, self.hid_4_node_num))
            atten_node = self.hid_4_node_num
            self.last_hid_layer = nn.Sequential(*block(self.hid_4_node_num, self.hidden_last_node))  # 这一层是为了避免注意力层直接与输出接触
            out_hid_layer_node = self.hidden_last_node

        elif hid_layer_num == 5:
            self.hidden = nn.Sequential(
                *block(self.first_layer_node_num, self.hid_2_node_num),
                *block(self.hid_2_node_num, self.hid_3_node_num),
                *block(self.hid_3_node_num, self.hid_4_node_num),
            )

            self.last_hid_layer_1 = nn.Sequential(*block(self.hid_4_node_num, self.hid_5_node_num))
            atten_node = self.hid_5_node_num
            self.last_hid_layer = nn.Sequential(*block(self.hid_5_node_num, self.hidden_last_node))  # 这一层是为了避免注意力层直接与输出接触
            out_hid_layer_node = self.hidden_last_node

        self.output_layer = nn.utils.spectral_norm(nn.Linear(out_hid_layer_node, 1))

        self.atten1 = SelfAttention(self.first_layer_node_num)
        self.atten2 = SelfAttention(atten_node)

    def forward(self, y_norm, x_input):
        disc_input = torch.cat((x_input, y_norm), -1)

        y = self.input_layer(disc_input)
        y = self.leakyRelu0(y)

        if self.attention == 2 or self.attention == 3:
            y = self.atten1(y)

        y = self.hidden(y)

        if self.hid_layer_num != 2:
            y = self.last_hid_layer_1(y)

        if self.attention == 1 or self.attention == 2:
            y = self.atten2(y)

        y = self.last_hid_layer(y)

        validity = self.output_layer(y)

        return validity
