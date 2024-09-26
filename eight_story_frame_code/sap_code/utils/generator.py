import torch
import torch.nn as nn

from utils.self_attention import SelfAttention


class Generator(nn.Module):
    def __init__(self, output_dim, x_dim, self_attention, dropout, dropout_para, hid_layer_num, first_layer_node_num, hid_2_node_num, hid_3_node_num, hid_4_node_num, hid_5_node_num):
        super(Generator, self).__init__()
        self.output_dim = output_dim
        self.x_dim = x_dim
        self.attention = self_attention
        self.dropout = dropout
        self.dropout_para = dropout_para

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
                layers.append(nn.Dropout(self.dropout_para))

            layers.append(nn.ReLU())
            return layers

        self.in_layer = nn.Linear(self.output_dim + self.x_dim, self.first_layer_node_num)
        self.relu0 = nn.ReLU()

        # 默认三层，最少二层；两层时只能使用单层注意力
        self.hidden = nn.Sequential(
            *block(self.first_layer_node_num, self.hid_2_node_num, dropout_opt=self.dropout),
        )
        atten_node = self.hid_2_node_num

        self.last_hid_layer = nn.Sequential(*block(self.hid_2_node_num, self.hid_3_node_num, dropout_opt=self.dropout))  # 这一层是为了避免注意力层直接与输出接触
        out_hid_layer_node = self.hid_3_node_num

        if hid_layer_num == 2:
            self.hidden = nn.Sequential(
                *block(self.first_layer_node_num, self.hid_2_node_num, dropout_opt=self.dropout),
            )
            atten_node = self.first_layer_node_num

            out_hid_layer_node = self.hid_2_node_num

        elif hid_layer_num == 4:
            self.hidden = nn.Sequential(
                *block(self.first_layer_node_num, self.hid_2_node_num, dropout_opt=self.dropout),
                *block(self.hid_2_node_num, self.hid_3_node_num, dropout_opt=self.dropout),
            )
            atten_node = self.hid_3_node_num

            self.last_hid_layer = nn.Sequential(*block(self.hid_3_node_num, self.hid_4_node_num, dropout_opt=self.dropout))  # 这一层是为了避免注意力层直接与输出接触
            out_hid_layer_node = self.hid_4_node_num

        elif hid_layer_num == 5:
            self.hidden = nn.Sequential(
                *block(self.first_layer_node_num, self.hid_2_node_num, dropout_opt=self.dropout),
                *block(self.hid_2_node_num, self.hid_3_node_num, dropout_opt=self.dropout),
                *block(self.hid_3_node_num, self.hid_4_node_num, dropout_opt=self.dropout),
            )
            atten_node = self.hid_4_node_num

            self.last_hid_layer = nn.Sequential(*block(self.hid_4_node_num, self.hid_5_node_num, dropout_opt=self.dropout))  # 这一层是为了避免注意力层直接与输出接触
            out_hid_layer_node = self.hid_5_node_num

        # output layer
        self.out_layer = nn.Linear(out_hid_layer_node, self.output_dim)

        self.tanh = nn.Tanh()

        self.atten1 = SelfAttention(self.first_layer_node_num)
        self.atten2 = SelfAttention(atten_node)

    def forward(self, noise_value, x_label):
        gen_input = torch.cat((x_label, noise_value), -1)

        y = self.in_layer(gen_input)
        y = self.relu0(y)

        if self.attention == 1 or self.attention == 2:
            y = self.atten1(y)

        y = self.hidden(y)

        if self.attention == 2 or self.attention == 3:
            y = self.atten2(y)

        if self.hid_layer_num != 2:
            y = self.last_hid_layer(y)

        y = self.out_layer(y)

        y_norm = self.tanh(y)

        return y_norm
