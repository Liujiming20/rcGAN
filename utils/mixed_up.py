import torch


def mixup_fun(x, y, alpha=1.0):
    batch_size = x.size(0)

    lambda_para = torch.distributions.beta.Beta(alpha, alpha).sample()

    index = torch.randperm(batch_size)

    x_shuffled = x[index]
    y_shuffld = y[index]

    x_mixed = lambda_para * x + (1 - lambda_para) * x_shuffled
    y_mixed = lambda_para * y + (1 - lambda_para) * y_shuffld

    return x_mixed, y_mixed
