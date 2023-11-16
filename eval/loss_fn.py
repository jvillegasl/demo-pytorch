import torch.nn.functional as F


def loss_fn(output, target):
    return F.l1_loss(output, target)
