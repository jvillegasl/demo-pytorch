import torch
from torch import Tensor

@torch.no_grad()
def accuracy(output: Tensor, target: Tensor):
    assert output.shape == target.shape

    diff = target - output
    diff_pct = diff / target

    return diff_pct.mean()