from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


class ToTensor(object):
    def __call__(self, sample: tuple[Any, Any]):
        x, y = sample

        return torch.tensor(x), torch.tensor(y)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample: tuple[Tensor, Tensor]):
        x, y = sample

        x = (x - self.std[0]) / self.mean[0]
        y = (y - self.std[1]) / self.mean[1]

        x = x - torch.full_like(x, 1)
        y = y - torch.full_like(y, 1)

        return x, y
