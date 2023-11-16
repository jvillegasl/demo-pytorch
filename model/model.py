import torch
import torch.nn as nn
from torch import Tensor


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer1(x.unsqueeze(-1)).squeeze(-1)
