import torch
from torch import Tensor


def collate_fn(batch: list[tuple[Tensor, Tensor]]):
    x = [t[0].unsqueeze(0) for t in batch]
    y = [t[1].unsqueeze(0) for t in batch]

    x = torch.cat(x)
    y = torch.cat(y)

    return x, y
