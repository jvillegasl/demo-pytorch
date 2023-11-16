from typing import Any
from torch.utils.data import Dataset
import random


class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.data = []
        self.transform = transform

        for _ in range(10_000):
            x = random.randrange(-273 * 10000, 500 * 10000) / 10000
            y = x + 273
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple[Any, Any]:
        sample = self.data[index]

        if (self.transform):
            sample = self.transform(sample)

        return sample
