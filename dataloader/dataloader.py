from torchvision.transforms import Compose

from dataloader.datasets import MyDataset
import dataloader.transforms as T
from base.base_dataloader import BaseDataloader
from dataloader.collate_fn import collate_fn


class MyDataloader(BaseDataloader):
    def __init__(
            self,
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            validation_split: float
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.validation_split = validation_split

        transform = Compose([
            T.ToTensor(),
            T.Normalize([386.5, 386.5], [-273, 0])
        ])

        self.dataset = MyDataset(transform=transform)

        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            validation_split=validation_split,
            collate_fn=collate_fn
        )
