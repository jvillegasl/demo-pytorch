import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataloader(DataLoader):
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn
    ):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_sampler, self.val_sampler = self._split_sampler(
            validation_split)

        self.init_kawrgs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        super().__init__(sampler=self.train_sampler, **self.init_kawrgs)

    def _split_sampler(self, split: float):
        n_samples = len(self.dataset)  # type: ignore

        idx_full = np.arange(n_samples)

        np.random.shuffle(idx_full)

        split_idx = int(split * n_samples)
        val_idx = idx_full[:split_idx]
        train_idx = idx_full[split_idx:]

        train_sampler = SubsetRandomSampler(train_idx)  # type: ignore
        val_sampler = SubsetRandomSampler(val_idx)  # type: ignore

        self.shuffle = False
        # sampler option is mutually exclusive with shuffle
        self.n_samples = len(train_idx)

        return train_sampler, val_sampler

    def get_validation_dataloader(self):
        return DataLoader(sampler=self.val_sampler, **self.init_kawrgs)
