import random
from torch import Tensor

from dataloader.dataloader import MyDataloader


class TestDataloader():

    @classmethod
    def setup_class(cls):
        cls.batch_size = random.randint(10, 500)
        cls.validation_split = random.random()

        cls.dl = MyDataloader(
            batch_size=cls.batch_size,
            num_workers=1,
            shuffle=True,
            validation_split=cls.validation_split
        )

    def test_batch_shape(self):
        batch = next(iter(self.dl))

        assert isinstance(batch, tuple)
        assert len(batch) == 2

        xb, yb = batch

        assert isinstance(xb, Tensor)
        assert isinstance(yb, Tensor)

        assert xb.shape == yb.shape == (self.batch_size,)
