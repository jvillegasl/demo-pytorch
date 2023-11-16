import torch
from torch import Tensor
import random

from model.model import MyModel


class TestModel():

    @classmethod
    def setup_class(cls):
        cls.model = MyModel()

    def test_prediction_shape(self):
        N = random.randint(1, 100)
        x = torch.rand(N)

        y = self.model(x)

        assert isinstance(y, Tensor)
        assert y.shape == (N,)
