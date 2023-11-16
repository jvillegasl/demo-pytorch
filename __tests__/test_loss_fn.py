import torch
import random

from eval.loss_fn import loss_fn


class TestLossFn():

    @classmethod
    def setup_class(cls):
        N = random.randint(10, 100)

        cls.output = torch.rand(N)

    def test_zero_loss(self):
        assert loss_fn(self.output, self.output) == 0

    def test_increasing_loss(self):
        output = self.output
        target = output.clone()

        prev_loss = 0
        for i, _ in enumerate(output):
            output[i] += 1

            new_loss = loss_fn(output, target)

            assert new_loss > prev_loss

            prev_loss = new_loss
