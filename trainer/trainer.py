from typing import Any, Callable
import torch
import torch.nn as nn
from torch import Tensor

from base.base_dataloader import BaseDataloader


class Trainer():

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[Any, Any], Tensor],
        metric_fns: list[Callable[[Any, Any], Tensor]],
        optimizer,
        device,
        data_loader: BaseDataloader,
        val_data_loader: BaseDataloader,
        num_epochs: int,
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.device = device
        self.loss_fn = loss_fn
        self.metrics_fns = metric_fns
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def _train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            assert isinstance(data, Tensor)
            assert isinstance(target, Tensor)

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            for metric_fn in self.metrics_fns:
                metric = metric_fn(output, target)
                print(metric_fn.__name__, metric)

            print('Train Epoch {}: {} Loss: {:.6f}'.format(
                epoch,
                self._progress(batch_idx),
                loss.item()
            ))

        self._valid_epoch(epoch)

    def _valid_epoch(self, epoch):
        summary = {
            'epoch': epoch,
            'loss': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_data_loader):
                assert isinstance(data, Tensor)
                assert isinstance(target, Tensor)

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss_fn(output, target)

                for metric_fn in self.metrics_fns:
                    metric = metric_fn(output, target)
                    # print(metric_fn.__name__, metric)

                    if hasattr(summary)

                summary['loss'].append(loss.item())

    def _progress(self, batch_idx):
        template = '[{}/{} ({:.0f}%)]'

        current = batch_idx * self.data_loader.batch_size
        total = self.data_loader.n_samples

        return template.format(current, total, 100.0 * current / total)
