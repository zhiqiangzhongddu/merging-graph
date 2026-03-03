from typing import Dict, Iterable, Optional

import torch
from torch import nn


class FinetuneTask(nn.Module):
    name = "base"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def parameters_to_optimize(self) -> Iterable[nn.Parameter]:
        return self.parameters()

    def build_optimizers(self, model: nn.Module) -> Optional[Dict[str, torch.optim.Optimizer]]:
        return None

    def train_epoch(self, model: nn.Module, loader, device, optimizers=None):
        raise NotImplementedError

    def evaluate_split(self, model: nn.Module, loader, device, prefix: str, mask_attr: str):
        raise NotImplementedError

    def on_epoch_end(self, model: nn.Module, loader, device):
        return None
