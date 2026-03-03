from torch import nn


class PretrainTask(nn.Module):
    """Base class for pretraining objectives."""

    name = "base"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def parameters_to_optimize(self):
        """Return task parameters that should be optimized during pretraining."""
        return self.parameters()

    def step(self, model: nn.Module, data, device):
        raise NotImplementedError
