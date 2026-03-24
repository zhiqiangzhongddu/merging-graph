from typing import Optional

import torch
from torch import nn


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        width = in_dim
        num_layers = max(1, int(num_layers))
        for layer_idx in range(num_layers - 1):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hidden_dim
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_head(
    in_dim: int,
    task_type: str,
    num_classes: Optional[int],
    head_type: str = "linear",
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.0,
) -> nn.Module:
    task = task_type.lower()
    if task == "classification":
        if num_classes is None:
            raise ValueError("num_classes is required for classification heads")
        out_dim = int(num_classes)
    elif task == "regression":
        out_dim = 1
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    if head_type == "mlp":
        return MLPHead(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    return LinearHead(in_dim=in_dim, out_dim=out_dim)
