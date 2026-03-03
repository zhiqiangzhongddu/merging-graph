"""GraphPrompt modules for finetuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GraphPrompt(nn.Module):
    """Feature-weighted prompt used in official GraphPrompt finetuning."""

    def __init__(
        self,
        in_channels: int,
        init: str = "xavier",
        init_std: float = 0.02,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.init = str(init).lower()
        self.init_std = float(init_std)
        self.weight = nn.Parameter(torch.empty(1, self.in_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init in {"identity", "ones"}:
            with torch.no_grad():
                self.weight.fill_(1.0)
                if self.init_std > 0:
                    self.weight.add_(torch.randn_like(self.weight) * self.init_std)
        elif self.init in {"xavier", "xavier_uniform"}:
            nn.init.xavier_uniform_(self.weight)
        else:
            raise ValueError(f"Unknown GraphPrompt init mode: {self.init}")

    def forward(self, embeddings: Tensor) -> Tensor:
        return embeddings * self.weight


class GraphPromptPlus(nn.Module):
    """Official GraphPrompt+ style: softmax-weighted mixture of prompt masks."""

    def __init__(
        self,
        in_channels: int,
        p_num: int = 4,
        init: str = "xavier",
        init_std: float = 0.02,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.p_num = max(1, int(p_num))
        self.init = str(init).lower()
        self.init_std = float(init_std)

        self.p_list = nn.Parameter(torch.empty(self.p_num, self.in_channels))
        # Learnable weights over prompt masks (official extension uses softmax weights).
        self.temp = nn.Parameter(torch.empty(self.p_num, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init in {"identity", "ones"}:
            with torch.no_grad():
                self.p_list.fill_(1.0)
                if self.init_std > 0:
                    self.p_list.add_(torch.randn_like(self.p_list) * self.init_std)
        elif self.init in {"xavier", "xavier_uniform"}:
            nn.init.xavier_uniform_(self.p_list)
        else:
            raise ValueError(f"Unknown GraphPromptPlus init mode: {self.init}")
        nn.init.uniform_(self.temp, a=0.0, b=0.1)

    def forward(self, embeddings: Tensor) -> Tensor:
        alpha = F.softmax(self.temp, dim=0).view(1, self.p_num)
        prompt = alpha.mm(self.p_list)
        return embeddings * prompt


class GraphPromptPlusStageWise(nn.Module):
    """
    Stage-wise GraphPrompt+ prompt bank inspired by the extension code path.

    Stages:
    - 0: input features
    - 1: after first hidden layer
    - 2: after second hidden layer
    - 3: readout/final node representation
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        p_num: int = 4,
        init: str = "xavier",
        init_std: float = 0.02,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.p_num = max(1, int(p_num))
        self.init = str(init).lower()
        self.init_std = float(init_std)

        self.stage_masks = nn.ParameterDict(
            {
                "0": nn.Parameter(torch.empty(1, self.in_channels)),
                "1": nn.Parameter(torch.empty(1, self.hidden_channels)),
                "2": nn.Parameter(torch.empty(1, self.hidden_channels)),
                "3": nn.Parameter(torch.empty(1, self.out_channels)),
            }
        )
        available_stages = [0]
        if self.num_layers >= 2:
            available_stages.append(1)
        if self.num_layers >= 3:
            available_stages.append(2)
        available_stages.append(3)
        self._active_stage_ids = tuple(available_stages[: self.p_num])
        self.temp = nn.Parameter(torch.empty(len(self._active_stage_ids), 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for mask in self.stage_masks.values():
            if self.init in {"identity", "ones"}:
                with torch.no_grad():
                    mask.fill_(1.0)
                    if self.init_std > 0:
                        mask.add_(torch.randn_like(mask) * self.init_std)
            elif self.init in {"xavier", "xavier_uniform"}:
                nn.init.xavier_uniform_(mask)
            else:
                raise ValueError(f"Unknown GraphPromptPlusStageWise init mode: {self.init}")
        nn.init.uniform_(self.temp, a=0.0, b=0.1)

    @property
    def active_stage_ids(self):
        return self._active_stage_ids

    def has_stage(self, stage_id: int) -> bool:
        return int(stage_id) in self._active_stage_ids

    def stage_coefficients(self) -> Tensor:
        return F.softmax(self.temp, dim=0).view(-1)

    def iter_stage_coefficients(self):
        coeff = self.stage_coefficients()
        for idx, stage_id in enumerate(self._active_stage_ids):
            yield stage_id, coeff[idx]

    def apply_stage(self, stage_id: int, embeddings: Tensor) -> Tensor:
        stage_key = str(int(stage_id))
        if stage_key not in self.stage_masks:
            return embeddings
        mask = self.stage_masks[stage_key]
        if embeddings.size(-1) != mask.size(-1):
            raise ValueError(
                f"Stage-wise prompt dim mismatch at stage={stage_id}: "
                f"expected {int(mask.size(-1))}, got {int(embeddings.size(-1))}."
            )
        return embeddings * mask

    def forward(self, embeddings: Tensor) -> Tensor:
        # Fallback behavior for non-stagewise backbones: weighted mask mixture on same representation.
        terms = []
        for stage_id, coeff in self.iter_stage_coefficients():
            stage_key = str(int(stage_id))
            mask = self.stage_masks[stage_key]
            if embeddings.size(-1) != mask.size(-1):
                continue
            terms.append(coeff.view(1, 1) * (embeddings * mask))
        if not terms:
            return embeddings
        return torch.stack(terms, dim=0).sum(dim=0)


class GraphPromptTuningLoss(nn.Module):
    """Prototype contrastive objective used by GraphPrompt."""

    def __init__(self, tau: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.tau = float(tau)
        self.reduction = str(reduction).lower()

    def forward(self, embeddings: Tensor, centers: Tensor, labels: Tensor) -> Tensor:
        labels = torch.as_tensor(labels).view(-1).long()
        logits = F.cosine_similarity(embeddings.unsqueeze(1), centers.unsqueeze(0), dim=-1)
        logits = logits / max(self.tau, 1e-12)
        losses = F.cross_entropy(logits, labels, reduction="none")
        if self.reduction == "sum":
            return losses.sum()
        if self.reduction == "none":
            return losses
        return losses.mean()


def compute_class_centers(embeddings: Tensor, labels: Tensor, num_classes: int):
    """Compute per-class prototype centers and class counts."""
    labels = torch.as_tensor(labels).view(-1).long()
    num_classes = int(num_classes)
    if labels.numel() == 0:
        centers = embeddings.new_zeros((num_classes, embeddings.size(-1)))
        counts = embeddings.new_zeros((num_classes, 1))
        return centers, counts

    centers = embeddings.new_zeros((num_classes, embeddings.size(-1)))
    index = labels.unsqueeze(1).expand(-1, embeddings.size(-1))
    centers = centers.scatter_add_(dim=0, index=index, src=embeddings)

    counts = torch.bincount(labels, minlength=num_classes).to(
        dtype=embeddings.dtype,
        device=embeddings.device,
    ).unsqueeze(1)
    centers = centers / counts.clamp_min(1.0)
    return centers, counts
