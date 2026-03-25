import torch
import torch.nn.functional as F

from code.pretrain.base import PretrainTask
from code.finetune.task_heads import TaskAwareObjective, build_task_aware_classifier


class FinetuneSupervised(PretrainTask):
    """Supervised finetuning task that uses the finetune dataset config for heads."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = getattr(getattr(cfg, "finetune", None), "dataset", None) or getattr(cfg, "dataset", None)
        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        out_repr_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        self.repr_dim = out_repr_dim
        self.objective = TaskAwareObjective(cfg, repr_dim=self.repr_dim)
        self.task_level = self.objective.task_level
        self.task_type = self.objective.task_type
        self.label_dim = self.objective.label_dim
        self.classifier = build_task_aware_classifier(
            input_dim=self.repr_dim,
            task_type=self.task_type,
            label_dim=self.label_dim,
            num_classes=self.objective.num_classes,
        )

    def _forward(self, model, data, device, mask_attr: str = "train_mask", return_outputs: bool = False):
        data = data.to(device)
        node_repr, graph_repr = model(data)
        return self.objective.forward_with_model_outputs(
            classifier=self.classifier,
            node_repr=node_repr,
            graph_repr=graph_repr,
            data=data,
            device=device,
            mask_attr=mask_attr,
            return_outputs=return_outputs,
        )

    def step(self, model, data, device):
        loss, primary = self._forward(model=model, data=data, device=device, mask_attr="train_mask")
        if self.task_type == "regression":
            return loss, {"train_mae": primary}
        return loss, {"train_acc": primary}

    def evaluate(self, model, data, device):
        loss, primary = self._forward(model=model, data=data, device=device, mask_attr="val_mask")
        if self.task_type == "regression":
            return loss, {"val_mae": primary}
        return loss, {"val_acc": primary}
