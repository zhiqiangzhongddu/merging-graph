import json
import os
from typing import Any, Dict, Optional

import torch


def cfg_to_dict(cfg) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return json.loads(cfg.dump())
    return yaml.safe_load(cfg.dump())


def save_checkpoint(
    path: str,
    model,
    optimizer,
    epoch: int,
    cfg,
    dataset_meta: Dict[str, Any],
    metrics: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg_payload = cfg if isinstance(cfg, dict) else cfg_to_dict(cfg)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg_payload,
        "dataset": dataset_meta,
        "metrics": metrics,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
