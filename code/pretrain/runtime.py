"""Runtime orchestration for the `run_pretrain.py` entrypoint."""

from __future__ import annotations

from datetime import datetime

from code.utils.run_helpers import aggregate_run_metrics, checkpoint_path_for_runner, collect_run_metrics
from code.utils.save_results import append_workflow_result

from .pretrainer import PretrainRunner
from .utils import run_pretrain_tasks


def run_pretrain(cfg) -> int:
    """Execute one pretraining workflow for the provided config."""
    if getattr(cfg.pretrain, "run_all", False):
        return run_pretrain_tasks(cfg)

    started_at = datetime.now().astimezone()
    run_cfg = cfg.clone()
    run_cfg.seed = int(getattr(cfg, "seed", 0))
    runner = PretrainRunner(run_cfg)
    runner.fit()

    ended_at = datetime.now().astimezone()
    skipped = bool(getattr(runner, "_skip_due_to_existing_checkpoint", False))
    if skipped and not bool(getattr(cfg.save_results, "save_skipped", False)):
        return 0

    summary = aggregate_run_metrics([collect_run_metrics(runner, log_prefix="[Pretrain][Summary]")])
    metric_summary = summary["metric_stats"]
    best_epochs = summary["epoch_values"].get("best_epoch")

    append_workflow_result(
        cfg=run_cfg,
        workflow="pretrain",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=[checkpoint_path_for_runner(runner)],
        seeds=[int(run_cfg.seed)],
        best_epochs=best_epochs,
        metric_summary=metric_summary,
    )
    return 0
