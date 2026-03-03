from typing import Any, Dict, List

from code.pretrain.pretrainer import PretrainRunner


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def _looks_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "false", "1", "0", "yes", "no", "y", "n")


def _looks_int(value: str) -> bool:
    try:
        int(str(value).strip())
        return True
    except Exception:
        return False


def parse_pretrain_tasks(tsv_path: str) -> List[Dict[str, Any]]:
    """
    Read pretrain tasks from TSV.

    Supported row formats:
    1) `model dataset task_level induced method`
    2) `model dataset task_level induced method [batch] [seed] [epochs]`
    3) `dataset task_level induced method [epochs] [batch] [seed]`
    """
    tasks: List[Dict[str, Any]] = []
    if not tsv_path:
        return tasks

    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 4:
                    continue
                task = None
                # Format 1/2: model dataset task_level induced method [batch] [seed] [epochs]
                if (
                    len(parts) >= 5
                    and _looks_bool(parts[3])
                ):
                    model, dataset, task_level, induced_str, method = parts[:5]
                    batch = int(parts[5]) if len(parts) > 5 and _looks_int(parts[5]) else None
                    seed = int(parts[6]) if len(parts) > 6 and _looks_int(parts[6]) else None
                    epochs = int(parts[7]) if len(parts) > 7 and _looks_int(parts[7]) else None
                    task = {
                        "model": model,
                        "dataset": dataset,
                        "task_level": task_level,
                        "induced": _parse_bool(induced_str),
                        "method": method,
                        "epochs": epochs,
                        "batch": batch,
                        "seed": seed,
                    }
                # Format 3: dataset task_level induced method [epochs] [batch] [seed]
                elif (
                    len(parts) >= 4
                    and _looks_bool(parts[2])
                ):
                    dataset, task_level, induced_str, method = parts[:4]
                    epochs = int(parts[4]) if len(parts) > 4 and _looks_int(parts[4]) else None
                    batch = int(parts[5]) if len(parts) > 5 and _looks_int(parts[5]) else None
                    seed = int(parts[6]) if len(parts) > 6 and _looks_int(parts[6]) else None
                    task = {
                        "dataset": dataset,
                        "task_level": task_level,
                        "induced": _parse_bool(induced_str),
                        "method": method,
                        "epochs": epochs,
                        "batch": batch,
                        "seed": seed,
                    }
                if task is not None:
                    tasks.append(task)
    except FileNotFoundError:
        print(f"[Pretrain] Tasks TSV not found: {tsv_path}")

    return tasks


def _build_task_cfg(base_cfg, task: Dict[str, Any]):
    run_cfg = base_cfg.clone()
    if "model" in task and task["model"]:
        run_cfg.model.name = task["model"]
    run_cfg.pretrain.dataset.name = task["dataset"]
    run_cfg.pretrain.dataset.task_level = task["task_level"]
    run_cfg.pretrain.dataset.induced = task["induced"]
    run_cfg.pretrain.dataset.num_classes = None
    run_cfg.pretrain.dataset.label_dim = None
    run_cfg.model.in_dim = 0
    run_cfg.pretrain.method = task["method"]
    if task.get("epochs") is not None:
        run_cfg.pretrain.epochs = task["epochs"]
    if task.get("batch") is not None:
        run_cfg.pretrain.batch_size = task["batch"]
    if task.get("seed") is not None:
        run_cfg.seed = task["seed"]
    return run_cfg


def run_pretrain_tasks(cfg) -> int:
    """Run all pretrain tasks defined in cfg.pretrain.tasks_tsv."""
    tasks = parse_pretrain_tasks(getattr(cfg.pretrain, "tasks_tsv", ""))
    if not tasks:
        print("[Pretrain] No tasks found to run.")
        return 1

    results: List[bool] = []
    for task in tasks:
        run_cfg = _build_task_cfg(cfg, task)
        seed_value = task.get("seed", getattr(run_cfg, "seed", None))
        try:
            runner = PretrainRunner(run_cfg)
            runner.fit()
            results.append(True)
        except Exception as exc:
            print(
                f"[Pretrain] Failed {task['dataset']} ({task['task_level']}, induced={task['induced']}) "
                f"{task['method']} seed={seed_value}: {exc}"
            )
            results.append(False)

    return 0 if all(results) else 1
