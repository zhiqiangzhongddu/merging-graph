"""
Utilities for querying an LLM to summarize pretrained model context
including architecture, pretraining method, and dataset usage.
"""

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

try:
    import api_key as api_key_module  # project root file holding OPENAI_API_KEY
except Exception:
    api_key_module = None

from code.utils import ensure_dir


class LLMModelDescriber:
    """LLM client wrapper for generating pretrained model context."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = self._get_openai_api_key(api_key)
        self.client = OpenAI(api_key=self.api_key)

    def _get_openai_api_key(self, explicit: Optional[str] = None) -> str:
        """Resolve the OpenAI API key from argument, api_key.py, or env var."""
        if explicit:
            return explicit
        if api_key_module and getattr(api_key_module, "OPENAI_API_KEY", None):
            return api_key_module.OPENAI_API_KEY
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY or add it to api_key.py.")

    @staticmethod
    def _summarize_hyperparams(pretrain_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Extract commonly useful hyperparameters for summarization."""
        summary = {}
        for key in ("epochs", "batch_size", "lr", "weight_decay", "num_workers"):
            if key in pretrain_cfg:
                summary[key] = pretrain_cfg[key]
        summary["monitor_metric"] = "loss"
        return summary

    @staticmethod
    def _build_prompt(
        architecture: Dict[str, Any],
        dataset_context: Dict[str, Any],
        pretrain_cfg: Dict[str, Any],
    ) -> str:
        """Create a concise prompt describing what we need from the LLM."""
        model_str = architecture.get("model", "")
        task_str = architecture.get("task", "")
        dataset_meta = architecture.get("dataset_meta", {})

        dataset_desc = dataset_context.get("dataset_description", {}) or {}
        task_desc = dataset_context.get("task_description", {}) or {}

        dataset_name = dataset_desc.get("dataset", {}).get("name") or dataset_meta.get("name", "")
        task_level = task_desc.get("task", {}).get("type") or ""
        method = pretrain_cfg.get("method", "unknown")
        hyperparams = LLMModelDescriber._summarize_hyperparams(pretrain_cfg)

        return (
            "You are summarizing a pretrained graph encoder.\n"
            "Provide a crisp JSON response with keys:\n"
            "- architecture_summary: 1-3 sentences describing the encoder and prediction head using the provided text.\n"
            "- pretrain_method: describe the training objective/method.\n"
            "- dataset_usage: 1-2 sentences on the dataset/task used for pretraining.\n"
            "- key_hyperparameters: echo important numeric settings.\n"
            f"Model architecture:\n{model_str}\n"
            f"Task head:\n{task_str}\n"
            f"Dataset meta: {json.dumps(dataset_meta)}\n"
            f"Dataset description: {dataset_desc.get('description', '')}\n"
            f"Task description: {json.dumps(task_desc.get('task', {}))}\n"
            f"Method: {method}\n"
            f"Hyperparameters: {json.dumps(hyperparams)}\n"
            f"Dataset name: {dataset_name}; Task type: {task_level}\n"
        )

    def describe_model(
        self,
        *,
        architecture: Dict[str, Any],
        dataset_context: Dict[str, Any],
        pretrain_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call an OpenAI chat model to produce a model context summary.

        Returns a dictionary parsed from the model's JSON response, with a fallback
        containing the raw content if JSON parsing fails.
        """
        cfg = pretrain_cfg if pretrain_cfg is not None else architecture.get("config", {}).get("pretrain", {})
        prompt = self._build_prompt(
            architecture=architecture,
            dataset_context=dataset_context,
            pretrain_cfg=cfg,
        )

        messages = [
            {"role": "system", "content": "Be concise, factual, and format your reply as JSON."},
            {"role": "user", "content": prompt},
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"raw": content}
        return parsed

    def save_context(
        self,
        *,
        context: Dict[str, Any],
        architecture: Dict[str, Any],
        dataset_context: Dict[str, Any],
        pretrain_cfg: Dict[str, Any],
        output_root: str,
        run_name: str,
        source_paths: Dict[str, Any],
    ) -> str:
        """
        Save model context to disk and return the written path.
        """
        out_dir = os.path.join(output_root, "model", run_name)
        ensure_dir(out_dir)

        out_path = os.path.join(out_dir, "model_llm_context.json")
        payload = {
            "run_name": run_name,
            "source_paths": source_paths,
            "architecture": {
                "model": architecture.get("model"),
                "task": architecture.get("task"),
                "dataset_meta": architecture.get("dataset_meta"),
            },
            "pretrain_config": pretrain_cfg,
            "dataset_context": dataset_context,
            "llm_context": context,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_path
