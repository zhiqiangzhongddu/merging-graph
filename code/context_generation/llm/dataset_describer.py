"""
Utilities for querying an LLM (e.g., OpenAI GPT models) to generate
dataset and task descriptions based on a PyG dataset instance.
"""

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

try:
    import api_key as api_key_module  # project root file holding OPENAI_API_KEY
except Exception:
    api_key_module = None

from code.data_loader.datasets import get_basic_dataset_info
from code.utils import ensure_dir


class LLMDatasetDescriber:
    """LLM client wrapper for generating dataset/task context."""

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
    def _build_prompt(dataset_info: Dict[str, Any]) -> str:
        """Create a concise prompt describing what we need from the LLM."""
        return (
            "You are an expert on graph datasets.\n"
            "Given the dataset metadata below, describe the dataset and its prediction task.\n"
            "Return JSON with keys: "
            "description, "
            "task (task type and target), "
            "and labels (list of objects with name/meaning; include label count if known).\n"
            f"Dataset name: {dataset_info.get('name')}\n"
            f"Domain: {dataset_info.get('domain')}\n"
            f"Source class: {dataset_info.get('source')}\n"
        )

    def describe_dataset(self, dataset: Any) -> Dict[str, Any]:
        """
        Call an OpenAI chat model to produce a dataset description and task details.

        Returns a dictionary parsed from the model's JSON response, with a fallback
        containing the raw content if JSON parsing fails.
        """
        info = get_basic_dataset_info(dataset)

        messages = [
            {"role": "system", "content": "Be concise, factual, and format your reply as JSON."},
            {"role": "user", "content": self._build_prompt(info)},
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
        parsed["dataset"] = info
        return parsed

    def save_context(self, context: Dict[str, Any], output_root: str, folder_name: str) -> Dict[str, str]:
        """
        Save dataset and task context to disk.

        Returns a dict with the written file paths.
        """
        out_dir = os.path.join(output_root, "dataset", folder_name)
        ensure_dir(out_dir)

        dataset_path = os.path.join(out_dir, "dataset_llm_context.json")
        task_path = os.path.join(out_dir, "task_llm_context.json")

        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "description": context.get("description"),
                    "dataset": context.get("dataset"),
                },
                f,
                indent=2,
            )
        with open(task_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": context.get("task"),
                    "labels": context.get("labels"),
                    "dataset": context.get("dataset"),
                },
                f,
                indent=2,
            )
        return {"dataset": dataset_path, "task": task_path}
