"""Prompt-aware encoder modules for finetuning."""

from .prompt_encoder import (
    PromptGCNConv,
    PromptGINConv,
    PromptGNNEncoder,
    build_prompt_encoder_from_cfg,
    resolve_edgeprompt_add_self_loops_from_cfg,
)

__all__ = [
    "PromptGCNConv",
    "PromptGINConv",
    "PromptGNNEncoder",
    "build_prompt_encoder_from_cfg",
    "resolve_edgeprompt_add_self_loops_from_cfg",
]
