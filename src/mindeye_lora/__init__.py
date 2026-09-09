"""LoRA vs. full fine-tuning for MindEye2 subject adaptation."""

__version__ = "0.1.0"

from .config import ArmConfig, ExperimentConfig, load_config  # noqa: F401
from .env import Workspace, get_workspace, setup_environment  # noqa: F401
from .lora import (  # noqa: F401
    LoRAConfig,
    LoRALinear,
    apply_lora,
    merge_lora,
    set_trainable,
)

__all__ = [
    "ArmConfig",
    "ExperimentConfig",
    "LoRAConfig",
    "LoRALinear",
    "Workspace",
    "apply_lora",
    "get_workspace",
    "load_config",
    "merge_lora",
    "set_trainable",
    "setup_environment",
]
