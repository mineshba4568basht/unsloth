# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unsloth - 2x faster LLM finetuning with 60% less memory."""

__version__ = "2024.12.4"
__author__ = "Unsloth Team"
__license__ = "Apache 2.0"

import sys
import os

# Minimum Python version check
if sys.version_info < (3, 8):
    raise RuntimeError(
        "Unsloth requires Python 3.8 or higher. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}."
    )

def _check_cuda_available():
    """Check if CUDA is available and warn if not."""
    try:
        import torch
        if not torch.cuda.is_available():
            import warnings
            warnings.warn(
                "CUDA is not available. Unsloth requires a CUDA-capable GPU for optimal performance. "
                "Some features may not work correctly.",
                RuntimeWarning,
                stacklevel=3,
            )
        return torch.cuda.is_available()
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. Please install it with: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )


def _check_dependencies():
    """Verify that required dependencies are installed."""
    required = {
        "transformers": "transformers",
        "peft": "peft",
        "trl": "trl",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "triton": "triton",
    }
    missing = []
    for module_name, package_name in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing)}. "
            "Please install them with: pip install unsloth[cu121]"
        )


# Run dependency checks on import
_check_dependencies()
_check_cuda_available()

# Core public API imports
from .models import FastLanguageModel
from .trainer import UnslothTrainer, UnslothTrainingArguments
from .save import save_model, push_to_hub_merged

__all__ = [
    "FastLanguageModel",
    "UnslothTrainer",
    "UnslothTrainingArguments",
    "save_model",
    "push_to_hub_merged",
    "__version__",
]
