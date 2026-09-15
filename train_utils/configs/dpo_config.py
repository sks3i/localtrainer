"""DPO (Direct Preference Optimization) specific configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DPOConfig:
	"""Configuration for preference alignment training via TRL DPOTrainer.

	Attributes:
		dataset_id: HuggingFace dataset hub ID or local path. Must have
		    ``prompt``, ``chosen``, and ``rejected`` columns.
		dataset_split: Dataset split to use for training.
		eval_dataset_split: Dataset split for evaluation. ``None`` skips eval.
		beta: KL penalty coefficient. Higher values stay closer to the reference model.
		loss_type: DPO loss variant — "sigmoid" | "hinge" | "ipo" | "kto_pair".
		reference_model_id: HF model ID for the frozen reference model.
		    ``None`` uses the same model_id as the policy with no LoRA applied.
		label_smoothing: Label smoothing for the DPO loss (sigmoid variant only).
		max_prompt_length: Maximum token length for the prompt portion.
		max_length: Maximum total token length (prompt + response).
		max_num_samples: Cap the dataset size. ``None`` uses the full split.
	"""

	dataset_id: str = "trl-lib/ultrafeedback_binarized"
	dataset_split: str = "train"
	eval_dataset_split: Optional[str] = "test"
	beta: float = 0.1
	loss_type: str = "sigmoid"
	reference_model_id: Optional[str] = None
	label_smoothing: float = 0.0
	max_prompt_length: int = 512
	max_length: int = 1024
	max_num_samples: Optional[int] = None
	dataset_kwargs: dict = field(default_factory=dict)
