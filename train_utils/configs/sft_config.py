"""SFT (Supervised Fine-Tuning) specific configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTConfig:
	"""Configuration for supervised fine-tuning via TRL SFTTrainer.

	Attributes:
		dataset_id: HuggingFace dataset hub ID or local path.
		dataset_split: Dataset split to use for training (e.g. "train").
		eval_dataset_split: Dataset split for evaluation. ``None`` skips eval.
		text_field: Column name containing the raw text (used when not using chat template).
		use_chat_template: Whether to apply the tokenizer's chat template to ``messages`` column.
		messages_field: Column name containing the chat messages list (role/content dicts).
		packing: Enable sequence packing to fill the context window efficiently.
		max_num_samples: Cap the dataset size. ``None`` uses the full split.
	"""

	dataset_id: str = "mlabonne/guanaco-llama2-1k"
	dataset_split: str = "train"
	eval_dataset_split: Optional[str] = None
	text_field: Optional[str] = "text"
	use_chat_template: bool = False
	messages_field: str = "messages"
	packing: bool = True
	max_num_samples: Optional[int] = None
	dataset_kwargs: dict = field(default_factory=dict)
