"""Image + text SFT configuration (``trainer_type: vlm_sft``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VLMSFTConfig:
	"""Configuration for supervised fine-tuning of a vision-language model.

	Data is JSONL, one example per line: ``{"image": <path>, "prompt": <str>, "answer": <str>}``.
	Extra keys are kept in the dataset but ignored by the collator. The loss is computed on the
	answer tokens only (plus the end-of-turn token), never on the prompt or the image tokens.

	Attributes:
		train_file: JSONL file with training examples.
		eval_file: JSONL file for eval loss. ``None`` skips eval.
		image_root: Directory that relative ``image`` paths are resolved against.
		max_num_samples: Cap the training set size. ``None`` uses the whole file.
		max_eval_samples: Cap the eval set size.
		max_image_side: Downscale images whose longest side exceeds this (pixels) before the
		    processor sees them, to bound the number of image tokens. ``None`` keeps the original.
		    Normalized targets (e.g. 0-1000 coordinates) are unaffected by resizing.
		save_merged: After training, merge the LoRA adapter into the base weights and save a full
		    model (plus processor) to ``<output_dir>/merged`` so it loads with ``from_pretrained``.
		resume_from_checkpoint: A ``checkpoint-N`` dir to resume (weights, optimizer, scheduler, step).
	"""

	train_file: str = ""
	eval_file: Optional[str] = None
	image_root: Optional[str] = None
	max_num_samples: Optional[int] = None
	max_eval_samples: Optional[int] = 500
	max_image_side: Optional[int] = 1024
	save_merged: bool = True
	resume_from_checkpoint: Optional[str] = None
