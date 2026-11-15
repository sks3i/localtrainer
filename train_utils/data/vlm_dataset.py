"""Image + text SFT dataset: JSONL rows of {image, prompt, answer}, images opened lazily by the collator."""

from __future__ import annotations

import logging
from typing import Optional

from datasets import Dataset, load_dataset

from train_utils.configs.base_config import RunConfig
from train_utils.configs.vlm_sft_config import VLMSFTConfig
from train_utils.data.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("image", "prompt", "answer")


class VLMSFTDataset(BaseDataset):
	"""Loads ``vlm_sft.train_file`` / ``eval_file``. Rows stay as paths + strings; the VLM adapter's
	collator opens images and tokenizes, so the dataset is cheap to build and model agnostic."""

	def __init__(self, config: RunConfig, tokenizer=None) -> None:
		super().__init__(config, tokenizer)
		if config.vlm_sft is None:
			raise ValueError("RunConfig.vlm_sft must be set when using VLMSFTDataset.")
		self.cfg: VLMSFTConfig = config.vlm_sft

	def _load(self, path: str, cap: Optional[int]) -> Dataset:
		ds = load_dataset("json", data_files=path, split="train")
		missing = [c for c in REQUIRED_COLUMNS if c not in ds.column_names]
		if missing:
			raise ValueError(f"{path} is missing columns {missing}")
		if cap is not None and cap < len(ds):
			ds = ds.select(range(cap))
		logger.info("Loaded %d examples from %s", len(ds), path)
		return ds

	def get_train_dataset(self) -> Dataset:
		return self._load(self.cfg.train_file, self.cfg.max_num_samples)

	def get_eval_dataset(self) -> Optional[Dataset]:
		if not self.cfg.eval_file:
			return None
		return self._load(self.cfg.eval_file, self.cfg.max_eval_samples)


class VLMGRPODataset(BaseDataset):
	"""Prompt rows for ``vlm_grpo``: ``{image, prompt, ...}``. The extra keys (targets) go to the
	reward function; there is no answer column, because the model's samples are the training data."""

	def __init__(self, config: RunConfig, tokenizer=None) -> None:
		super().__init__(config, tokenizer)
		if config.vlm_grpo is None:
			raise ValueError("RunConfig.vlm_grpo must be set when using VLMGRPODataset.")
		self.cfg = config.vlm_grpo

	def _load(self, path: str, cap: Optional[int]) -> Dataset:
		# Plain JSON lines, not datasets.load_dataset: targets are nested and vary by source (a mask
		# RLE for some rows, a point list for others), which Arrow would force into one schema.
		import json

		with open(path) as f:
			rows = [json.loads(line) for line in f if line.strip()]
		for c in ("image", "prompt"):
			if rows and c not in rows[0]:
				raise ValueError(f"{path} is missing column {c!r}")
		if cap is not None:
			rows = rows[:cap]
		logger.info("Loaded %d prompts from %s", len(rows), path)
		return rows

	def get_train_dataset(self):
		return self._load(self.cfg.train_file, self.cfg.max_num_samples)

	def get_eval_dataset(self):
		return self._load(self.cfg.dev_file, None) if self.cfg.dev_file else None
