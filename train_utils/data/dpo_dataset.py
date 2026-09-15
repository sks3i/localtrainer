"""DPO dataset wrapper: loads a preference dataset with chosen/rejected pairs."""

from __future__ import annotations

import logging
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from train_utils.configs.base_config import RunConfig
from train_utils.configs.dpo_config import DPOConfig
from train_utils.data.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# Expected column names in the raw dataset.
_REQUIRED_COLUMNS: frozenset[str] = frozenset({"prompt", "chosen", "rejected"})


class DPODataset(BaseDataset):
	"""Dataset adapter for DPO / preference alignment training.

	The source dataset must contain at minimum three columns:

	* ``prompt`` — the input prompt string.
	* ``chosen`` — the preferred completion string.
	* ``rejected`` — the non-preferred completion string.

	These map directly to the columns expected by TRL's :class:`DPOTrainer`.
	Any extra columns are dropped to keep memory usage lean.
	"""

	def __init__(self, config: RunConfig, tokenizer: PreTrainedTokenizerBase) -> None:
		"""Initialise the DPO dataset.

		Args:
			config: Full run config. ``config.dpo`` must not be ``None``.
			tokenizer: Tokenizer (not used for column formatting but available for subclasses).

		Raises:
			ValueError: If ``config.dpo`` is ``None``.
		"""
		super().__init__(config, tokenizer)
		if config.dpo is None:
			raise ValueError("RunConfig.dpo must be set when using DPODataset.")
		self.dpo_cfg: DPOConfig = config.dpo

	def _load_split(self, split: str) -> Dataset:
		"""Load a dataset split and validate that required columns are present.

		Args:
			split: Split name to load.

		Returns:
			Loaded and validated :class:`Dataset`.

		Raises:
			ValueError: If required columns (prompt/chosen/rejected) are missing.
		"""
		logger.info("Loading DPO dataset %s (split=%s)", self.dpo_cfg.dataset_id, split)
		dataset = load_dataset(
			self.dpo_cfg.dataset_id,
			split=split,
			**self.dpo_cfg.dataset_kwargs,
		)
		missing = _REQUIRED_COLUMNS - set(dataset.column_names)
		if missing:
			raise ValueError(
				f"DPO dataset '{self.dpo_cfg.dataset_id}' is missing required columns: {missing}. "
				f"Available columns: {dataset.column_names}"
			)
		if self.dpo_cfg.max_num_samples is not None:
			dataset = dataset.select(range(min(self.dpo_cfg.max_num_samples, len(dataset))))
			logger.info("Capped DPO dataset to %d samples.", len(dataset))
		return dataset

	def _keep_required_columns(self, dataset: Dataset) -> Dataset:
		"""Drop all columns except prompt/chosen/rejected.

		TRL's DPOTrainer is strict about unexpected columns during tokenisation.

		Args:
			dataset: Full dataset with possibly many columns.

		Returns:
			Dataset with only the three required columns.
		"""
		extra = set(dataset.column_names) - _REQUIRED_COLUMNS
		if extra:
			logger.debug("Dropping extra DPO dataset columns: %s", extra)
			dataset = dataset.remove_columns(list(extra))
		return dataset

	def get_train_dataset(self) -> Dataset:
		"""Return the DPO training split with only required columns.

		Returns:
			Training :class:`Dataset` with prompt / chosen / rejected columns.
		"""
		dataset = self._load_split(self.dpo_cfg.dataset_split)
		return self._keep_required_columns(dataset)

	def get_eval_dataset(self) -> Optional[Dataset]:
		"""Return the DPO evaluation split if ``dpo.eval_dataset_split`` is set.

		Returns:
			Evaluation :class:`Dataset` or ``None``.
		"""
		if self.dpo_cfg.eval_dataset_split is None:
			return None
		dataset = self._load_split(self.dpo_cfg.eval_dataset_split)
		return self._keep_required_columns(dataset)
