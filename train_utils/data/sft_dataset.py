"""SFT dataset wrapper: loads a HuggingFace dataset and applies chat-template formatting."""

from __future__ import annotations

import logging
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from train_utils.configs.base_config import RunConfig
from train_utils.configs.sft_config import SFTConfig
from train_utils.data.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class SFTDataset(BaseDataset):
	"""Dataset adapter for supervised fine-tuning.

	Supports two input formats:

	1. **Raw text** (``sft.use_chat_template=False``): expects a single text
	   column (``sft.text_field``) containing fully formatted prompt+response.

	2. **Chat messages** (``sft.use_chat_template=True``): expects a column
	   (``sft.messages_field``) of message lists in OpenAI format:
	   ``[{"role": "user", "content": "..."}, ...]``.
	   The tokenizer's ``apply_chat_template`` is called to produce a single
	   ``text`` column, which TRL's SFTTrainer then tokenises.
	"""

	def __init__(self, config: RunConfig, tokenizer: PreTrainedTokenizerBase) -> None:
		"""Initialise the SFT dataset.

		Args:
			config: Full run config. ``config.sft`` must not be ``None``.
			tokenizer: Tokenizer used to apply the chat template.

		Raises:
			ValueError: If ``config.sft`` is ``None``.
		"""
		super().__init__(config, tokenizer)
		if config.sft is None:
			raise ValueError("RunConfig.sft must be set when using SFTDataset.")
		self.sft_cfg: SFTConfig = config.sft

	def _load_split(self, split: str) -> Dataset:
		"""Load a single dataset split from the HuggingFace hub or local path.

		Args:
			split: Dataset split name (e.g. "train", "test").

		Returns:
			The loaded :class:`Dataset` split.
		"""
		logger.info("Loading dataset %s (split=%s)", self.sft_cfg.dataset_id, split)
		dataset = load_dataset(
			self.sft_cfg.dataset_id,
			split=split,
			**self.sft_cfg.dataset_kwargs,
		)
		if self.sft_cfg.max_num_samples is not None:
			dataset = dataset.select(range(min(self.sft_cfg.max_num_samples, len(dataset))))
			logger.info("Capped dataset to %d samples.", len(dataset))
		return dataset

	def _apply_chat_template(self, dataset: Dataset) -> Dataset:
		"""Apply the tokenizer's chat template to a messages column.

		Converts each list of role/content dicts into a single formatted string
		stored in the ``text`` column.

		Args:
			dataset: Dataset containing ``sft.messages_field``.

		Returns:
			Dataset with an additional ``text`` column.
		"""
		messages_field = self.sft_cfg.messages_field

		def _format(example: dict) -> dict:
			text = self.tokenizer.apply_chat_template(
				example[messages_field],
				tokenize=False,
				add_generation_prompt=False,
			)
			return {"text": text}

		logger.info("Applying chat template to column '%s'.", messages_field)
		return dataset.map(_format, remove_columns=[messages_field])

	def get_train_dataset(self) -> Dataset:
		"""Return the training split, optionally with chat template applied.

		Returns:
			Training :class:`Dataset` with a ``text`` column.
		"""
		dataset = self._load_split(self.sft_cfg.dataset_split)
		if self.sft_cfg.use_chat_template:
			dataset = self._apply_chat_template(dataset)
		return dataset

	def get_eval_dataset(self) -> Optional[Dataset]:
		"""Return the evaluation split if ``sft.eval_dataset_split`` is set.

		Returns:
			Evaluation :class:`Dataset` or ``None``.
		"""
		if self.sft_cfg.eval_dataset_split is None:
			return None
		dataset = self._load_split(self.sft_cfg.eval_dataset_split)
		if self.sft_cfg.use_chat_template:
			dataset = self._apply_chat_template(dataset)
		return dataset
