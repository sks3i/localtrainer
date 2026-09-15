"""Abstract base class for all dataset wrappers in train_utils."""

from __future__ import annotations

from abc import ABC, abstractmethod

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from train_utils.configs.base_config import RunConfig


class BaseDataset(ABC):
	"""Abstract interface for dataset wrappers used by train_utils trainers.

	Subclass this to plug a custom dataset into the training framework without
	modifying any trainer code. The trainer calls :meth:`get_train_dataset` and
	optionally :meth:`get_eval_dataset` to obtain HuggingFace :class:`Dataset`
	objects ready for training.

	Example::

		class MyDataset(BaseDataset):
		    def get_train_dataset(self) -> Dataset:
		        return load_dataset("my/dataset", split="train")

		    def get_eval_dataset(self) -> Dataset | None:
		        return None
	"""

	def __init__(self, config: RunConfig, tokenizer: PreTrainedTokenizerBase) -> None:
		"""Initialise with the run config and tokenizer.

		Args:
			config: The full run configuration (use sub-configs as needed).
			tokenizer: Loaded tokenizer for the model being trained.
		"""
		self.config = config
		self.tokenizer = tokenizer

	@abstractmethod
	def get_train_dataset(self) -> Dataset:
		"""Return the training split as a HuggingFace Dataset.

		Returns:
			A :class:`datasets.Dataset` with columns appropriate for the trainer.
		"""
		...

	def get_eval_dataset(self) -> Dataset | None:
		"""Return the evaluation split, or ``None`` to skip evaluation.

		Returns:
			A :class:`datasets.Dataset` or ``None``.
		"""
		return None
