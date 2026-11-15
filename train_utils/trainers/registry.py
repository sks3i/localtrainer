"""Trainer registry — maps trainer_type strings to BaseTrainer subclasses.

To add a new training algorithm:
    1. Implement a new subclass of BaseTrainer in a dedicated file.
    2. Add one entry to TRAINER_REGISTRY below.

That is the only change required to make the new algorithm available via YAML config.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from train_utils.configs.base_config import RunConfig
from train_utils.data.base_dataset import BaseDataset
from train_utils.trainers.base_trainer import BaseTrainer

if TYPE_CHECKING:
	pass

logger = logging.getLogger(__name__)


def _import_trainers() -> dict[str, type[BaseTrainer]]:
	"""Lazily import trainer classes to avoid circular imports at module load.

	Returns:
		Mapping from trainer_type string to trainer class.
	"""
	from train_utils.trainers.dpo_trainer import DPOTrainer
	from train_utils.trainers.sft_trainer import SFTTrainer
	from train_utils.trainers.vlm_grpo_trainer import VLMGRPOTrainer
	from train_utils.trainers.vlm_sft_trainer import VLMSFTTrainer

	return {
		"sft": SFTTrainer,
		"dpo": DPOTrainer,
		"vlm_sft": VLMSFTTrainer,
		"vlm_grpo": VLMGRPOTrainer,
	}


# Public registry — consumers may read this dict to discover available trainers.
TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = _import_trainers()


def build_trainer(
	config: RunConfig,
	model: PreTrainedModel,
	tokenizer: PreTrainedTokenizerBase,
	dataset: BaseDataset,
	**kwargs,
) -> BaseTrainer:
	"""Instantiate the correct trainer for the given :class:`RunConfig`.

	Looks up ``config.trainer_type`` in :data:`TRAINER_REGISTRY` and
	constructs the trainer with the provided model, tokenizer, and dataset.

	Args:
		config: Full resolved run configuration.
		model: Model with LoRA adapters applied.
		tokenizer: Matching tokenizer.
		dataset: Dataset adapter for this run.
		**kwargs: Trainer-specific extras (``vlm_sft`` and ``vlm_grpo`` take ``adapter`` and ``callbacks``).

	Returns:
		A fully initialised :class:`BaseTrainer` subclass.

	Raises:
		ValueError: If ``config.trainer_type`` is not in the registry.

	Example::

		trainer = build_trainer(config, model, tokenizer, dataset)
		trainer.train()
	"""
	trainer_cls = TRAINER_REGISTRY.get(config.trainer_type)
	if trainer_cls is None:
		available = ", ".join(sorted(TRAINER_REGISTRY.keys()))
		raise ValueError(
			f"Unknown trainer_type '{config.trainer_type}'. "
			f"Available trainers: {available}"
		)
	logger.info("Instantiating trainer: %s", trainer_cls.__name__)
	return trainer_cls(config, model, tokenizer, dataset, **kwargs)
