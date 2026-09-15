"""SFT trainer — wraps TRL's SFTTrainer with train_utils config and dataset."""

from __future__ import annotations

import logging

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig as TRLSFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from train_utils.configs.base_config import RunConfig, TrainingArgs
from train_utils.data.base_dataset import BaseDataset
from train_utils.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def _build_sft_training_args(config: RunConfig) -> TRLSFTConfig:
	"""Map a :class:`RunConfig` to a TRL :class:`SFTConfig`.

	Args:
		config: Full resolved run config.

	Returns:
		A configured :class:`trl.SFTConfig` instance.
	"""
	t: TrainingArgs = config.training
	sft = config.sft

	extra = dict(t.extra_training_args)
	if sft is not None:
		extra.setdefault("packing", sft.packing)
		extra.setdefault("max_seq_length", config.model.max_seq_length)
		if sft.text_field and not sft.use_chat_template:
			extra.setdefault("dataset_text_field", sft.text_field)

	return TRLSFTConfig(
		output_dir=config.output_dir,
		run_name=config.run_name,
		num_train_epochs=t.num_train_epochs,
		per_device_train_batch_size=t.per_device_train_batch_size,
		per_device_eval_batch_size=t.per_device_eval_batch_size,
		gradient_accumulation_steps=t.gradient_accumulation_steps,
		learning_rate=t.learning_rate,
		lr_scheduler_type=t.lr_scheduler_type,
		warmup_ratio=t.warmup_ratio,
		weight_decay=t.weight_decay,
		max_grad_norm=t.max_grad_norm,
		logging_steps=t.logging_steps,
		save_steps=t.save_steps,
		eval_steps=t.eval_steps if sft and sft.eval_dataset_split else None,
		save_total_limit=t.save_total_limit,
		seed=t.seed,
		bf16=t.bf16,
		fp16=t.fp16,
		gradient_checkpointing=t.gradient_checkpointing,
		torch_compile=t.torch_compile,
		dataloader_num_workers=t.dataloader_num_workers,
		report_to=t.report_to,
		**extra,
	)


class SFTTrainer(BaseTrainer):
	"""Supervised fine-tuning trainer backed by TRL's SFTTrainer.

	Loads the train/eval datasets from the provided :class:`BaseDataset` and
	delegates all training to TRL. Compatible with both Unsloth-loaded and
	standard HF models.
	"""

	def __init__(
		self,
		config: RunConfig,
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizerBase,
		dataset: BaseDataset,
	) -> None:
		"""Initialise the SFT trainer.

		Args:
			config: Full run configuration. ``config.sft`` must not be ``None``.
			model: Model with LoRA adapters already applied.
			tokenizer: Matching tokenizer.
			dataset: Dataset adapter exposing train / eval splits.

		Raises:
			ValueError: If ``config.sft`` is ``None``.
		"""
		super().__init__(config, model, tokenizer, dataset)
		if config.sft is None:
			raise ValueError("RunConfig.sft must be set for SFTTrainer.")

	def _build_hf_trainer(self) -> TRLSFTTrainer:
		"""Construct the TRL SFTTrainer.

		Returns:
			A fully configured :class:`trl.SFTTrainer` ready to call ``.train()``.
		"""
		training_args = _build_sft_training_args(self.config)
		train_dataset = self.dataset.get_train_dataset()
		eval_dataset = self.dataset.get_eval_dataset()

		logger.info(
			"SFTTrainer | train_samples=%d | eval_samples=%s",
			len(train_dataset),
			len(eval_dataset) if eval_dataset else "N/A",
		)

		return TRLSFTTrainer(
			model=self.model,
			tokenizer=self.tokenizer,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
		)
