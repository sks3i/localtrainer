"""DPO trainer — wraps TRL's DPOTrainer with train_utils config and dataset."""

from __future__ import annotations

import logging

from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from trl import DPOConfig as TRLDPOConfig
from trl import DPOTrainer as TRLDPOTrainer

from train_utils.configs.base_config import RunConfig, TrainingArgs
from train_utils.data.base_dataset import BaseDataset
from train_utils.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def _build_dpo_training_args(config: RunConfig) -> TRLDPOConfig:
	"""Map a :class:`RunConfig` to a TRL :class:`DPOConfig`.

	Args:
		config: Full resolved run config.

	Returns:
		A configured :class:`trl.DPOConfig` instance.
	"""
	t: TrainingArgs = config.training
	dpo = config.dpo

	extra = dict(t.extra_training_args)
	if dpo is not None:
		extra.setdefault("beta", dpo.beta)
		extra.setdefault("loss_type", dpo.loss_type)
		extra.setdefault("label_smoothing", dpo.label_smoothing)
		extra.setdefault("max_prompt_length", dpo.max_prompt_length)
		extra.setdefault("max_length", dpo.max_length)

	return TRLDPOConfig(
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
		eval_steps=t.eval_steps if dpo and dpo.eval_dataset_split else None,
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


def _load_reference_model(config: RunConfig) -> PreTrainedModel | None:
	"""Load the frozen reference model for DPO.

	If ``dpo.reference_model_id`` is ``None``, TRL will create the reference
	by copying the policy weights internally (correct when using LoRA, since
	TRL disables adapter on the ref copy).

	Args:
		config: Full run config.

	Returns:
		Loaded reference model, or ``None`` to let TRL manage it.
	"""
	import torch

	if config.dpo is None or config.dpo.reference_model_id is None:
		return None

	ref_id = config.dpo.reference_model_id
	logger.info("Loading explicit reference model: %s", ref_id)
	ref_model = AutoModelForCausalLM.from_pretrained(
		ref_id,
		torch_dtype=torch.bfloat16,
		device_map="auto",
		trust_remote_code=config.model.trust_remote_code,
	)
	for param in ref_model.parameters():
		param.requires_grad_(False)
	return ref_model


class DPOTrainer(BaseTrainer):
	"""DPO preference alignment trainer backed by TRL's DPOTrainer.

	Loads the preference dataset (prompt/chosen/rejected) from the provided
	:class:`BaseDataset` and delegates all training to TRL. Supports an
	optional explicit reference model; when not provided, TRL creates the
	reference copy internally (recommended for LoRA).
	"""

	def __init__(
		self,
		config: RunConfig,
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizerBase,
		dataset: BaseDataset,
	) -> None:
		"""Initialise the DPO trainer.

		Args:
			config: Full run configuration. ``config.dpo`` must not be ``None``.
			model: Policy model with LoRA adapters already applied.
			tokenizer: Matching tokenizer.
			dataset: Dataset adapter exposing train / eval splits with
			    prompt / chosen / rejected columns.

		Raises:
			ValueError: If ``config.dpo`` is ``None``.
		"""
		super().__init__(config, model, tokenizer, dataset)
		if config.dpo is None:
			raise ValueError("RunConfig.dpo must be set for DPOTrainer.")

	def _build_hf_trainer(self) -> TRLDPOTrainer:
		"""Construct the TRL DPOTrainer.

		Returns:
			A fully configured :class:`trl.DPOTrainer` ready to call ``.train()``.
		"""
		training_args = _build_dpo_training_args(self.config)
		train_dataset = self.dataset.get_train_dataset()
		eval_dataset = self.dataset.get_eval_dataset()
		ref_model = _load_reference_model(self.config)

		logger.info(
			"DPOTrainer | train_samples=%d | eval_samples=%s | ref_model=%s",
			len(train_dataset),
			len(eval_dataset) if eval_dataset else "N/A",
			self.config.dpo.reference_model_id or "TRL-managed",
		)

		return TRLDPOTrainer(
			model=self.model,
			ref_model=ref_model,
			tokenizer=self.tokenizer,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
		)
