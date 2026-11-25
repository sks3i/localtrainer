"""VLM SFT trainer: HF Trainer + a VLM adapter's collator + an answer-token cross-entropy."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from train_utils.configs.base_config import RunConfig, TrainingArgs
from train_utils.data.base_dataset import BaseDataset
from train_utils.trainers.base_trainer import BaseTrainer
from train_utils.vlm.adapters import IGNORE_INDEX, VLMAdapter
from train_utils.vlm.setup import export_merged

logger = logging.getLogger(__name__)


def answer_token_loss(logits: torch.Tensor, labels: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
	"""Next-token cross-entropy on supervised positions only.

	Position t predicts token t+1, so logits[:, :-1] are scored against labels[:, 1:]. Only the
	(few) supervised positions are gathered before the fp32 upcast, so a 150k-vocab model does not
	materialize a (B, T, V) fp32 tensor for 300+ image and prompt positions it never scores.

	Normalization: sum over supervised tokens divided by ``num_items_in_batch``, the number of
	supervised tokens in the WHOLE accumulated batch (all micro-batches of one optimizer step).
	Averaging per micro-batch instead and then dividing by the accumulation steps weights a token in
	a short micro-batch more than one in a long micro-batch, so the gradient would depend on how
	examples happen to be grouped. When it is None (eval), this is a plain per-token mean.
	"""
	shift_logits = logits[:, :-1]
	shift_labels = labels[:, 1:].to(logits.device)
	sel = shift_labels != IGNORE_INDEX
	loss_sum = F.cross_entropy(shift_logits[sel].float(), shift_labels[sel], reduction="sum")
	denom = num_items_in_batch if num_items_in_batch is not None else sel.sum().clamp(min=1)
	if torch.is_tensor(denom):
		denom = denom.to(loss_sum.device)
	return loss_sum / denom


class _AnswerLossTrainer(Trainer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# compute_loss below already divides by the token count of the whole accumulated batch, so
		# the Trainer must not divide by gradient_accumulation_steps again.
		self.model_accepts_loss_kwargs = True
		self.loss_is_scaled_for_ga = True

	def create_optimizer(self):
		overrides = getattr(self, "lr_overrides", None) or {}
		if self.optimizer is None and overrides:
			opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
			groups: dict = {}
			for name, p in self.model.named_parameters():
				if not p.requires_grad:
					continue
				lr = next((v for k, v in overrides.items() if k in name), self.args.learning_rate)
				groups.setdefault(lr, []).append(p)
			for lr, ps in sorted(groups.items()):
				logger.info("param group lr=%g: %.1fM params", lr, sum(p.numel() for p in ps) / 1e6)
			self.optimizer = opt_cls([{"params": ps, "lr": lr, "weight_decay": self.args.weight_decay} for lr, ps in groups.items()], **opt_kwargs)
			return self.optimizer
		return super().create_optimizer()

	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		labels = inputs.pop("labels")
		outputs = model(**inputs)
		loss = answer_token_loss(outputs.logits, labels, num_items_in_batch)
		return (loss, outputs) if return_outputs else loss


def _training_arguments(config: RunConfig) -> TrainingArguments:
	import inspect

	t: TrainingArgs = config.training
	has_eval = bool(config.vlm_sft and config.vlm_sft.eval_file)
	# transformers 5 folded warmup_ratio into warmup_steps (a float < 1 is a ratio).
	if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
		warmup = {"warmup_ratio": t.warmup_ratio}
	else:
		warmup = {"warmup_steps": t.warmup_ratio}
	return TrainingArguments(
		output_dir=config.output_dir,
		run_name=config.run_name,
		num_train_epochs=t.num_train_epochs,
		per_device_train_batch_size=t.per_device_train_batch_size,
		per_device_eval_batch_size=t.per_device_eval_batch_size,
		gradient_accumulation_steps=t.gradient_accumulation_steps,
		learning_rate=t.learning_rate,
		lr_scheduler_type=t.lr_scheduler_type,
		**warmup,
		weight_decay=t.weight_decay,
		max_grad_norm=t.max_grad_norm,
		logging_steps=t.logging_steps,
		save_steps=t.save_steps,
		eval_strategy="steps" if has_eval else "no",
		eval_steps=t.eval_steps if has_eval else None,
		save_total_limit=t.save_total_limit,
		seed=t.seed,
		bf16=t.bf16,
		fp16=t.fp16,
		# The adapter enables checkpointing itself (Molmo has no HF gradient_checkpointing_enable).
		gradient_checkpointing=False,
		torch_compile=t.torch_compile,
		dataloader_num_workers=t.dataloader_num_workers,
		report_to=t.report_to,
		remove_unused_columns=False,  # the collator needs image/prompt/answer
		label_names=["labels"],
		**t.extra_training_args,
	)


class VLMSFTTrainer(BaseTrainer):
	"""SFT for vision-language models on {image, prompt, answer} rows.

	``tokenizer`` is the model's processor (it has ``save_pretrained`` like a tokenizer).
	"""

	def __init__(
		self,
		config: RunConfig,
		model,
		tokenizer,
		dataset: BaseDataset,
		adapter: VLMAdapter | None = None,
		callbacks: list | None = None,
	) -> None:
		super().__init__(config, model, tokenizer, dataset)
		if config.vlm_sft is None:
			raise ValueError("RunConfig.vlm_sft must be set for VLMSFTTrainer.")
		if adapter is None:
			raise ValueError("VLMSFTTrainer needs the VLMAdapter that loaded the model (for its collator).")
		self.adapter = adapter
		self.callbacks = callbacks or []

	def _build_hf_trainer(self) -> Trainer:
		if self.config.training.gradient_checkpointing:
			self.adapter.enable_gradient_checkpointing(self.model)
		train_dataset = self.dataset.get_train_dataset()
		eval_dataset = self.dataset.get_eval_dataset()
		logger.info(
			"VLMSFTTrainer | train_samples=%d | eval_samples=%s",
			len(train_dataset),
			len(eval_dataset) if eval_dataset is not None else "N/A",
		)
		trainer = _AnswerLossTrainer(
			model=self.model,
			args=_training_arguments(self.config),
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
			data_collator=self.adapter.collate,
			processing_class=self.tokenizer,
			callbacks=self.callbacks,
		)
		trainer.lr_overrides = dict(self.config.vlm_sft.lr_overrides)
		resume = self.config.vlm_sft.resume_from_checkpoint
		if resume:
			# BaseTrainer.train() calls hf_trainer.train() with no arguments.
			from functools import partial

			logger.info("Resuming from %s", resume)
			trainer.train = partial(trainer.train, resume_from_checkpoint=resume)
		return trainer

	def train(self) -> None:
		super().train()
		cfg = self.config
		if cfg.vlm_sft.save_merged and cfg.model.lora.use_lora:
			# Free the training copy before loading a bf16 base for the merge.
			self.model = None
			gc.collect()
			torch.cuda.empty_cache()
			export_merged(cfg.output_dir, Path(cfg.output_dir) / "merged", cfg.model)
