"""LoRA adapter application and persistence utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from train_utils.configs.model_config import LoRAConfig

logger = logging.getLogger(__name__)


def apply_lora(model: PreTrainedModel, lora_cfg: LoRAConfig, use_unsloth: bool = True) -> PreTrainedModel:
	"""Attach a LoRA adapter to the model.

	Dispatches to Unsloth's ``FastLanguageModel.get_peft_model`` when
	``use_unsloth=True`` (recommended — it applies custom fused kernels). Falls
	back to ``peft.get_peft_model`` for VLMs or models loaded via standard HF.

	Args:
		model: The base pretrained model to attach LoRA to.
		lora_cfg: LoRA hyperparameter configuration.
		use_unsloth: Whether the model was loaded via Unsloth.

	Returns:
		The model with LoRA adapters attached and non-adapter weights frozen.
	"""
	if not lora_cfg.use_lora:
		logger.info("LoRA disabled — training all parameters.")
		return model

	if use_unsloth:
		return _apply_unsloth_lora(model, lora_cfg)
	return _apply_peft_lora(model, lora_cfg)


def _apply_unsloth_lora(model: PreTrainedModel, lora_cfg: LoRAConfig) -> PreTrainedModel:
	"""Apply LoRA via Unsloth's FastLanguageModel.get_peft_model.

	Args:
		model: Base model loaded by Unsloth.
		lora_cfg: LoRA configuration.

	Returns:
		PEFT-wrapped model with Unsloth fused kernels.
	"""
	from unsloth import FastLanguageModel  # type: ignore[import]

	logger.info("Applying Unsloth LoRA (r=%d, alpha=%d)", lora_cfg.r, lora_cfg.lora_alpha)
	model = FastLanguageModel.get_peft_model(
		model,
		r=lora_cfg.r,
		target_modules=lora_cfg.target_modules,
		lora_alpha=lora_cfg.lora_alpha,
		lora_dropout=lora_cfg.lora_dropout,
		bias=lora_cfg.bias,
		use_gradient_checkpointing="unsloth",
		use_rslora=lora_cfg.use_rslora,
		random_state=42,
	)
	return model


def _apply_peft_lora(model: PreTrainedModel, lora_cfg: LoRAConfig) -> PreTrainedModel:
	"""Apply LoRA via the standard PEFT library.

	Args:
		model: Base model (HF standard load).
		lora_cfg: LoRA configuration.

	Returns:
		PEFT-wrapped model with frozen base weights.
	"""
	from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import]

	logger.info("Applying PEFT LoRA (r=%d, alpha=%d)", lora_cfg.r, lora_cfg.lora_alpha)
	peft_config = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		r=lora_cfg.r,
		lora_alpha=lora_cfg.lora_alpha,
		lora_dropout=lora_cfg.lora_dropout,
		bias=lora_cfg.bias,
		target_modules=lora_cfg.target_modules,
		use_rslora=lora_cfg.use_rslora,
	)
	model = get_peft_model(model, peft_config)
	model.print_trainable_parameters()
	return model


def save_lora_adapter(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, output_dir: str | Path) -> None:
	"""Save only the LoRA adapter weights and tokenizer to disk.

	Saves a fraction of the full model size — only the adapter delta tensors.
	The saved adapter can be loaded later with ``PeftModel.from_pretrained``.

	Args:
		model: PEFT-wrapped model (must have ``save_pretrained`` from PEFT).
		tokenizer: Tokenizer to save alongside the adapter.
		output_dir: Directory to write adapter weights and config to.
	"""
	out = Path(output_dir)
	out.mkdir(parents=True, exist_ok=True)
	model.save_pretrained(str(out))
	tokenizer.save_pretrained(str(out))
	logger.info("LoRA adapter saved to %s", out)
