"""Model and tokenizer loading with Unsloth and standard HuggingFace paths."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase

from train_utils.configs.model_config import ModelConfig, QuantizationConfig

logger = logging.getLogger(__name__)

ModelTokenizerPair = tuple[PreTrainedModel, PreTrainedTokenizerBase]


def _build_bnb_config(quant: QuantizationConfig) -> BitsAndBytesConfig:
	"""Construct a BitsAndBytesConfig from a QuantizationConfig dataclass.

	Args:
		quant: Quantization settings.

	Returns:
		Configured :class:`BitsAndBytesConfig` instance.
	"""
	compute_dtype = getattr(torch, quant.bnb_4bit_compute_dtype)
	return BitsAndBytesConfig(
		load_in_4bit=quant.load_in_4bit,
		load_in_8bit=quant.load_in_8bit,
		bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
		bnb_4bit_compute_dtype=compute_dtype,
		bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
	)


def _load_via_unsloth(cfg: ModelConfig) -> ModelTokenizerPair:
	"""Load an LLM using Unsloth's FastLanguageModel for optimised LoRA training.

	Unsloth applies custom CUDA kernels and 2x faster training automatically.
	Requires the ``unsloth`` package to be installed.

	Args:
		cfg: Model configuration.

	Returns:
		Tuple of (model, tokenizer).

	Raises:
		ImportError: If the ``unsloth`` package is not installed.
	"""
	try:
		from unsloth import FastLanguageModel  # type: ignore[import]
	except ImportError as exc:
		raise ImportError(
			"unsloth is not installed. Install it with: "
			"pip install 'unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git' "
			"or set model.use_unsloth=false in your config."
		) from exc

	dtype = getattr(torch, cfg.dtype) if cfg.dtype else None
	logger.info("Loading %s via Unsloth (max_seq_length=%d)", cfg.model_id, cfg.max_seq_length)

	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name=cfg.model_id,
		max_seq_length=cfg.max_seq_length,
		dtype=dtype,
		load_in_4bit=cfg.quantization.load_in_4bit,
		trust_remote_code=cfg.trust_remote_code,
	)
	return model, tokenizer


def _load_via_hf(cfg: ModelConfig) -> ModelTokenizerPair:
	"""Load a model with standard HuggingFace transformers.

	Used for VLMs or models not supported by Unsloth. Applies BitsAndBytes
	quantization when ``cfg.quantization.load_in_4bit`` or ``load_in_8bit`` is True.

	Args:
		cfg: Model configuration.

	Returns:
		Tuple of (model, tokenizer).
	"""
	logger.info("Loading %s via HuggingFace transformers", cfg.model_id)
	quant = cfg.quantization
	bnb_config = _build_bnb_config(quant) if (quant.load_in_4bit or quant.load_in_8bit) else None
	dtype = getattr(torch, cfg.dtype) if cfg.dtype else torch.bfloat16

	model_kwargs: dict[str, Any] = {
		"trust_remote_code": cfg.trust_remote_code,
		"torch_dtype": dtype,
		"device_map": "auto",
		"attn_implementation": cfg.attn_implementation,
	}
	if bnb_config is not None:
		model_kwargs["quantization_config"] = bnb_config

	model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
	tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
	return model, tokenizer


def _ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
	"""Add a pad token if the tokenizer does not define one.

	Many chat-tuned models (e.g. LLaMA) omit a pad token. Setting it to
	``eos_token`` is the standard workaround for batched training.

	Args:
		tokenizer: Tokenizer to patch in-place.
	"""
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id
		logger.debug("pad_token not set — using eos_token (%s) as pad_token.", tokenizer.eos_token)


def load_model_and_tokenizer(cfg: ModelConfig) -> ModelTokenizerPair:
	"""Load a model and its tokenizer according to the provided :class:`ModelConfig`.

	Dispatches to Unsloth when ``cfg.use_unsloth=True``, otherwise uses the
	standard HuggingFace path with optional BitsAndBytes quantization.

	Args:
		cfg: Model and quantization configuration.

	Returns:
		Tuple of (model, tokenizer) ready for training.

	Example::

		model, tokenizer = load_model_and_tokenizer(config.model)
	"""
	if cfg.use_unsloth:
		model, tokenizer = _load_via_unsloth(cfg)
	else:
		model, tokenizer = _load_via_hf(cfg)

	_ensure_pad_token(tokenizer)
	logger.info("Model loaded: %s | params: %.2fB", cfg.model_id, model.num_parameters() / 1e9)
	return model, tokenizer
