"""LoRA, quantization, and model loading configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
	"""LoRA adapter configuration.

	Controls the rank, scaling, target modules, and dropout for the low-rank adapter.
	Set ``use_lora=False`` to do full fine-tuning (not recommended on a single 4090).
	"""

	use_lora: bool = True
	r: int = 16
	lora_alpha: int = 32
	lora_dropout: float = 0.05
	bias: str = "none"
	target_modules: list[str] = field(
		default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
	)
	use_rslora: bool = False  # rank-stabilised LoRA


@dataclass
class QuantizationConfig:
	"""BitsAndBytes quantization configuration.

	Used when ``ModelConfig.use_unsloth=False`` and a quantized load is desired.
	Unsloth handles quantization internally when ``use_unsloth=True``.
	"""

	load_in_4bit: bool = True
	load_in_8bit: bool = False
	bnb_4bit_quant_type: str = "nf4"
	bnb_4bit_compute_dtype: str = "bfloat16"  # "float16" | "bfloat16" | "float32"
	bnb_4bit_use_double_quant: bool = True


@dataclass
class ModelConfig:
	"""HuggingFace model and tokenizer configuration.

	Attributes:
		model_id: HuggingFace model hub ID or local path.
		use_unsloth: Whether to load via Unsloth's FastLanguageModel (LLM only).
		    Set to False for VLMs or models not supported by Unsloth.
		max_seq_length: Maximum token sequence length used by Unsloth / tokenizer.
		dtype: Torch dtype string. ``None`` auto-detects (bfloat16 on Ampere+).
		lora: LoRA adapter configuration.
		quantization: BitsAndBytes quantization settings (used when use_unsloth=False).
		trust_remote_code: Passed to from_pretrained for custom architectures.
		attn_implementation: Attention kernel — "flash_attention_2" | "sdpa" | "eager".
	"""

	model_id: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
	use_unsloth: bool = True
	max_seq_length: int = 2048
	dtype: Optional[str] = None
	lora: LoRAConfig = field(default_factory=LoRAConfig)
	quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
	trust_remote_code: bool = False
	attn_implementation: str = "flash_attention_2"
