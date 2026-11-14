"""Root RunConfig dataclass that composes all sub-configs for a training run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from train_utils.configs.dpo_config import DPOConfig
from train_utils.configs.model_config import ModelConfig
from train_utils.configs.sft_config import SFTConfig
from train_utils.configs.vlm_sft_config import VLMSFTConfig

TrainerType = Literal["sft", "dpo", "vlm_sft"]


@dataclass
class TrainingArgs:
	"""Generic HuggingFace TrainingArguments overrides.

	Only the most commonly tuned hyperparameters are exposed here. All other
	TrainingArguments can be passed as ``extra_training_args``.

	Attributes:
		num_train_epochs: Number of full passes over the dataset.
		per_device_train_batch_size: Batch size per GPU.
		per_device_eval_batch_size: Eval batch size per GPU.
		gradient_accumulation_steps: Accumulate gradients before an optimizer step.
		learning_rate: Peak learning rate for the scheduler.
		lr_scheduler_type: Learning rate schedule — "cosine" | "linear" | "constant" etc.
		warmup_ratio: Fraction of total steps used for linear warmup.
		weight_decay: L2 regularisation coefficient.
		max_grad_norm: Gradient clipping norm.
		logging_steps: Log metrics every N optimizer steps.
		save_steps: Checkpoint every N optimizer steps.
		eval_steps: Evaluate every N optimizer steps (when eval dataset is provided).
		save_total_limit: Keep only the N most recent checkpoints.
		seed: Random seed for reproducibility.
		bf16: Use bfloat16 mixed precision (recommended for RTX 4090 / Ampere).
		fp16: Use float16 mixed precision. Mutually exclusive with bf16.
		gradient_checkpointing: Trade compute for VRAM by recomputing activations.
		torch_compile: Enable torch.compile for potential throughput gain.
		dataloader_num_workers: DataLoader worker processes.
		report_to: Tracker backend — "wandb" | "tensorboard" | "none".
		extra_training_args: Any additional kwargs forwarded verbatim to TrainingArguments.
	"""

	num_train_epochs: int = 3
	per_device_train_batch_size: int = 2
	per_device_eval_batch_size: int = 2
	gradient_accumulation_steps: int = 4
	learning_rate: float = 2e-4
	lr_scheduler_type: str = "cosine"
	warmup_ratio: float = 0.03
	weight_decay: float = 0.01
	max_grad_norm: float = 1.0
	logging_steps: int = 10
	save_steps: int = 100
	eval_steps: int = 100
	save_total_limit: int = 3
	seed: int = 42
	bf16: bool = True
	fp16: bool = False
	gradient_checkpointing: bool = True
	torch_compile: bool = False
	dataloader_num_workers: int = 4
	report_to: str = "wandb"
	extra_training_args: dict = field(default_factory=dict)


@dataclass
class RunConfig:
	"""Top-level config for a single training run.

	Attributes:
		trainer_type: Which training algorithm to use — "sft" | "dpo" | "vlm_sft".
		model: Model and adapter configuration.
		training: Generic training hyperparameters.
		sft: SFT-specific config. Required when trainer_type == "sft".
		dpo: DPO-specific config. Required when trainer_type == "dpo".
		vlm_sft: Image + text SFT config. Required when trainer_type == "vlm_sft".
		output_dir: Local directory where checkpoints and final weights are saved.
		run_name: Human-readable identifier for the run (used in W&B and checkpoint paths).
		wandb_project: W&B project name. Ignored when training.report_to != "wandb".
		wandb_entity: W&B entity (team or user). ``None`` uses the default entity.
	"""

	trainer_type: TrainerType = "sft"
	model: ModelConfig = field(default_factory=ModelConfig)
	training: TrainingArgs = field(default_factory=TrainingArgs)
	sft: Optional[SFTConfig] = field(default_factory=SFTConfig)
	dpo: Optional[DPOConfig] = None
	vlm_sft: Optional[VLMSFTConfig] = None
	output_dir: str = "outputs/"
	run_name: str = "run"
	wandb_project: Optional[str] = "train_utils"
	wandb_entity: Optional[str] = None
