"""Abstract base trainer defining the interface for all training algorithms."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from train_utils.configs.base_config import RunConfig
from train_utils.data.base_dataset import BaseDataset
from train_utils.utils.gpu_utils import log_gpu_info, log_gpu_memory
from train_utils.utils.yaml_utils import dump_yaml_config

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
	"""Abstract base for all train_utils trainers.

	Subclasses implement :meth:`_build_hf_trainer` to return a configured TRL
	or HuggingFace Trainer, then inherit :meth:`train` which handles the
	common setup (W&B env vars, VRAM logging, config snapshot).

	The expected lifecycle is::

		trainer = SFTTrainer(config, model, tokenizer, dataset)
		trainer.train()
	"""

	def __init__(
		self,
		config: RunConfig,
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizerBase,
		dataset: BaseDataset,
	) -> None:
		"""Store references to config, model, tokenizer, and dataset.

		Args:
			config: Full resolved run configuration.
			model: Pretrained model (with LoRA adapters if applicable).
			tokenizer: Tokenizer matching the model.
			dataset: Dataset wrapper exposing get_train_dataset / get_eval_dataset.
		"""
		self.config = config
		self.model = model
		self.tokenizer = tokenizer
		self.dataset = dataset

	# ------------------------------------------------------------------
	# Abstract interface
	# ------------------------------------------------------------------

	@abstractmethod
	def _build_hf_trainer(self) -> object:
		"""Construct and return the underlying HuggingFace / TRL trainer.

		Subclasses configure and return an instance of TRL's SFTTrainer,
		DPOTrainer, or any other compatible Trainer.

		Returns:
			A trainer instance with a ``.train()`` method.
		"""
		...

	# ------------------------------------------------------------------
	# Common helpers
	# ------------------------------------------------------------------

	def _setup_wandb(self) -> None:
		"""Set W&B environment variables from the run config.

		Only applied when ``training.report_to`` contains "wandb".
		"""
		if "wandb" not in self.config.training.report_to:
			return
		if self.config.wandb_project:
			os.environ.setdefault("WANDB_PROJECT", self.config.wandb_project)
		if self.config.wandb_entity:
			os.environ.setdefault("WANDB_ENTITY", self.config.wandb_entity)
		os.environ.setdefault("WANDB_RUN_NAME", self.config.run_name)
		logger.info("W&B project=%s | run=%s", self.config.wandb_project, self.config.run_name)

	def _save_config_snapshot(self) -> None:
		"""Persist the resolved RunConfig as YAML alongside the checkpoints.

		Writes ``<output_dir>/run_config.yaml`` so any run is self-documenting.
		"""
		snapshot_path = Path(self.config.output_dir) / "run_config.yaml"
		dump_yaml_config(self.config, snapshot_path)

	def _make_output_dir(self) -> None:
		"""Create the output directory if it does not exist."""
		Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

	# ------------------------------------------------------------------
	# Public entry point
	# ------------------------------------------------------------------

	def train(self) -> None:
		"""Run the full training lifecycle.

		Steps:
		    1. Log GPU hardware info.
		    2. Configure W&B environment.
		    3. Create output directory and save config snapshot.
		    4. Log pre-training VRAM.
		    5. Build the underlying HF/TRL trainer.
		    6. Start training.
		    7. Log post-training VRAM.
		    8. Persist the final model / adapter.
		"""
		log_gpu_info()
		self._setup_wandb()
		self._make_output_dir()
		self._save_config_snapshot()
		log_gpu_memory("before_train")

		logger.info("Building trainer for run_name=%s", self.config.run_name)
		hf_trainer = self._build_hf_trainer()

		logger.info("Starting training...")
		hf_trainer.train()

		log_gpu_memory("after_train")
		self._save_final(hf_trainer)
		logger.info("Training complete. Outputs saved to %s", self.config.output_dir)

	def _save_final(self, hf_trainer: object) -> None:
		"""Save the final model using the HF trainer's built-in save method.

		Subclasses may override this to call save_lora_adapter instead.

		Args:
			hf_trainer: The completed HF / TRL trainer instance.
		"""
		hf_trainer.save_model(self.config.output_dir)  # type: ignore[union-attr]
		self.tokenizer.save_pretrained(self.config.output_dir)
