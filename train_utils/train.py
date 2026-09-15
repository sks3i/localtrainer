"""CLI entry point for train_utils.

Usage::

    # Via the installed script:
    train configs/sft_example.yaml

    # As a module:
    python -m train_utils.train configs/sft_example.yaml

    # From another project, passing an absolute path:
    train /path/to/my_project/configs/run.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from train_utils.configs.base_config import RunConfig
from train_utils.data.dpo_dataset import DPODataset
from train_utils.data.sft_dataset import SFTDataset
from train_utils.models.loader import load_model_and_tokenizer
from train_utils.models.lora import apply_lora
from train_utils.trainers.registry import build_trainer
from train_utils.utils.yaml_utils import load_yaml_config

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Maps trainer_type → dataset class to instantiate.
_DATASET_MAP = {
	"sft": SFTDataset,
	"dpo": DPODataset,
}


def _parse_args() -> argparse.Namespace:
	"""Parse CLI arguments.

	Returns:
		Parsed namespace with a ``config`` attribute holding the config path.
	"""
	parser = argparse.ArgumentParser(
		prog="train",
		description="train_utils — run SFT or RL fine-tuning from a YAML config.",
	)
	parser.add_argument(
		"config",
		type=Path,
		help="Path to a YAML run config file (e.g. configs/sft_example.yaml).",
	)
	parser.add_argument(
		"--loglevel",
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		help="Logging verbosity level.",
	)
	return parser.parse_args()


def _build_dataset(config: RunConfig, tokenizer) -> object:
	"""Instantiate the appropriate dataset class for the configured trainer type.

	Args:
		config: Resolved run config.
		tokenizer: Loaded tokenizer.

	Returns:
		An instance of :class:`SFTDataset` or :class:`DPODataset`.

	Raises:
		ValueError: If ``config.trainer_type`` has no registered dataset class.
	"""
	dataset_cls = _DATASET_MAP.get(config.trainer_type)
	if dataset_cls is None:
		available = ", ".join(_DATASET_MAP.keys())
		raise ValueError(
			f"No dataset class registered for trainer_type '{config.trainer_type}'. "
			f"Available: {available}"
		)
	return dataset_cls(config, tokenizer)


def run(config_path: str | Path) -> None:
	"""Execute a full training run from a YAML config path.

	This function is the programmatic equivalent of the CLI entry point, useful
	for calling train_utils from another project's training script.

	Args:
		config_path: Path to the YAML run config.

	Example::

		from train_utils.train import run
		run("configs/sft_example.yaml")
	"""
	config: RunConfig = load_yaml_config(config_path)
	logger.info("trainer_type=%s | model=%s | output_dir=%s", config.trainer_type, config.model.model_id, config.output_dir)

	model, tokenizer = load_model_and_tokenizer(config.model)
	model = apply_lora(model, config.model.lora, use_unsloth=config.model.use_unsloth)

	dataset = _build_dataset(config, tokenizer)
	trainer = build_trainer(config, model, tokenizer, dataset)
	trainer.train()


def main() -> None:
	"""CLI entry point registered in pyproject.toml as the ``train`` script."""
	args = _parse_args()
	logging.getLogger().setLevel(getattr(logging, args.loglevel))
	run(args.config)


if __name__ == "__main__":
	main()
