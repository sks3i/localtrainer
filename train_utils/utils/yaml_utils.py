"""YAML configuration loader that maps raw dicts to typed RunConfig dataclasses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import dacite
import yaml

from train_utils.configs.base_config import RunConfig

logger = logging.getLogger(__name__)


def _load_raw_yaml(path: str | Path) -> dict[str, Any]:
	"""Read a YAML file and return the parsed dict.

	Args:
		path: Absolute or relative path to the YAML file.

	Returns:
		Parsed YAML contents as a plain dict.

	Raises:
		FileNotFoundError: If the YAML file does not exist.
	"""
	resolved = Path(path).resolve()
	if not resolved.exists():
		raise FileNotFoundError(f"Config file not found: {resolved}")
	with resolved.open("r") as fh:
		return yaml.safe_load(fh) or {}


def load_yaml_config(path: str | Path) -> RunConfig:
	"""Load a YAML config file and parse it into a typed :class:`RunConfig`.

	Nested keys are mapped to their corresponding dataclass fields via ``dacite``.
	Unknown or extra keys in the YAML are silently ignored to allow partial overrides.

	Args:
		path: Path to the YAML config file.

	Returns:
		A fully populated :class:`RunConfig` instance.

	Example::

		config = load_yaml_config("configs/sft_example.yaml")
		print(config.model.model_id)
	"""
	raw = _load_raw_yaml(path)
	logger.info("Loaded config from %s", path)
	config = dacite.from_dict(
		data_class=RunConfig,
		data=raw,
		config=dacite.Config(strict=False),
	)
	return config


def dump_yaml_config(config: RunConfig, path: str | Path) -> None:
	"""Serialise a :class:`RunConfig` back to a YAML file.

	Useful for saving the resolved config alongside a checkpoint so that runs
	are fully reproducible.

	Args:
		config: The RunConfig to serialise.
		path: Destination file path.
	"""
	import dataclasses

	resolved = Path(path)
	resolved.parent.mkdir(parents=True, exist_ok=True)
	with resolved.open("w") as fh:
		yaml.safe_dump(dataclasses.asdict(config), fh, default_flow_style=False, sort_keys=False)
	logger.info("Saved resolved config to %s", resolved)
