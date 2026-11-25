"""Freezing, LoRA and export for VLMs."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch

from train_utils.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
	"""(trainable, total) parameter counts. 4-bit weights are counted as stored elements x 2."""
	trainable = total = 0
	for p in model.parameters():
		n = p.numel() * (2 if p.dtype == torch.uint8 else 1)  # bnb packs two 4-bit values per byte
		total += n
		if p.requires_grad:
			trainable += n
	return trainable, total


def prepare_vlm_for_training(model: torch.nn.Module, cfg: ModelConfig) -> torch.nn.Module:
	"""Attach LoRA (restricted to the configured modules) or freeze modules for full fine-tuning."""
	lora = cfg.lora
	if lora.use_lora:
		from peft import LoraConfig, get_peft_model

		target = lora.target_modules_regex or lora.target_modules
		peft_cfg = LoraConfig(
			r=lora.r,
			lora_alpha=lora.lora_alpha,
			lora_dropout=lora.lora_dropout,
			bias=lora.bias,
			target_modules=target,
			modules_to_save=lora.modules_to_save or None,
			use_rslora=lora.use_rslora,
		)
		model = get_peft_model(model, peft_cfg)
	else:
		frozen = 0
		for name, p in model.named_parameters():
			if any(pat in name for pat in cfg.freeze_patterns):
				p.requires_grad_(False)
				frozen += p.numel()
		logger.info("Full fine-tuning; froze %.1fM params matching %s", frozen / 1e6, cfg.freeze_patterns)
	if cfg.upcast_trainable:
		n = 0
		for p in model.parameters():
			if p.requires_grad and p.dtype != torch.float32:
				p.data = p.data.float()
				n += p.numel()
		logger.info("Upcast %.1fM trainable params to fp32 (master weights)", n / 1e6)
	trainable, total = count_parameters(model)
	logger.info("Trainable params: %.1fM / %.1fM (%.2f%%)", trainable / 1e6, total / 1e6, 100 * trainable / max(total, 1))
	if trainable == 0:
		raise ValueError("No trainable parameters. Check lora.target_modules(_regex) / freeze_patterns.")
	return model


def export_merged(adapter_dir: str | Path, out_dir: str | Path, cfg: ModelConfig) -> Path:
	"""Merge a saved LoRA adapter into bf16 base weights and save a standalone model + processor.

	The base is reloaded unquantized: merging into 4-bit weights would round the merged delta to the
	4-bit grid, and the eval harness loads plain ``from_pretrained`` checkpoints anyway.
	"""
	from peft import PeftModel

	from train_utils.vlm.adapters import build_adapter

	adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
	base_cfg = ModelConfig(**{**cfg.__dict__})
	base_cfg.quantization = type(cfg.quantization)(load_in_4bit=False, load_in_8bit=False)
	adapter = build_adapter(base_cfg)
	model, processor = adapter.load()
	model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
	out_dir.mkdir(parents=True, exist_ok=True)
	model.save_pretrained(str(out_dir), safe_serialization=True)
	processor.save_pretrained(str(out_dir))
	if cfg.trust_remote_code or cfg.vlm_adapter == "molmo":
		_copy_remote_code(cfg.model_id, out_dir)
	logger.info("Merged model saved to %s", out_dir)
	return out_dir


def _copy_remote_code(model_id: str, out_dir: Path) -> None:
	"""Remote-code models need their *.py next to the weights to load again with trust_remote_code."""
	from huggingface_hub import snapshot_download

	src = Path(model_id) if Path(model_id).exists() else Path(snapshot_download(model_id, allow_patterns=["*.py"]))
	for f in src.glob("*.py"):
		if not (out_dir / f.name).exists():
			shutil.copy(f, out_dir / f.name)
	# save_pretrained can leave hub-qualified entries ("allenai/Molmo-7B-D-0924--modeling_molmo.X") in
	# auto_map; those load the class from the hub cache while the config loads from here, and
	# transformers refuses the mismatched classes. Point every entry at the local copy.
	for j in out_dir.glob("*config.json"):
		cfg = json.loads(j.read_text())
		amap = cfg.get("auto_map")
		if isinstance(amap, dict):
			cfg["auto_map"] = {k: (v.split("--", 1)[1] if isinstance(v, str) and "--" in v else v) for k, v in amap.items()}
			j.write_text(json.dumps(cfg, indent=2) + "\n")
