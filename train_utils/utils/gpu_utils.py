"""GPU introspection utilities: device setup, VRAM reporting, and dtype detection."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Minimum VRAM (bytes) required before issuing a low-VRAM warning.
_LOW_VRAM_THRESHOLD_GB: float = 4.0


def get_device() -> torch.device:
	"""Return the best available compute device.

	Prefers CUDA, then MPS (Apple Silicon), then CPU.

	Returns:
		The resolved :class:`torch.device`.
	"""
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")
	logger.info("Using device: %s", device)
	return device


def get_compute_dtype(prefer_bf16: bool = True) -> torch.dtype:
	"""Return the best floating-point dtype for the current GPU.

	Ampere and newer GPUs (including RTX 4090) support native bfloat16 via TF32.
	Older GPUs fall back to float16 when ``prefer_bf16=True`` is requested but
	unavailable.

	Args:
		prefer_bf16: When True, return bfloat16 if supported; float16 otherwise.
		    When False, always return float16.

	Returns:
		Recommended :class:`torch.dtype` for mixed-precision training.
	"""
	if not torch.cuda.is_available():
		return torch.float32

	if prefer_bf16 and torch.cuda.is_bf16_supported():
		logger.debug("bf16 supported — using torch.bfloat16")
		return torch.bfloat16

	logger.debug("bf16 not supported — using torch.float16")
	return torch.float16


def log_gpu_memory(label: str = "") -> None:
	"""Log current and peak VRAM usage for all visible CUDA devices.

	Intended to be called at key points (before/after model load, after
	trainer setup) to track memory consumption throughout a run.

	Args:
		label: Optional descriptive label prepended to the log line.
	"""
	if not torch.cuda.is_available():
		logger.info("[%s] No CUDA device — skipping VRAM log.", label)
		return

	for i in range(torch.cuda.device_count()):
		props = torch.cuda.get_device_properties(i)
		allocated = torch.cuda.memory_allocated(i) / 1024**3
		reserved = torch.cuda.memory_reserved(i) / 1024**3
		total = props.total_memory / 1024**3
		logger.info(
			"[%s] GPU %d (%s) | allocated=%.2fGB | reserved=%.2fGB | total=%.2fGB",
			label,
			i,
			props.name,
			allocated,
			reserved,
			total,
		)
		if total - reserved < _LOW_VRAM_THRESHOLD_GB:
			logger.warning(
				"[%s] GPU %d is running low on VRAM (%.2fGB free). "
				"Consider reducing batch size or enabling gradient checkpointing.",
				label,
				i,
				total - reserved,
			)


def log_gpu_info() -> None:
	"""Log static hardware properties for all visible CUDA devices.

	Logs device name, compute capability, total VRAM, and SM count.
	Useful at the start of a training run for reproducibility records.
	"""
	if not torch.cuda.is_available():
		logger.info("No CUDA devices found.")
		return

	for i in range(torch.cuda.device_count()):
		props = torch.cuda.get_device_properties(i)
		logger.info(
			"GPU %d: %s | compute_cap=%d.%d | VRAM=%.1fGB | SMs=%d",
			i,
			props.name,
			props.major,
			props.minor,
			props.total_memory / 1024**3,
			props.multi_processor_count,
		)
