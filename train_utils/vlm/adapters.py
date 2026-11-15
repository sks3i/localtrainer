"""VLM adapters: load a vision-language model and turn (image, prompt, answer) examples into batches.

Every VLM family has its own processor API and its own image tensors (Qwen-VL flattens patches and
passes ``image_grid_thw``, LFM2-VL pads tiles and passes ``spatial_shapes``, Molmo passes crops plus
``image_input_idx``). An adapter hides that behind one interface so the trainer stays generic:

	model, processor = adapter.load(model_cfg)
	batch = adapter.collate(examples)   # dict of tensors incl. "labels"

Labels follow the HF convention: same shape as ``input_ids``, NOT shifted (the loss shifts them),
and ``-100`` wherever no loss should be taken. Only the answer and the end-of-turn token that
closes it are supervised:

	[system/user turn][image tokens ...][prompt][assistant header] | answer tokens | <end> \\n [pad]
	-100 ...............................................................  ids ....... id   -100 -100

Why mask the prompt: the prompt is given at test time, so its likelihood is not something we need
to learn. Training on it spends gradient on modelling user text (and the fixed template), and with
~300 image + prompt tokens vs ~15 answer tokens it would dominate the loss.
Why keep the end token: the model has to learn to STOP after one point. Without it, generation
runs on to max_new_tokens.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from train_utils.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _dtype(cfg: ModelConfig) -> torch.dtype:
	return getattr(torch, cfg.dtype) if cfg.dtype else torch.bfloat16


def _bnb_config(cfg: ModelConfig):
	"""BitsAndBytes config, or None when the model is loaded unquantized."""
	q = cfg.quantization
	if not (q.load_in_4bit or q.load_in_8bit):
		return None
	from transformers import BitsAndBytesConfig

	return BitsAndBytesConfig(
		load_in_4bit=q.load_in_4bit,
		load_in_8bit=q.load_in_8bit,
		bnb_4bit_quant_type=q.bnb_4bit_quant_type,
		bnb_4bit_compute_dtype=getattr(torch, q.bnb_4bit_compute_dtype),
		bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
		llm_int8_skip_modules=q.skip_modules or None,
	)


def answer_labels(input_ids: list[int], answer_ids: list[int], end_id: int | None) -> list[int]:
	"""Labels that supervise only the trailing answer span of ``input_ids``.

	``answer_ids`` is the tokenization of everything after the generation prompt (answer, end token,
	and whatever the chat template appends after it, e.g. "\\n"). It must be an exact suffix of
	``input_ids``; this is checked rather than assumed, because BPE can merge tokens across the
	prompt/answer boundary and then the span would be silently off by one.

	Tokens after the last ``end_id`` in the span are masked: they come after the model would have
	stopped, so they are never generated.
	"""
	n = len(answer_ids)
	if n == 0 or input_ids[-n:] != answer_ids:
		raise ValueError(
			"answer tokens are not a suffix of the full sequence; the prompt/answer boundary "
			"tokenizes differently in context. Inspect the chat template for this model."
		)
	keep = n
	if end_id is not None and end_id in answer_ids:
		keep = n - answer_ids[::-1].index(end_id)  # up to and including the last end token
	labels = [IGNORE_INDEX] * (len(input_ids) - n) + answer_ids[:keep] + [IGNORE_INDEX] * (n - keep)
	return labels


class VLMAdapter(ABC):
	"""Interface between a VLM family and the generic trainer."""

	def __init__(self, cfg: ModelConfig, max_image_side: int | None = None, image_root: str | None = None) -> None:
		self.cfg = cfg
		self.max_image_side = max_image_side
		self.image_root = Path(image_root).expanduser() if image_root else None
		self.processor: Any = None

	@abstractmethod
	def load(self) -> tuple[torch.nn.Module, Any]:
		"""Load and return (model, processor)."""

	@abstractmethod
	def collate(self, examples: list[dict]) -> dict[str, torch.Tensor]:
		"""Batch examples (dicts with image/prompt/answer) into model inputs plus ``labels``."""

	@abstractmethod
	def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
		"""Turn on activation checkpointing in the way this model supports."""

	@abstractmethod
	def generate(self, model: torch.nn.Module, image, prompt: str, max_new_tokens: int = 64) -> str:
		"""Greedy-decode an answer for one example (for in-training task metrics)."""

	def load_image(self, path_or_image):
		"""Open an image as RGB, resolving relative paths, and cap its longest side."""
		from PIL import Image

		img = path_or_image
		if not isinstance(img, Image.Image):
			p = Path(img)
			if not p.is_absolute() and self.image_root is not None:
				p = self.image_root / p
			img = Image.open(p)
		img = img.convert("RGB")
		if self.max_image_side and max(img.size) > self.max_image_side:
			s = self.max_image_side / max(img.size)
			img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
		return img


class HFChatVLMAdapter(VLMAdapter):
	"""Any model that loads with ``AutoModelForImageTextToText`` and has a chat template.

	Covers Qwen2/2.5/3-VL, LFM2-VL, SmolVLM, Idefics3, Gemma-3 and similar. The full batch goes
	through the processor in one call, so each model's own image tensors (and image token expansion)
	are produced by its processor, not re-implemented here.
	"""

	def load(self):
		from transformers import AutoModelForImageTextToText, AutoProcessor

		cfg = self.cfg
		self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code, **cfg.processor_kwargs)
		self.processor.tokenizer.padding_side = "right"
		kwargs: dict[str, Any] = {
			"dtype": _dtype(cfg),
			"device_map": "cuda",
			"attn_implementation": cfg.attn_implementation,
			"trust_remote_code": cfg.trust_remote_code,
		}
		bnb = _bnb_config(cfg)
		if bnb is not None:
			kwargs["quantization_config"] = bnb
		model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, **kwargs)
		tok = self.processor.tokenizer
		self.end_id = tok.convert_tokens_to_ids(tok.eos_token) if tok.eos_token else None
		return model, self.processor

	def _messages(self, prompt: str, answer: str | None) -> list[dict]:
		msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
		if answer is not None:
			msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
		return msgs

	def collate(self, examples):
		proc, tok = self.processor, self.processor.tokenizer
		texts, suffixes, images = [], [], []
		for ex in examples:
			prompt_text = proc.apply_chat_template(self._messages(ex["prompt"], None), add_generation_prompt=True, tokenize=False)
			full_text = proc.apply_chat_template(self._messages(ex["prompt"], ex["answer"]), tokenize=False)
			if not full_text.startswith(prompt_text):
				raise ValueError("chat template output with the answer does not start with the generation prompt")
			texts.append(full_text)
			suffixes.append(tok.encode(full_text[len(prompt_text) :], add_special_tokens=False))
			images.append([self.load_image(ex["image"])])
		batch = proc(text=texts, images=images, return_tensors="pt", padding=True)
		labels = torch.full_like(batch["input_ids"], IGNORE_INDEX)
		for i, suffix in enumerate(suffixes):
			n = int(batch["attention_mask"][i].sum())  # right padding: real tokens are [0, n)
			ids = batch["input_ids"][i, :n].tolist()
			labels[i, :n] = torch.tensor(answer_labels(ids, suffix, self.end_id))
		batch["labels"] = labels
		return dict(batch)

	@torch.inference_mode()
	def generate(self, model, image, prompt, max_new_tokens=64):
		messages = [{"role": "user", "content": [{"type": "image", "image": self.load_image(image)}, {"type": "text", "text": prompt}]}]
		inputs = self.processor.apply_chat_template(
			messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
		).to(model.device)
		out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
		return self.processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

	def enable_gradient_checkpointing(self, model):
		# Non-reentrant checkpointing does not need the checkpointed block's inputs to require grad,
		# so it works with frozen embeddings (LoRA) without enable_input_require_grads().
		model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
		model.config.use_cache = False

	# --- RL (vlm_grpo): batched sampling, then scoring the sampled tokens ---------------------

	def _prompt_batch(self, examples: list[dict], padding_side: str):
		proc, tok = self.processor, self.processor.tokenizer
		texts = [proc.apply_chat_template(self._messages(ex["prompt"], None), add_generation_prompt=True, tokenize=False) for ex in examples]
		images = [[self.load_image(ex["image"])] for ex in examples]
		old = tok.padding_side
		tok.padding_side = padding_side
		try:
			return proc(text=texts, images=images, return_tensors="pt", padding=True)
		except ValueError as e:
			# Seen once after ~1k good batches: the fast tokenizer returned unpadded rows ("expected
			# sequence of length 439 (got 440)") and the same batch then worked on retry. Fall back to
			# one processor call per example and pad here.
			logger.warning("batched processor call failed (%s); padding per example", e)
			return self._pad_rows([proc(text=[t], images=[im], return_tensors="pt") for t, im in zip(texts, images)], padding_side)
		finally:
			tok.padding_side = old

	def _pad_rows(self, rows: list, padding_side: str):
		"""Merge single-example processor outputs: per-token tensors (1, L, ...) are padded to the
		longest L (input_ids with the pad id, everything else with 0); image tensors are concatenated."""
		from transformers import BatchFeature

		pad_id = self.processor.tokenizer.pad_token_id
		L = max(r["input_ids"].shape[1] for r in rows)
		out = {}
		for k in rows[0].keys():
			vs = [r[k] for r in rows]
			if torch.is_tensor(vs[0]) and vs[0].dim() >= 2 and vs[0].shape[:2] == rows[0]["input_ids"].shape:
				padded = []
				for r, v in zip(rows, vs):
					n = L - v.shape[1]
					fill = v.new_full((1, n, *v.shape[2:]), pad_id if k == "input_ids" else 0)
					padded.append(torch.cat([fill, v] if padding_side == "left" else [v, fill], dim=1))
				out[k] = torch.cat(padded, dim=0)
			elif torch.is_tensor(vs[0]):
				out[k] = torch.cat(vs, dim=0)
			else:
				out[k] = vs
		return BatchFeature(out)

	@torch.no_grad()
	def sample(self, model, examples: list[dict], max_new_tokens: int, do_sample: bool = True, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> list[list[int]]:
		"""One completion per example, as token ids up to and including the first end token.

		Batched generation needs LEFT padding: every row's next token is then predicted at the same
		last position. Every sampling knob is passed explicitly, because generate() otherwise falls
		back to the checkpoint's generation_config (Qwen3-VL ships top_k=20, top_p=0.8, T=0.7), and
		the trainer's log-probs would then refer to a different distribution than the one sampled.
		"""
		batch = self._prompt_batch(examples, "left").to(model.device)
		kw = {"do_sample": do_sample}
		if do_sample:
			kw.update(temperature=temperature, top_p=top_p, top_k=top_k)
		else:
			kw.update(temperature=None, top_p=None, top_k=None)
		tok = self.processor.tokenizer
		out = model.generate(**batch, max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=tok.pad_token_id, **kw)
		new = out[:, batch["input_ids"].shape[1] :].tolist()
		ends = set(self._end_ids(model))
		completions = []
		for row in new:
			cut = next((i + 1 for i, t in enumerate(row) if t in ends), len(row))
			completions.append(row[:cut])
		return completions

	def _end_ids(self, model) -> list[int]:
		eos = model.generation_config.eos_token_id
		ids = list(eos) if isinstance(eos, (list, tuple)) else ([eos] if eos is not None else [])
		return ids or ([self.end_id] if self.end_id is not None else [])

	def decode(self, ids: list[int]) -> str:
		return self.processor.decode(ids, skip_special_tokens=True)

	def policy_inputs(self, examples: list[dict], completions: list[list[int]]) -> dict[str, torch.Tensor]:
		"""Model inputs for prompt + sampled completion, plus ``completion_mask``.

		The completion is appended as token ids, not re-tokenized text: decode -> encode does not
		round-trip exactly with BPE, and the log-probs must be of the tokens that were sampled.
		Image tensors are unchanged by appending text after the prompt, so they come straight from the
		processor. ``completion_mask[b, t]`` is 1 where input_ids[b, t] is a sampled token, so the
		log-prob of that token is read from the logits at t - 1.
		"""
		pb = self._prompt_batch(examples, "right")
		pad = self.processor.tokenizer.pad_token_id
		lens = pb["attention_mask"].sum(1).tolist()
		T = max(n + len(c) for n, c in zip(lens, completions))
		B = len(examples)
		ids = torch.full((B, T), pad, dtype=pb["input_ids"].dtype)
		attn = torch.zeros((B, T), dtype=pb["attention_mask"].dtype)
		cmask = torch.zeros((B, T), dtype=torch.bool)
		for b, (n, c) in enumerate(zip(lens, completions)):
			ids[b, :n] = pb["input_ids"][b, :n]
			ids[b, n : n + len(c)] = torch.tensor(c, dtype=ids.dtype)
			attn[b, : n + len(c)] = 1
			cmask[b, n : n + len(c)] = True
		from transformers import BatchFeature

		batch = {}
		for k, v in pb.items():
			if k in ("input_ids", "attention_mask"):
				continue
			if torch.is_tensor(v) and v.shape[:2] == pb["input_ids"].shape:
				# Per-token side inputs (e.g. Qwen3-VL's mm_token_type_ids): completion tokens are text, 0.
				ext = v.new_zeros((B, T, *v.shape[2:]))
				for b, n in enumerate(lens):
					ext[b, :n] = v[b, :n]
				v = ext
			batch[k] = v
		batch.update(input_ids=ids, attention_mask=attn, completion_mask=cmask)
		return BatchFeature(batch)


class MolmoAdapter(VLMAdapter):
	"""allenai/Molmo (remote code). Needs the transformers version its remote code was written for.

	Molmo's processor builds ``" User: {prompt} Assistant:"`` plus image crops, and its batching
	convention pads with -1 (input ids, crops, crop masks, patch indices); the model derives the
	attention mask from ``input_ids != -1``. The answer is appended as ``" " + answer`` followed by EOS,
	matching what Molmo generates.
	"""

	def load(self):
		from transformers import AutoModelForCausalLM, AutoProcessor

		cfg = self.cfg
		self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True, torch_dtype="auto", **cfg.processor_kwargs)
		kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": _dtype(cfg), "device_map": "cuda"}
		bnb = _bnb_config(cfg)
		if bnb is not None:
			kwargs["quantization_config"] = bnb
		model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
		self.dtype = _dtype(cfg)
		return model, self.processor

	def collate(self, examples):
		tok = self.processor.tokenizer
		eos = tok.eos_token_id
		rows = []
		for ex in examples:
			out = self.processor.process(images=[self.load_image(ex["image"])], text=ex["prompt"])
			answer_ids = tok.encode(" " + ex["answer"], add_special_tokens=False) + [eos]
			ids = out["input_ids"].tolist() + answer_ids
			out["input_ids"] = torch.tensor(ids)
			out["labels"] = torch.tensor(answer_labels(ids, answer_ids, eos))
			rows.append(out)
		batch = {}
		for key, pad in (("input_ids", -1), ("labels", IGNORE_INDEX), ("images", -1), ("image_masks", -1), ("image_input_idx", -1)):
			if key not in rows[0]:
				continue
			batch[key] = _pad_stack([r[key] for r in rows], pad)
		batch["images"] = batch["images"].to(self.dtype)
		return batch

	@torch.inference_mode()
	def generate(self, model, image, prompt, max_new_tokens=64):
		from transformers import GenerationConfig

		inputs = self.processor.process(images=[self.load_image(image)], text=prompt)
		inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
		inputs["images"] = inputs["images"].to(self.dtype)
		base = model.get_base_model() if hasattr(model, "get_base_model") else model
		cfg = GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>")
		out = base.generate_from_batch(inputs, cfg, tokenizer=self.processor.tokenizer)
		return self.processor.tokenizer.decode(out[0, inputs["input_ids"].size(1) :], skip_special_tokens=True)

	def enable_gradient_checkpointing(self, model):
		# Molmo's blocks call self._activation_checkpoint_fn around attention and MLP when it is set;
		# it has no HF gradient_checkpointing_enable().
		from functools import partial

		from torch.utils.checkpoint import checkpoint

		fn = partial(checkpoint, use_reentrant=False)
		base = model.get_base_model() if hasattr(model, "get_base_model") else model
		n = 0
		for module in base.modules():
			if hasattr(module, "_activation_checkpoint_fn"):
				module._activation_checkpoint_fn = fn
				n += 1
		logger.info("Molmo: activation checkpointing on %d blocks", n)


def _pad_stack(tensors: list[torch.Tensor], pad_value) -> torch.Tensor:
	"""Stack along a new batch dim, right-padding dim 0 of each tensor to the longest."""
	n = max(t.shape[0] for t in tensors)
	out = tensors[0].new_full((len(tensors), n, *tensors[0].shape[1:]), pad_value)
	for i, t in enumerate(tensors):
		out[i, : t.shape[0]] = t
	return out


VLM_ADAPTERS: dict[str, type[VLMAdapter]] = {
	"hf_chat": HFChatVLMAdapter,
	"molmo": MolmoAdapter,
}


def build_adapter(cfg: ModelConfig, max_image_side: int | None = None, image_root: str | None = None) -> VLMAdapter:
	if cfg.vlm_adapter not in VLM_ADAPTERS:
		raise ValueError(f"Unknown vlm_adapter '{cfg.vlm_adapter}'. Available: {', '.join(VLM_ADAPTERS)}")
	return VLM_ADAPTERS[cfg.vlm_adapter](cfg, max_image_side=max_image_side, image_root=image_root)
