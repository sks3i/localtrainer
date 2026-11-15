"""VLM GRPO trainer: sample G completions per prompt, score them, and push up the better ones.

A plain PyTorch loop rather than HF Trainer: the batch for each step does not exist until the
current policy has generated it, and one step mixes generation (eval mode, KV cache) with
training passes (train mode, gradient checkpointing).

Memory on one GPU: the reference policy is the same weights with the LoRA adapter disabled, so
pi_ref costs one extra no-grad forward pass, not a second copy of the model.
"""

from __future__ import annotations

import gc
import importlib
import json
import logging
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from train_utils.configs.base_config import RunConfig
from train_utils.configs.vlm_grpo_config import VLMGRPOConfig
from train_utils.data.base_dataset import BaseDataset
from train_utils.grpo.math import aggregate, group_advantages, grpo_token_loss, zero_std_mask
from train_utils.trainers.base_trainer import BaseTrainer
from train_utils.utils.gpu_utils import log_gpu_memory
from train_utils.vlm.adapters import VLMAdapter
from train_utils.vlm.setup import export_merged

logger = logging.getLogger(__name__)


def load_reward_fn(spec: str):
	module, _, name = spec.partition(":")
	if not name:
		raise ValueError(f"reward_fn must look like 'package.module:function', got {spec!r}")
	return getattr(importlib.import_module(module), name)


def sampled_token_logprobs(model, batch: dict, temperature: float = 1.0, with_entropy: bool = False):
	"""Log-probs (B, T) of the tokens in ``input_ids`` at ``completion_mask`` positions (0 elsewhere).

	Token t is predicted by the logits at t - 1. Only completion positions are upcast to fp32 and
	pushed through log_softmax: the ~400 image + prompt positions are never scored, and a (B, T, V)
	fp32 tensor at V = 151k would be several GB.
	"""
	inputs = {k: v for k, v in batch.items() if k != "completion_mask"}
	logits = model(**inputs).logits
	cmask = batch["completion_mask"]
	sel = cmask[:, 1:]
	picked = logits[:, :-1][sel].float() / temperature  # (N, V), N = completion tokens in the batch
	logp_all = F.log_softmax(picked, dim=-1)
	tok = batch["input_ids"][:, 1:][sel]
	out = torch.zeros(cmask.shape, dtype=torch.float32, device=logits.device)
	out[:, 1:][sel] = logp_all.gather(1, tok.unsqueeze(1)).squeeze(1)
	ent = None
	if with_entropy:
		ent = -(logp_all.exp() * logp_all).sum(-1).detach()
	return out, ent


class VLMGRPOTrainer(BaseTrainer):
	"""GRPO on {image, prompt, ...} rows with a programmatic reward.

	``model`` must carry a LoRA adapter (peft): the reference policy is ``model.disable_adapter()``.
	"""

	def __init__(self, config: RunConfig, model, tokenizer, dataset: BaseDataset, adapter: VLMAdapter | None = None, callbacks: list | None = None) -> None:
		super().__init__(config, model, tokenizer, dataset)
		if config.vlm_grpo is None:
			raise ValueError("RunConfig.vlm_grpo must be set for VLMGRPOTrainer.")
		if adapter is None:
			raise ValueError("VLMGRPOTrainer needs the VLMAdapter that loaded the model.")
		if not config.model.lora.use_lora:
			raise ValueError("vlm_grpo uses the LoRA-disabled model as the KL reference; set model.lora.use_lora.")
		self.cfg: VLMGRPOConfig = config.vlm_grpo
		if self.cfg.num_generations < 2:
			raise ValueError("num_generations must be >= 2: the group mean is the baseline.")
		self.adapter = adapter
		self.reward_fn = load_reward_fn(self.cfg.reward_fn)
		self.out = Path(config.output_dir)
		self.metrics_file = self.out / "metrics.jsonl"
		self.dev_file = self.out / "dev_metrics.jsonl"
		self.samples_file = self.out / "samples.jsonl"

	def _build_hf_trainer(self):  # BaseTrainer interface; this trainer runs its own loop
		return self

	# --- pieces of one step ----------------------------------------------------------------

	def _rollout(self, rows: list[dict]) -> tuple[list[dict], list[list[int]], list[str]]:
		"""G samples per row; returns rows repeated G times (group-major), completions and texts."""
		c = self.cfg
		expanded = [r for r in rows for _ in range(c.num_generations)]
		self.model.eval()
		comps: list[list[int]] = []
		for i in range(0, len(expanded), c.generation_batch_size):
			comps += self.adapter.sample(self.model, expanded[i : i + c.generation_batch_size], c.max_new_tokens, True, c.temperature, c.top_p, c.top_k)
		texts = [self.adapter.decode(x) for x in comps]
		return expanded, comps, texts

	def _logprobs(self, rows, comps, idx, reference: bool = False) -> list[torch.Tensor]:
		"""No-grad log-probs for completions ``idx``, under the current policy or the reference."""
		c = self.cfg
		out = []
		self.model.eval()
		with torch.no_grad():
			for i in range(0, len(idx), c.micro_batch_size):
				mb = idx[i : i + c.micro_batch_size]
				batch = self.adapter.policy_inputs([rows[j] for j in mb], [comps[j] for j in mb]).to(self.model.device)
				if reference:
					with self.model.disable_adapter():
						lp, _ = sampled_token_logprobs(self.model, batch, c.temperature)
				else:
					lp, _ = sampled_token_logprobs(self.model, batch, c.temperature)
				out += [lp[k][batch["completion_mask"][k]].cpu() for k in range(len(mb))]
		return out

	def _step(self, rows: list[dict], step: int) -> dict:
		c = self.cfg
		t0 = time.perf_counter()
		rows_g, comps, texts = self._rollout(rows)
		t_gen = time.perf_counter() - t0

		scored = [self.reward_fn(r, t) for r, t in zip(rows_g, texts)]
		rewards = torch.tensor([float(s["reward"]) for s in scored])
		adv = group_advantages(rewards, c.num_generations, c.scale_rewards)
		zero = zero_std_mask(rewards, c.num_generations)
		keep = [i for i in range(len(rows_g)) if not (c.skip_zero_std_groups and zero[i])]

		m: dict = {"step": step, "reward": rewards.mean().item(), "frac_zero_std_groups": zero.float().mean().item()}
		for k in scored[0]:
			if k != "reward" and isinstance(scored[0][k], (int, float, bool)):
				m[k] = sum(float(s[k]) for s in scored) / len(scored)
		m["completion_len"] = sum(len(x) for x in comps) / len(comps)
		m["truncated"] = sum(len(x) == c.max_new_tokens for x in comps) / len(comps)
		self._log_samples(step, rows_g, texts, rewards, adv)

		if not keep:
			m.update(skipped=1, t_gen=t_gen, t_step=time.perf_counter() - t0)
			return m

		old = self._logprobs(rows_g, comps, keep) if c.num_iterations > 1 else None
		ref = self._logprobs(rows_g, comps, keep, reference=True) if c.beta > 0 else None

		# Normalizers over the whole step so the micro-batch split does not change the gradient.
		n_tok = float(sum(len(comps[i]) for i in keep))
		n_seq = float(len(keep))
		sums = {"loss": 0.0, "kl": 0.0, "clipped": 0.0, "entropy": 0.0}
		for _ in range(c.num_iterations):
			self.model.train()
			order = keep[:]
			random.shuffle(order)
			pos = {j: k for k, j in enumerate(keep)}
			for i in range(0, len(order), c.micro_batch_size):
				mb = order[i : i + c.micro_batch_size]
				batch = self.adapter.policy_inputs([rows_g[j] for j in mb], [comps[j] for j in mb]).to(self.model.device)
				cm = batch["completion_mask"]
				logp, ent = sampled_token_logprobs(self.model, batch, c.temperature, with_entropy=True)

				def dense(seqs):  # list of 1-D per-completion tensors -> (B, T) aligned with cm
					d = torch.zeros_like(logp)
					for k, j in enumerate(mb):
						d[k][cm[k]] = seqs[pos[j]].to(d.device)
					return d

				old_lp = dense(old) if old is not None else logp.detach()
				ref_lp = dense(ref) if ref is not None else None
				a = adv[mb].to(logp.device)
				per_tok, st = grpo_token_loss(logp, old_lp, a, ref_lp, c.beta, c.epsilon_low, c.epsilon_high)
				loss = aggregate(per_tok, cm, c.loss_aggregation, n_tok, n_seq) / c.num_iterations
				loss.backward()
				with torch.no_grad():
					sums["loss"] += loss.item()
					sums["clipped"] += (st["clipped"] & cm).sum().item()
					sums["entropy"] += ent.sum().item()
					if "kl" in st:
						sums["kl"] += (st["kl"] * cm).sum().item()
			gn = torch.nn.utils.clip_grad_norm_(self.params, self.config.training.max_grad_norm)
			self.opt.step()
			self.sched.step()
			self.opt.zero_grad(set_to_none=True)

		denom = n_tok * c.num_iterations
		m.update(
			loss=sums["loss"],
			kl=sums["kl"] / denom,
			clip_frac=sums["clipped"] / denom,
			entropy=sums["entropy"] / denom,
			grad_norm=float(gn),
			lr=self.sched.get_last_lr()[0],
			n_trained=len(keep),
			t_gen=t_gen,
			t_step=time.perf_counter() - t0,
			mem_gb=torch.cuda.max_memory_allocated() / 2**30,
		)
		return m

	# --- dev, logging, checkpoints -------------------------------------------------------------

	@torch.no_grad()
	def evaluate(self, step: int) -> dict:
		"""Greedy decode the dev prompts and average the reward (= accuracy for a 0/1 reward)."""
		c = self.cfg
		self.model.eval()
		texts = []
		for i in range(0, len(self.dev_rows), c.generation_batch_size):
			chunk = self.dev_rows[i : i + c.generation_batch_size]
			texts += [self.adapter.decode(x) for x in self.adapter.sample(self.model, chunk, c.max_new_tokens, do_sample=False)]
		scored = [self.reward_fn(r, t) for r, t in zip(self.dev_rows, texts)]
		m = {"step": step, "dev/reward": sum(s["reward"] for s in scored) / len(scored)}
		groups: dict[str, list[float]] = {}
		for r, s in zip(self.dev_rows, scored):
			if r.get("group"):
				groups.setdefault(r["group"], []).append(float(s["reward"]))
		for g, v in sorted(groups.items()):
			m[f"dev/{g}"] = sum(v) / len(v)
		logger.info("dev @ step %d: %s", step, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
		with self.dev_file.open("a") as f:
			f.write(json.dumps(m) + "\n")
		return m

	def _log_samples(self, step, rows, texts, rewards, adv, n_groups: int = 2):
		G = self.cfg.num_generations
		with self.samples_file.open("a") as f:
			for g in range(min(n_groups, len(rows) // G)):
				sl = slice(g * G, (g + 1) * G)
				f.write(json.dumps({"step": step, "prompt": rows[g * G]["prompt"][:200], "image": rows[g * G]["image"], "texts": texts[sl], "rewards": rewards[sl].tolist(), "adv": [round(x, 3) for x in adv[sl].tolist()]}) + "\n")

	def _save(self, step: int, final: bool = False) -> Path:
		d = self.out if final else self.out / f"checkpoint-{step}"
		d.mkdir(parents=True, exist_ok=True)
		self.model.save_pretrained(str(d))
		self.tokenizer.save_pretrained(str(d))
		if not final:
			torch.save({"opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "step": step, "data_pos": self.data_pos, "rng": random.getstate()}, d / "trainer_state.pt")
			ckpts = sorted(self.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
			for old in ckpts[: -self.config.training.save_total_limit]:
				import shutil

				shutil.rmtree(old)
		return d

	# --- main loop --------------------------------------------------------------------------

	def train(self) -> None:
		c, t = self.cfg, self.config.training
		self._make_output_dir()
		self._save_config_snapshot()
		random.seed(t.seed)
		torch.manual_seed(t.seed)
		if t.gradient_checkpointing:
			self.adapter.enable_gradient_checkpointing(self.model)

		train_rows = list(self.dataset.get_train_dataset())
		random.Random(t.seed).shuffle(train_rows)
		self.dev_rows = list(self.dataset.get_eval_dataset() or [])[: c.dev_n]
		steps = c.max_steps or len(train_rows) // c.prompts_per_step
		logger.info("VLMGRPOTrainer | prompts=%d | dev=%d | steps=%d | G=%d | prompts/step=%d", len(train_rows), len(self.dev_rows), steps, c.num_generations, c.prompts_per_step)

		self.params = [p for p in self.model.parameters() if p.requires_grad]
		self.opt = torch.optim.AdamW(self.params, lr=t.learning_rate, weight_decay=t.weight_decay, betas=(0.9, 0.99))
		warm = max(1, int(t.warmup_ratio * steps)) if t.warmup_ratio < 1 else int(t.warmup_ratio)

		def lr_lambda(s):
			if s < warm:
				return (s + 1) / warm
			if t.lr_scheduler_type == "constant":
				return 1.0
			return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))

		self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
		start, self.data_pos = 0, 0
		if c.resume_from_checkpoint:
			from peft import set_peft_model_state_dict
			from safetensors.torch import load_file

			d = Path(c.resume_from_checkpoint)
			set_peft_model_state_dict(self.model, load_file(str(d / "adapter_model.safetensors")))
			st = torch.load(d / "trainer_state.pt", weights_only=False)
			self.opt.load_state_dict(st["opt"])
			self.sched.load_state_dict(st["sched"])
			start, self.data_pos = st["step"], st["data_pos"]
			random.setstate(st["rng"])
			logger.info("Resumed from %s at step %d", d, start)

		if self.dev_rows and start == 0:
			self.evaluate(0)
		log_gpu_memory("before_train")
		for step in range(start + 1, steps + 1):
			if self.data_pos + c.prompts_per_step > len(train_rows):
				random.shuffle(train_rows)
				self.data_pos = 0
			rows = train_rows[self.data_pos : self.data_pos + c.prompts_per_step]
			self.data_pos += c.prompts_per_step
			m = self._step(rows, step)
			with self.metrics_file.open("a") as f:
				f.write(json.dumps(m) + "\n")
			if step % t.logging_steps == 0 or step == 1:
				logger.info("step %d/%d | %s", step, steps, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k != "step"})
			if self.dev_rows and step % c.dev_every_steps == 0:
				self.evaluate(step)
			if step % t.save_steps == 0:
				self._save(step)
		if self.dev_rows and steps % c.dev_every_steps:
			self.evaluate(steps)
		self._save(steps, final=True)
		log_gpu_memory("after_train")
		if c.save_merged:
			base_cfg = self.config.model
			self.model = None
			gc.collect()
			torch.cuda.empty_cache()
			export_merged(self.out, self.out / "merged", base_cfg)
		logger.info("Training complete. Outputs saved to %s", self.out)
