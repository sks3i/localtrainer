"""Image + text GRPO configuration (``trainer_type: vlm_grpo``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VLMGRPOConfig:
	"""Group Relative Policy Optimization for a vision-language model with a verifiable reward.

	Data is JSONL, one prompt per line: ``{"image": <path>, "prompt": <str>, ...}``. Every other key is
	passed through untouched to the reward function, which is where the target (a mask, a point set,
	a reference answer) lives. There is no ``answer`` column: the model's own samples are the data.

	One optimizer step:
	    1. take ``prompts_per_step`` prompts and sample ``num_generations`` (G) completions for each
	    2. score every completion with ``reward_fn(row, text) -> dict`` (must contain "reward")
	    3. advantage of completion i in its group: A_i = (r_i - mean(r)) / (std(r) + eps)
	    4. maximize the clipped surrogate  min(rho A, clip(rho, 1-eps_lo, 1+eps_hi) A)  per token,
	       minus ``beta`` x KL(policy || reference), where the reference is the starting model
	       (the base weights with the LoRA adapter switched off, so it costs no extra memory)

	Attributes:
		train_file: JSONL prompts.
		dev_file: JSONL prompts scored with the same reward under greedy decoding every
		    ``dev_every_steps``; mean reward = accuracy for a 0/1 reward. ``None`` skips it.
		dev_n: Cap on dev prompts.
		dev_every_steps: Optimizer steps between dev evaluations.
		image_root: Directory that relative ``image`` paths are resolved against.
		max_image_side: Downscale images whose longest side exceeds this before the processor.
		reward_fn: ``"package.module:function"``, called as ``fn(row, text) -> {"reward": float, ...}``.
		    Other numeric keys are averaged and logged (e.g. a parse-failure flag).
		num_generations: G, completions per prompt. The group is the baseline, so G >= 2.
		prompts_per_step: Prompts per optimizer step (so G x this completions per step).
		generation_batch_size: Sequences per ``generate`` call (memory knob only).
		micro_batch_size: Sequences per forward/backward pass (memory knob only).
		max_new_tokens: Generation cap. A completion cut off before EOS gets no EOS token, and the
		    reward function sees the truncated text.
		temperature: Sampling temperature. Log-probs are computed on logits / temperature, so the
		    ratio and KL refer to the distribution that was actually sampled from.
		top_p / top_k: Kept at 1.0 / 0 by default. Truncated sampling makes the sampled distribution
		    differ from pi_theta, so the importance ratio would be biased.
		beta: KL coefficient. 0 skips the reference forward pass entirely.
		epsilon_low / epsilon_high: Clip range of the ratio, [1 - eps_lo, 1 + eps_hi]. A higher upper
		    bound than lower ("clip-higher", DAPO) lets low-probability tokens that turned out good
		    grow faster, which helps keep entropy up.
		num_iterations: Gradient passes over each batch of rollouts (PPO epochs, mu). With 1 the
		    update is exactly on-policy, the ratio is 1, and clipping never fires.
		scale_rewards: "group" divides by the group std (original GRPO); "none" keeps raw
		    r - mean (Dr. GRPO), so near-unanimous groups do not get amplified.
		loss_aggregation: "token" averages over every completion token in the step (DAPO);
		    "sequence" averages within each completion first, then over completions (original
		    GRPO), which gives tokens of short completions more weight.
		skip_zero_std_groups: Drop groups whose rewards are all equal from the policy loss (their
		    advantage is 0 anyway); they then cost no backward pass. Their KL term is dropped too.
		max_steps: Optimizer steps. ``None`` means one pass over ``train_file``.
		save_merged: Merge the adapter into the base weights at the end (``<output_dir>/merged``).
		resume_from_checkpoint: A ``checkpoint-N`` dir with adapter + optimizer + trainer state.
	"""

	train_file: str = ""
	dev_file: Optional[str] = None
	dev_n: int = 300
	dev_every_steps: int = 50
	image_root: Optional[str] = None
	max_image_side: Optional[int] = 1024
	max_num_samples: Optional[int] = None
	reward_fn: str = ""
	num_generations: int = 8
	prompts_per_step: int = 8
	generation_batch_size: int = 32
	micro_batch_size: int = 8
	max_new_tokens: int = 32
	temperature: float = 1.0
	top_p: float = 1.0
	top_k: int = 0
	beta: float = 0.02
	epsilon_low: float = 0.2
	epsilon_high: float = 0.28
	num_iterations: int = 1
	scale_rewards: str = "group"
	loss_aggregation: str = "token"
	skip_zero_std_groups: bool = True
	max_steps: Optional[int] = None
	save_merged: bool = True
	resume_from_checkpoint: Optional[str] = None
