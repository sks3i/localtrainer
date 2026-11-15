"""GRPO math, as pure tensor functions (unit-tested in tests/test_vlm_grpo.py).

Notation: one prompt q, G sampled completions o_1..o_G with scalar rewards r_1..r_G, token t of
completion i is o_{i,t}, and pi_old is the policy that sampled them.

Why a group baseline. The policy gradient is E[ A * grad log pi(o|q) ]. Any baseline b(q) that does
not depend on the sampled o leaves it unbiased and, chosen well, cuts its variance. PPO learns b(q)
with a value network (a second LLM-sized model, and a regression problem with its own errors).
GRPO instead samples G completions of the same prompt and uses their mean reward: a Monte Carlo
estimate of V(q) that costs only extra samples. (The mean includes r_i itself, which biases each
term by a factor (G-1)/G; the leave-one-out mean removes it. That rescales the advantage, nothing
more, so in practice it does not matter.)

Why the ratio. With several gradient passes per batch (num_iterations > 1) the samples come from
pi_old, not the current pi_theta. Importance weighting with rho = pi_theta / pi_old corrects for
that, and clipping rho to [1 - eps_lo, 1 + eps_hi] stops a single batch from moving the policy far
where that correction is unreliable. With one pass, pi_theta == pi_old at the moment the gradient
is taken: rho = 1, grad rho = grad log pi, and the objective reduces to REINFORCE with a group
baseline.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, scale: str = "group", eps: float = 1e-4) -> torch.Tensor:
	"""Advantages for rewards laid out as [g0_s0 .. g0_s(G-1), g1_s0 ...] (each prompt's G in a row).

	scale="group": (r - mean) / (std + eps), the original GRPO. For a 0/1 reward with success rate p
	    in the group, std = sqrt(p(1-p)), so a correct sample gets +sqrt((1-p)/p) and a wrong one
	    -sqrt(p/(1-p)). A group with one success in eight (p=1/8) gives that success A = +2.65: rare
	    wins on hard prompts are pushed hardest.
	scale="none": r - mean (Dr. GRPO). The total "push" on a group is then largest at p = 1/2 and
	    vanishes as p -> 0 or 1, i.e. it weights prompts by how uncertain the policy is on them.
	The std is the population std (divide by G), matching the formula above.
	Groups with all-equal rewards get A = 0 either way: no signal, which is why prompt difficulty
	matters so much (see :func:`zero_std_fraction`).
	"""
	r = rewards.view(-1, group_size).float()
	adv = r - r.mean(dim=1, keepdim=True)
	if scale == "group":
		adv = adv / (r.std(dim=1, unbiased=False, keepdim=True) + eps)
	elif scale != "none":
		raise ValueError(f"unknown scale_rewards {scale!r}")
	return adv.view(-1)


def zero_std_mask(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
	"""Per-completion bool: True if its whole group got the same reward (zero advantage)."""
	r = rewards.view(-1, group_size)
	same = (r == r[:, :1]).all(dim=1, keepdim=True)
	return same.expand_as(r).reshape(-1)


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
	"""Per-token estimator of KL(pi_theta || pi_ref) from a token sampled from pi_theta.

	With x = log pi_ref - log pi_theta:   k3 = exp(x) - x - 1
	  * unbiased: E_{pi_theta}[exp(x)] = sum_o pi_ref(o) = 1 and -E[x] = KL, so E[k3] = 1 + KL - 1 = KL
	  * always >= 0 (e^x >= 1 + x), unlike the naive k1 = -x, which is unbiased but negative half
	    the time and has high variance
	  * low variance when the two policies are close, which is the regime the penalty keeps us in
	Strictly, the samples come from pi_old; with one pass per batch that is pi_theta.
	"""
	x = ref_logp - logp
	return torch.exp(x) - x - 1


def grpo_token_loss(
	logp: torch.Tensor,
	old_logp: torch.Tensor,
	advantages: torch.Tensor,
	ref_logp: torch.Tensor | None = None,
	beta: float = 0.0,
	eps_low: float = 0.2,
	eps_high: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	"""Per-token GRPO loss (to minimize) and per-token diagnostics.

	logp, old_logp, ref_logp: (B, T) log-probs of the sampled tokens. advantages: (B,), one per
	completion, broadcast over its tokens (GRPO has no per-token credit assignment: every token of a
	good completion is pushed up equally, a token-level critic is what PPO would add).

	    rho   = exp(logp - old_logp)
	    obj   = min(rho * A, clip(rho, 1 - eps_low, 1 + eps_high) * A)
	    loss  = -obj + beta * k3(logp, ref_logp)

	The min makes it pessimistic: for A > 0 the gain is capped once rho > 1 + eps_high, for A < 0
	once rho < 1 - eps_low; moving further in the "good" direction earns no more gradient. Moves in
	the "bad" direction are never clipped, so a mistake can always be undone.
	"""
	a = advantages.unsqueeze(1)
	ratio = torch.exp(logp - old_logp)
	clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
	obj = torch.min(ratio * a, clipped * a)
	loss = -obj
	stats = {"clipped": ((ratio < 1 - eps_low) & (a < 0)) | ((ratio > 1 + eps_high) & (a > 0))}
	if ref_logp is not None and beta > 0:
		kl = kl_k3(logp, ref_logp)
		loss = loss + beta * kl
		stats["kl"] = kl
	return loss, stats


def aggregate(per_token: torch.Tensor, mask: torch.Tensor, how: str, n_tokens_total: float, n_seqs_total: float) -> torch.Tensor:
	"""Reduce a (B, T) per-token quantity to the scalar this micro-batch contributes to the step.

	Normalizers are totals over the WHOLE optimizer step (all micro-batches), so summing micro-batch
	losses gives exactly the loss of one big batch, whatever the micro-batch split.
	  how="token":    sum over all tokens / total tokens in the step (DAPO)
	  how="sequence": sum_i (mean over tokens of i) / total completions (original GRPO)
	"""
	m = mask.float()
	if how == "token":
		return (per_token * m).sum() / n_tokens_total
	if how == "sequence":
		return ((per_token * m).sum(1) / m.sum(1).clamp(min=1)).sum() / n_seqs_total
	raise ValueError(f"unknown loss_aggregation {how!r}")
