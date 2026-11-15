"""CPU tests for the GRPO math (advantages, KL estimator, clipped loss, aggregation)."""

import math

import torch

from train_utils.grpo.math import aggregate, group_advantages, grpo_token_loss, kl_k3, zero_std_mask


def test_group_advantages_binary_closed_form():
	# One success in a group of 8: p = 1/8, std = sqrt(p(1-p)); winner gets sqrt((1-p)/p).
	r = torch.tensor([1.0] + [0.0] * 7 + [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
	a = group_advantages(r, 8, eps=0.0)
	p = 1 / 8
	assert math.isclose(a[0].item(), math.sqrt((1 - p) / p), rel_tol=1e-5)
	assert math.isclose(a[1].item(), -math.sqrt(p / (1 - p)), rel_tol=1e-5)
	assert torch.allclose(a.view(2, 8).sum(1), torch.zeros(2), atol=1e-5)  # zero-mean per group
	assert torch.allclose(a[8:].abs(), torch.ones(8))  # p = 1/2 -> +-1


def test_group_advantages_unscaled_and_zero_std():
	r = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
	a = group_advantages(r, 4, scale="none")
	assert torch.allclose(a[:4], torch.zeros(4))
	assert torch.allclose(a[4:], torch.tensor([-0.25, 0.75, -0.25, -0.25]))
	assert zero_std_mask(r, 4).tolist() == [True] * 4 + [False] * 4


def test_kl_k3_unbiased_and_nonnegative():
	torch.manual_seed(0)
	p = torch.softmax(torch.randn(50), 0)
	q = torch.softmax(torch.randn(50), 0)
	exact = (p * (p.log() - q.log())).sum()
	idx = torch.multinomial(p, 400_000, replacement=True)
	k3 = kl_k3(p.log()[idx], q.log()[idx])
	assert (k3 >= 0).all()
	assert math.isclose(k3.mean().item(), exact.item(), rel_tol=0.02)


def test_on_policy_gradient_is_reinforce():
	# With old_logp = logp.detach() the ratio is 1 and d(loss)/d(logp) = -A for every token.
	logp = torch.log(torch.tensor([[0.5, 0.2], [0.1, 0.9]])).requires_grad_()
	adv = torch.tensor([2.0, -1.0])
	loss, st = grpo_token_loss(logp, logp.detach(), adv)
	loss.sum().backward()
	assert torch.allclose(logp.grad, torch.tensor([[-2.0, -2.0], [1.0, 1.0]]))
	assert not st["clipped"].any()


def test_clipping_zeroes_gradient_only_past_the_trust_region():
	old = torch.zeros(1, 4)
	# ratios 1.5 (A>0: clipped), 0.5 (A>0: not clipped, bad direction), 0.5 (A<0: clipped), 1.5 (A<0: not clipped)
	logp = torch.log(torch.tensor([[1.5, 0.5, 0.5, 1.5]])).requires_grad_()
	adv = torch.tensor([1.0])
	loss, st = grpo_token_loss(logp[:, :2], old[:, :2], adv, eps_low=0.2, eps_high=0.28)
	loss.sum().backward()
	g = logp.grad[0, :2]
	assert g[0] == 0 and g[1] != 0
	assert st["clipped"].tolist() == [[True, False]]
	logp.grad = None
	loss, st = grpo_token_loss(logp[:, 2:], old[:, 2:], -adv, eps_low=0.2, eps_high=0.28)
	loss.sum().backward()
	g = logp.grad[0, 2:]
	assert g[0] == 0 and g[1] != 0


def test_aggregate_is_split_invariant():
	torch.manual_seed(0)
	x = torch.randn(4, 6)
	mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]).bool()
	n_tok, n_seq = float(mask.sum()), 4.0
	for how in ("token", "sequence"):
		whole = aggregate(x, mask, how, n_tok, n_seq)
		split = aggregate(x[:1], mask[:1], how, n_tok, n_seq) + aggregate(x[1:], mask[1:], how, n_tok, n_seq)
		assert torch.allclose(whole, split)
	assert torch.allclose(aggregate(x, mask, "token", n_tok, n_seq), x[mask].mean())
