"""CPU tests for VLM SFT label masking and the answer-token loss."""

import pytest
import torch
import torch.nn.functional as F

from train_utils.vlm.adapters import IGNORE_INDEX, answer_labels
from train_utils.trainers.vlm_sft_trainer import answer_token_loss

END = 99


def test_answer_labels_masks_prompt_and_after_end():
	prompt = [1, 2, 3, 4]
	suffix = [10, 11, 12, END, 7]  # answer, end-of-turn, trailing "\n" from the template
	labels = answer_labels(prompt + suffix, suffix, END)
	assert labels == [IGNORE_INDEX] * 4 + [10, 11, 12, END, IGNORE_INDEX]


def test_answer_labels_rejects_boundary_mismatch():
	with pytest.raises(ValueError):
		answer_labels([1, 2, 3, 10, 11], [9, 10, 11], END)


def test_answer_token_loss_matches_shifted_ce():
	torch.manual_seed(0)
	logits = torch.randn(2, 6, 5)
	labels = torch.full((2, 6), IGNORE_INDEX)
	labels[0, 4:] = torch.tensor([1, 2])
	labels[1, 3:] = torch.tensor([3, 4, 0])
	ref = F.cross_entropy(logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
	assert torch.allclose(answer_token_loss(logits, labels), ref)


def test_answer_token_loss_normalizes_over_accumulated_tokens():
	# Two micro-batches with 2 and 3 supervised tokens: summing the per-step losses must equal the
	# per-token mean over all 5 tokens, which is what one big batch would give.
	torch.manual_seed(0)
	l1, l2 = torch.randn(1, 4, 5), torch.randn(1, 5, 5)
	y1 = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1, 2]])
	y2 = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 0]])
	accumulated = answer_token_loss(l1, y1, 5) + answer_token_loss(l2, y2, 5)
	all_ce = torch.cat(
		[F.cross_entropy(l1[0, 1:3], y1[0, 2:], reduction="none"), F.cross_entropy(l2[0, 1:4], y2[0, 2:], reduction="none")]
	)
	assert torch.allclose(accumulated, all_ce.mean())
