"""CPU tests for the no-stride chunk replay buffer."""

from __future__ import annotations

import torch

from smolvla_rltoken.rl.replay import ChunkReplayBuffer
from smolvla_rltoken.rl.td import bootstrap_coeff, discounted_return
from smolvla_rltoken.rollout.transition import ChunkTransition


def _transition(**overrides) -> ChunkTransition:
    payload = dict(
        z_rl=torch.ones(4),
        proprio=torch.zeros(3),
        reference_action=torch.full((4, 2), 0.5),
        executed_action=torch.full((4, 2), 0.2),
        reward_sequence=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        n_steps=3,
        next_z_rl=torch.ones(4) * 2,
        next_proprio=torch.ones(3),
        next_reference_action=torch.full((4, 2), 0.7),
        terminated=0.0,
        truncated=1.0,
        episode_id=1,
        chunk_id=2,
    )
    payload.update(overrides)
    return ChunkTransition(**payload)


def test_add_increments_size_by_one():
    buf = ChunkReplayBuffer(8, rl_token_dim=4, proprio_dim=3, chunk_len=4, action_dim=2)
    assert not hasattr(buf, "offsets")
    assert buf.stride == 1
    buf.add(_transition())
    assert len(buf) == 1
    buf.add(_transition(chunk_id=3, terminated=1.0, truncated=0.0))
    assert len(buf) == 2


def test_sample_shapes_and_executed_not_reference():
    buf = ChunkReplayBuffer(8, rl_token_dim=4, proprio_dim=3, chunk_len=4, action_dim=2)
    executed = torch.arange(8.0).reshape(4, 2)
    buf.add(_transition(executed_action=executed))
    batch = buf.sample(1)
    assert batch["x"].shape == (1, 7)
    assert batch["x_next"].shape == (1, 7)
    assert batch["executed_action"].shape == (1, 4, 2)
    assert batch["reference_action"].shape == (1, 4, 2)
    assert batch["next_reference_action"].shape == (1, 4, 2)
    assert batch["reward_sequence"].shape == (1, 4)
    assert batch["n_steps"].shape == (1,)
    assert batch["terminated"].shape == (1,)
    assert batch["truncated"].shape == (1,)
    torch.testing.assert_close(batch["executed_action"][0], executed)
    assert not torch.equal(batch["executed_action"], batch["reference_action"])
    assert "offset" not in batch


def test_terminated_and_truncated_are_separate():
    buf = ChunkReplayBuffer(4, rl_token_dim=4, proprio_dim=3, chunk_len=4, action_dim=2)
    buf.add(_transition(terminated=1.0, truncated=0.0))
    buf.add(_transition(terminated=0.0, truncated=1.0, chunk_id=1))
    assert buf.terminated[0].item() == 1.0
    assert buf.truncated[0].item() == 0.0
    assert buf.terminated[1].item() == 0.0
    assert buf.truncated[1].item() == 1.0


def test_discounted_return_uses_n_steps():
    rewards = torch.tensor([[0.0, 0.0, 1.0, 9.0], [1.0, 0.0, 0.0, 0.0]])
    n_steps = torch.tensor([3, 1])
    gamma = 0.5
    got = discounted_return(rewards, n_steps, gamma)
    expected = torch.tensor([0.5**2 * 1.0, 1.0])
    torch.testing.assert_close(got, expected)


def test_bootstrap_coeff_gamma_n_and_terminated():
    n_steps = torch.tensor([6, 10, 4])
    terminated = torch.tensor([0.0, 1.0, 0.0])
    coeff = bootstrap_coeff(n_steps, terminated, 0.99)
    torch.testing.assert_close(coeff[0], torch.tensor(0.99**6))
    torch.testing.assert_close(coeff[1], torch.tensor(0.0))
    torch.testing.assert_close(coeff[2], torch.tensor(0.99**4))
    assert coeff[0].item() != (0.99**10)
