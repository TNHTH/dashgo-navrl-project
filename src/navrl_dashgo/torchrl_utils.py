from __future__ import annotations

from typing import Iterable, Union

import torch
import torch.nn as nn
from tensordict import TensorDict


class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta: float = 0.995,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_shape = (
            torch.Size(input_shape) if isinstance(input_shape, Iterable) else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta
        self.register_buffer("running_mean", torch.zeros(self.input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(self.input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

    def running_mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        debias = self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean = self.running_mean / debias
        debiased_mean_sq = self.running_mean_sq / debias
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor) -> None:
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)
        weight = self.beta
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor) -> torch.Tensor:
        mean, var = self.running_mean_var()
        return (input_vector - mean) / torch.sqrt(var)

    def denormalize(self, input_vector: torch.Tensor) -> torch.Tensor:
        mean, var = self.running_mean_var()
        return input_vector * torch.sqrt(var) + mean


def make_mlp(num_units: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {
        "alpha": torch.distributions.constraints.positive,
        "beta": torch.distributions.constraints.positive,
    }

    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, validate_args=None):
        super().__init__(torch.distributions.Beta(alpha, beta), 1, validate_args=validate_args)

    @property
    def deterministic_sample(self) -> torch.Tensor:
        return self.mean


class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = 1.0 + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1.0 + self.beta_softplus(self.beta_layer(features)) + 1e-6
        alpha = alpha.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return alpha, beta


class GAE(nn.Module):
    def __init__(self, gamma: float, lmbda: float) -> None:
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))

    def forward(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1.0 - terminated.float()
        gae = 0.0
        for step in reversed(range(num_steps)):
            delta = reward[:, step] + self.gamma * next_value[:, step] * not_done[:, step] - value[:, step]
            gae = delta + self.gamma * self.lmbda * not_done[:, step] * gae
            advantages[:, step] = gae
        returns = advantages + value
        return advantages, returns


def make_batch(tensordict: TensorDict, num_minibatches: int):
    flat_td = tensordict.reshape(-1)
    usable = (flat_td.shape[0] // num_minibatches) * num_minibatches
    perm = torch.randperm(usable, device=flat_td.device).reshape(num_minibatches, -1)
    for indices in perm:
        yield flat_td[indices]


def squeeze_trailing_agent_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 3 and tensor.shape[-2] == 1:
        return tensor.squeeze(-2)
    return tensor
