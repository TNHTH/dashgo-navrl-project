from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

from .torchrl_utils import BetaActor, GAE, IndependentBeta, ValueNorm, make_batch, make_mlp, squeeze_trailing_agent_dim


class NonFiniteTrainingStateError(RuntimeError):
    pass


def ensure_finite_tensor(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        return
    if not torch.is_tensor(tensor):
        return
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return
    flat = tensor.detach().reshape(-1)
    bad_values = flat[~finite_mask.reshape(-1)][:8].cpu().tolist()
    raise NonFiniteTrainingStateError(f"{name} 含非有限值: sample={bad_values}")


def ensure_finite_module_grads(name: str, module: nn.Module) -> None:
    for param_name, param in module.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise NonFiniteTrainingStateError(f"{name}.{param_name}.grad 含非有限值")


def bootstrap_done_flags(next_tensordict: TensorDict) -> torch.Tensor:
    return next_tensordict["terminated"] | next_tensordict["truncated"]


class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]),
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]),
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]),
            nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128),
            nn.LayerNorm(128),
        ).to(self.device)

        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64]),
        ).to(self.device)

        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(
                dynamic_obstacle_network,
                [("agents", "observation", "dynamic_obstacle")],
                ["_dynamic_obstacle_feature"],
            ),
            TensorDictModule(
                Rearrange("n a d -> n (a d)"),
                [("agents", "observation", "state")],
                ["_state_feature"],
            ),
            CatTensors(
                ["_cnn_feature", "_state_feature", "_dynamic_obstacle_feature"],
                "_feature",
                del_keys=False,
            ),
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        action_shape = tuple(action_spec.shape)
        self.n_agents = action_shape[-2]
        self.action_dim = action_shape[-1]
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True,
        ).to(self.device)

        self.critic = TensorDictModule(nn.LazyLinear(1), ["_feature"], ["state_value"]).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)
        self.gae = GAE(float(cfg.gamma), float(cfg.gae_lambda))
        self.critic_loss_fn = nn.HuberLoss(delta=10.0)

        self.feature_extractor_optim = torch.optim.Adam(
            self.feature_extractor.parameters(), lr=float(cfg.feature_extractor.learning_rate)
        )
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.actor.learning_rate))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.critic.learning_rate))

        dummy_input = observation_spec.zero().to(self.device)
        self.__call__(dummy_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        ensure_finite_tensor("feature_extractor._feature", tensordict.get("_feature", None))
        self.actor(tensordict)
        ensure_finite_tensor("actor.alpha", tensordict.get("alpha", None))
        ensure_finite_tensor("actor.beta", tensordict.get("beta", None))
        ensure_finite_tensor("actor.action_normalized", tensordict.get(("agents", "action_normalized"), None))
        ensure_finite_tensor("actor.sample_log_prob", tensordict.get("sample_log_prob", None))
        self.critic(tensordict)
        ensure_finite_tensor("critic.state_value", tensordict.get("state_value", None))
        actions = (2.0 * tensordict["agents", "action_normalized"] * float(self.cfg.actor.action_limit)) - float(
            self.cfg.actor.action_limit
        )
        ensure_finite_tensor("actor.action", actions)
        tensordict["agents", "action"] = actions
        return tensordict

    def update(self, tensordict: TensorDict) -> dict[str, float]:
        next_tensordict = tensordict["next"].reshape(-1).to(self.device)
        with torch.no_grad():
            next_tensordict = self.feature_extractor(next_tensordict)
            ensure_finite_tensor("next._feature", next_tensordict.get("_feature", None))
            next_values = self.critic(next_tensordict)["state_value"].reshape(*tensordict.batch_size, 1)
            ensure_finite_tensor("next.state_value", next_values)

        rewards = squeeze_trailing_agent_dim(tensordict["next", "agents", "reward"]).to(self.device)
        dones = bootstrap_done_flags(tensordict["next"]).to(self.device)
        values = tensordict["state_value"].to(self.device)
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)
        ensure_finite_tensor("collector.reward", rewards)
        ensure_finite_tensor("collector.value", values)
        ensure_finite_tensor("collector.next_value", next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        ensure_finite_tensor("gae.adv_raw", adv)
        ensure_finite_tensor("gae.ret_raw", ret)
        adv_mean = adv.mean()
        adv_std = adv.std().clip(1e-7)
        ensure_finite_tensor("gae.adv_mean", adv_mean)
        ensure_finite_tensor("gae.adv_std", adv_std)
        adv = (adv - adv_mean) / adv_std
        ensure_finite_tensor("gae.adv_normalized", adv)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        ensure_finite_tensor("gae.ret_normalized", ret)
        tensordict = tensordict.to(self.device)
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for _ in range(int(self.cfg.training_epoch_num)):
            for minibatch in make_batch(tensordict, int(self.cfg.num_minibatches)):
                infos.append(self._update(minibatch))
        infos = torch.stack(infos).to_tensordict().apply(torch.mean, batch_size=[])
        return {key: float(value.item()) for key, value in infos.items()}

    def _update(self, tensordict: TensorDict) -> TensorDict:
        self.feature_extractor(tensordict)
        ensure_finite_tensor("minibatch._feature", tensordict.get("_feature", None))
        action_dist = self.actor.get_dist(tensordict)
        ensure_finite_tensor("minibatch.alpha", tensordict.get("alpha", None))
        ensure_finite_tensor("minibatch.beta", tensordict.get("beta", None))
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")])
        ensure_finite_tensor("minibatch.sample_log_prob", tensordict.get("sample_log_prob", None))
        ensure_finite_tensor("minibatch.log_probs", log_probs)

        action_entropy = action_dist.entropy()
        ensure_finite_tensor("minibatch.entropy", action_entropy)
        entropy_loss = -float(self.cfg.entropy_loss_coefficient) * torch.mean(action_entropy)
        ensure_finite_tensor("loss.entropy", entropy_loss)

        advantage = tensordict["adv"]
        ensure_finite_tensor("minibatch.adv", advantage)
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ensure_finite_tensor("minibatch.ratio", ratio)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.0 - float(self.cfg.actor.clip_ratio), 1.0 + float(self.cfg.actor.clip_ratio))
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        ensure_finite_tensor("loss.actor", actor_loss)

        b_value = tensordict["state_value"]
        ret = tensordict["ret"]
        value = self.critic(tensordict)["state_value"]
        ensure_finite_tensor("minibatch.state_value", value)
        ensure_finite_tensor("minibatch.ret", ret)
        value_clipped = b_value + (value - b_value).clamp(
            -float(self.cfg.critic.clip_ratio),
            float(self.cfg.critic.clip_ratio),
        )
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)
        ensure_finite_tensor("loss.critic", critic_loss)

        loss = entropy_loss + actor_loss + critic_loss
        ensure_finite_tensor("loss.total", loss)
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        ensure_finite_module_grads("feature_extractor", self.feature_extractor)
        ensure_finite_module_grads("actor", self.actor)
        ensure_finite_module_grads("critic", self.critic)

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=float(self.cfg.max_grad_norm))
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), max_norm=float(self.cfg.max_grad_norm)
        )
        nn.utils.clip_grad.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=float(self.cfg.max_grad_norm))
        ensure_finite_tensor("grad_norm.actor", actor_grad_norm)
        ensure_finite_tensor("grad_norm.critic", critic_grad_norm)

        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        explained_var = 1.0 - F.mse_loss(value, ret) / ret.var().clamp(min=1.0e-6)
        ensure_finite_tensor("stats.explained_var", explained_var)
        return TensorDict(
            {
                "actor_loss": actor_loss.detach(),
                "critic_loss": critic_loss.detach(),
                "entropy": entropy_loss.detach(),
                "actor_grad_norm": actor_grad_norm.detach(),
                "critic_grad_norm": critic_grad_norm.detach(),
                "explained_var": explained_var.detach(),
            },
            [],
        )
