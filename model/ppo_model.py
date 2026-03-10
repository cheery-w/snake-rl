"""
PPO Actor-Critic 神经网络模型
包含共享主干 + 独立 Actor/Critic 输出头
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Tuple


class PPOModel(nn.Module):
    """
    PPO Actor-Critic 网络

    架构:
        共享主干 (shared backbone)
            ↓
        Actor 头 → 动作概率分布 (softmax)
        Critic 头 → 状态价值标量 V(s)

    使用::

        model = PPOModel(state_size=11, action_size=3)
        dist, value = model(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    """

    def __init__(
        self,
        state_size:   int,
        action_size:  int,
        hidden_sizes: List[int] = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.action_size = action_size

        # ── 共享主干 ────────────────────────────────────────────────
        backbone_layers: List[nn.Module] = []
        in_dim = state_size
        for h in hidden_sizes:
            backbone_layers.append(nn.Linear(in_dim, h))
            backbone_layers.append(nn.Tanh())   # PPO 常用 Tanh 代替 ReLU
            in_dim = h
        self.backbone = nn.Sequential(*backbone_layers)

        # ── Actor 头（策略网络）────────────────────────────────────
        self.actor = nn.Linear(in_dim, action_size)

        # ── Critic 头（价值网络）───────────────────────────────────
        self.critic = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """正交初始化（PPO 常用方案）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Actor 头使用更小的增益，让初始策略更均匀
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        # Critic 头使用 1.0 增益
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        前向传播

        Args:
            x: (batch, state_size) float32

        Returns:
            dist:  Categorical 分布对象（可采样动作、计算 log_prob、熵）
            value: (batch, 1) 状态价值
        """
        feat   = self.backbone(x)
        logits = self.actor(feat)
        value  = self.critic(feat)
        dist   = Categorical(logits=logits)
        return dist, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """仅计算状态价值（无需动作分布时使用）"""
        feat  = self.backbone(x)
        return self.critic(feat)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算动作、log_prob、熵、价值（训练时一次性使用）

        Args:
            x:      (batch, state_size)
            action: (batch,) 可选；None 时从分布中采样

        Returns:
            action, log_prob, entropy, value
        """
        dist, value = self.forward(x)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)
