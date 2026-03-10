"""
DQN 神经网络模型定义
包含标准 DQN 和 Dueling DQN 两种架构
"""

import torch
import torch.nn as nn
from typing import List


class DQNModel(nn.Module):
    """
    标准全连接 DQN 网络

    架构: Input -> [Linear -> ReLU] x N -> Output(Q 值)
    """

    def __init__(self, state_size: int, action_size: int,
                 hidden_sizes: List[int] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        layers: List[nn.Module] = []
        in_dim = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, action_size))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, state_size) float32

        Returns:
            q: (batch, action_size) Q 值
        """
        return self.net(x)


class DuelingDQNModel(nn.Module):
    """
    Dueling DQN 网络

    将 Q 值分解为:
        Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

    其中:
        V(s)    - 状态价值函数（标量）
        A(s,a)  - 动作优势函数（向量）

    优点: 在动作价值差异较小时收敛更快、更稳定
    """

    def __init__(self, state_size: int, action_size: int,
                 hidden_sizes: List[int] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.action_size = action_size

        # 共享特征提取（除最后一层）
        shared_layers: List[nn.Module] = []
        in_dim = state_size
        for h in hidden_sizes[:-1]:
            shared_layers.append(nn.Linear(in_dim, h))
            shared_layers.append(nn.ReLU(inplace=True))
            in_dim = h
        self.shared = nn.Sequential(*shared_layers)

        last_h = hidden_sizes[-1]

        # 价值流: 输出标量 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, last_h),
            nn.ReLU(inplace=True),
            nn.Linear(last_h, 1),
        )

        # 优势流: 输出向量 A(s, ·)
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, last_h),
            nn.ReLU(inplace=True),
            nn.Linear(last_h, action_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            q: (batch, action_size)
        """
        feat  = self.shared(x)
        value = self.value_stream(feat)                           # (B, 1)
        adv   = self.advantage_stream(feat)                       # (B, A)
        # Q = V + A - mean(A)  →  去均值使 V 的估计更稳定
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q


def build_network(state_size: int, action_size: int,
                  hidden_sizes: List[int], use_dueling: bool) -> nn.Module:
    """工厂函数，根据配置构建网络"""
    if use_dueling:
        return DuelingDQNModel(state_size, action_size, hidden_sizes)
    return DQNModel(state_size, action_size, hidden_sizes)
