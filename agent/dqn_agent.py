"""
DQN 智能体
支持 Double DQN + Dueling DQN 组合（Rainbow 子集）
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from model.dqn_model import build_network
from agent.memory import ReplayBuffer
from config import Config


class DQNAgent:
    """
    DQN 智能体

    核心机制:
      - ε-贪心探索策略（epsilon 每轮衰减）
      - 经验回放（打破样本相关性）
      - 目标网络硬更新（减少训练振荡）
      - Double DQN（减少 Q 值高估偏差）
      - Dueling DQN（由 build_network 控制）
    """

    def __init__(self, cfg: Config = None):
        self.cfg    = cfg or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 在线网络（持续更新）
        self.online_net = build_network(
            cfg.STATE_SIZE, cfg.ACTION_SIZE,
            cfg.HIDDEN_SIZES, cfg.USE_DUELING,
        ).to(self.device)

        # 目标网络（定期同步）
        self.target_net = build_network(
            cfg.STATE_SIZE, cfg.ACTION_SIZE,
            cfg.HIDDEN_SIZES, cfg.USE_DUELING,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.LR)
        self.memory    = ReplayBuffer(cfg.MEMORY_SIZE)

        # 探索参数
        self.epsilon = cfg.EPS_START
        self.steps   = 0   # 全局步数计数，用于目标网络更新

        print(f"[Agent] Device: {self.device}  |  "
              f"Dueling={cfg.USE_DUELING}  Double={cfg.USE_DOUBLE}")
        total_params = sum(p.numel() for p in self.online_net.parameters())
        print(f"[Agent] Online net params: {total_params:,}")

    # ------------------------------------------------------------------
    # 动作选择
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> int:
        """
        ε-贪心策略选择动作

        Args:
            state:         当前状态向量 (state_size,)
            deterministic: True 时完全贪心（评估/演示使用）

        Returns:
            action (int): 0=直走, 1=右转, 2=左转
        """
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.cfg.ACTION_SIZE)

        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # 经验存储
    # ------------------------------------------------------------------

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # 学习（一次梯度更新）
    # ------------------------------------------------------------------

    def learn(self) -> float:
        """
        从经验回放中采样，执行一次 DQN 更新

        Returns:
            loss value (float)，样本不足时返回 0.0
        """
        if len(self.memory) < self.cfg.BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.cfg.BATCH_SIZE
        )

        # 转换为 tensor
        s  = torch.tensor(states,      dtype=torch.float32,  device=self.device)
        a  = torch.tensor(actions,     dtype=torch.long,      device=self.device)
        r  = torch.tensor(rewards,     dtype=torch.float32,  device=self.device)
        s_ = torch.tensor(next_states, dtype=torch.float32,  device=self.device)
        d  = torch.tensor(dones,       dtype=torch.float32,  device=self.device)

        # ── 计算目标 Q 值 ───────────────────────────────────────────
        with torch.no_grad():
            if self.cfg.USE_DOUBLE:
                # Double DQN:
                #   用在线网络选动作，用目标网络估值
                #   减少 Q 值高估偏差
                best_actions = self.online_net(s_).argmax(dim=1, keepdim=True)
                q_next = self.target_net(s_).gather(1, best_actions).squeeze(1)
            else:
                q_next = self.target_net(s_).max(dim=1)[0]

            # 终止状态时目标 Q = r（不考虑未来）
            target_q = r + self.cfg.GAMMA * q_next * (1.0 - d)

        # ── 计算当前 Q 值 ───────────────────────────────────────────
        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # ── Huber 损失（比 MSE 对异常值更鲁棒）───────────────────────
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── 硬更新目标网络 ──────────────────────────────────────────
        self.steps += 1
        if self.steps % self.cfg.TARGET_UPDATE == 0:
            self.update_target()

        return float(loss.item())

    def update_target(self):
        """将在线网络权重完整复制到目标网络"""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # ε 衰减
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """每轮训练结束后调用，乘法衰减 ε"""
        self.epsilon = max(self.cfg.EPS_END,
                          self.epsilon * self.cfg.EPS_DECAY)

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------

    def save(self, path: str, metadata: dict = None):
        """保存模型权重和训练元数据"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "epsilon":           self.epsilon,
            "steps":             self.steps,
        }
        if metadata:
            ckpt.update(metadata)
        torch.save(ckpt, path)

    def load(self, path: str):
        """加载模型权重，恢复训练状态"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epsilon = ckpt.get("epsilon", self.cfg.EPS_END)
        self.steps   = ckpt.get("steps",   0)
        print(f"[Agent] Loaded checkpoint: {path}  ε={self.epsilon:.4f}")
        return ckpt

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def memory_size(self) -> int:
        return len(self.memory)
