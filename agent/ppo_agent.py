"""
PPO（近端策略优化）智能体
使用 Actor-Critic 架构，适合处理连续/高维状态空间
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

from model.ppo_model import PPOModel
from config import Config


class RolloutBuffer:
    """
    PPO 轨迹缓冲区

    存储一段完整轨迹（rollout），用于计算 GAE 优势估计和策略更新
    """

    def __init__(self):
        self.states:    List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(np.asarray(state,   dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.rewards)


class PPOAgent:
    """
    PPO 智能体

    核心机制:
        - Actor-Critic 网络（共享主干）
        - GAE（广义优势估计，λ 加权多步回报）
        - 重要性采样裁剪（clip ratio ε）
        - 熵正则化（鼓励探索）
        - 多个 epoch 重复利用同一轨迹

    参数（可在 Config 中覆盖，或传入自定义值）:
        ppo_epochs      更新轮次，默认 4
        clip_eps        裁剪参数，默认 0.2
        gae_lambda      GAE λ，默认 0.95
        vf_coef         价值函数损失系数，默认 0.5
        ent_coef        熵正则系数，默认 0.01
        rollout_steps   每次收集的步数，默认 512
    """

    def __init__(
        self,
        cfg: Config = None,
        ppo_epochs:    int   = 4,
        clip_eps:      float = 0.2,
        gae_lambda:    float = 0.95,
        vf_coef:       float = 0.5,
        ent_coef:      float = 0.01,
        rollout_steps: int   = 512,
        mini_batch:    int   = 64,
    ):
        self.cfg    = cfg or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ppo_epochs    = ppo_epochs
        self.clip_eps      = clip_eps
        self.gae_lambda    = gae_lambda
        self.vf_coef       = vf_coef
        self.ent_coef      = ent_coef
        self.rollout_steps = rollout_steps
        self.mini_batch    = mini_batch

        self.model = PPOModel(
            self.cfg.STATE_SIZE,
            self.cfg.ACTION_SIZE,
            self.cfg.HIDDEN_SIZES,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.cfg.LR, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.total_steps = 0

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[PPOAgent] Device: {self.device}  |  Params: {total_params:,}")

    # ------------------------------------------------------------------
    # 动作选择
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        选择动作

        Args:
            state:         状态向量 (state_size,)
            deterministic: True 时选最大概率动作（评估 / 演示用）

        Returns:
            (action, log_prob, value)
        """
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        dist, value = self.model(s)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    def store(self, state, action, log_prob, reward, value, done):
        """将一步经验存入缓冲区"""
        self.buffer.add(state, action, log_prob, reward, value, done)
        self.total_steps += 1

    # ------------------------------------------------------------------
    # 计算 GAE 优势
    # ------------------------------------------------------------------

    def _compute_gae(
        self, last_value: float, last_done: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        广义优势估计（GAE-λ）

        Returns:
            advantages (N,), returns (N,)
        """
        buf      = self.buffer
        n        = len(buf)
        advs     = np.zeros(n, dtype=np.float32)
        gae      = 0.0
        next_val = last_value
        next_done = last_done

        for t in reversed(range(n)):
            mask     = 1.0 - float(next_done)
            delta    = (buf.rewards[t]
                        + self.cfg.GAMMA * next_val * mask
                        - buf.values[t])
            gae      = delta + self.cfg.GAMMA * self.gae_lambda * mask * gae
            advs[t]  = gae
            next_val  = buf.values[t]
            next_done = buf.dones[t]

        returns = advs + np.array(buf.values, dtype=np.float32)
        return advs, returns

    # ------------------------------------------------------------------
    # 学习
    # ------------------------------------------------------------------

    def learn(
        self, last_value: float = 0.0, last_done: bool = True
    ) -> dict:
        """
        使用当前缓冲区中的轨迹执行 PPO 更新

        Args:
            last_value: 最后一步之后的状态价值（用于 bootstrap）
            last_done:  最后一步是否终止

        Returns:
            metrics dict: pg_loss, vf_loss, entropy, approx_kl
        """
        advs, returns = self._compute_gae(last_value, last_done)

        # 归一化优势（稳定训练）
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # 转为 tensor
        buf = self.buffer
        s   = torch.tensor(np.stack(buf.states),    dtype=torch.float32,  device=self.device)
        a   = torch.tensor(buf.actions,             dtype=torch.long,     device=self.device)
        lp  = torch.tensor(buf.log_probs,           dtype=torch.float32,  device=self.device)
        adv = torch.tensor(advs,                    dtype=torch.float32,  device=self.device)
        ret = torch.tensor(returns,                 dtype=torch.float32,  device=self.device)

        n = len(s)
        total_pg   = 0.0
        total_vf   = 0.0
        total_ent  = 0.0
        total_kl   = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            # mini-batch 随机打乱
            idx = np.random.permutation(n)
            for start in range(0, n, self.mini_batch):
                end = start + self.mini_batch
                mb  = idx[start:end]

                _, new_lp, entropy, new_val = self.model.get_action_and_value(
                    s[mb], a[mb]
                )

                ratio    = (new_lp - lp[mb]).exp()
                pg_loss1 = -adv[mb] * ratio
                pg_loss2 = -adv[mb] * ratio.clamp(
                    1 - self.clip_eps, 1 + self.clip_eps
                )
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss  = nn.functional.mse_loss(new_val, ret[mb])
                ent_loss = entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((lp[mb] - new_lp) ** 2).mean().item() * 0.5

                total_pg  += pg_loss.item()
                total_vf  += vf_loss.item()
                total_ent += ent_loss.item()
                total_kl  += approx_kl
                num_updates += 1

        self.buffer.clear()
        n = max(num_updates, 1)
        return {
            "pg_loss":   total_pg  / n,
            "vf_loss":   total_vf  / n,
            "entropy":   total_ent / n,
            "approx_kl": total_kl  / n,
        }

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------

    def save(self, path: str, metadata: dict = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_steps":     self.total_steps,
        }
        if metadata:
            ckpt.update(metadata)
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.total_steps = ckpt.get("total_steps", 0)
        print(f"[PPOAgent] Loaded checkpoint: {path}  "
              f"steps={self.total_steps}")
        return ckpt
