"""
经验回放缓冲区
提供标准均匀采样 ReplayBuffer
"""

import numpy as np
import random
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """
    均匀经验回放缓冲区

    存储五元组 (state, action, reward, next_state, done)
    随机采样打破时序相关性，稳定 DQN 训练

    Args:
        capacity: 缓冲区最大容量（超出时自动丢弃最旧样本）
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """压入一条经验"""
        self.buffer.append((
            np.asarray(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        随机采样一个 batch

        Returns:
            states      (B, S)
            actions     (B,)
            rewards     (B,)
            next_states (B, S)
            dones       (B,)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.stack(next_states),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        """缓冲区中是否有足够样本（至少 1 条）"""
        return len(self) > 0
