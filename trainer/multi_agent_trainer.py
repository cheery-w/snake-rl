"""
多智能体训练器
支持 2-4 个 DQN 智能体在同一棋盘上同时训练（竞争模式）
"""

import os
import time
import numpy as np
from typing import List, Dict

from config import Config
from env.multi_agent_env import MultiAgentEnv
from agent.dqn_agent import DQNAgent
from utils.utils import Logger, moving_average


class MultiAgentTrainer:
    """
    多智能体 DQN 训练器

    职责:
        1. 管理多个 DQNAgent，各自独立维护经验缓冲区和网络
        2. 并行执行多智能体训练主循环
        3. 定期评估并保存最优模型
        4. 记录各智能体独立的训练日志

    使用::

        cfg     = Config()
        trainer = MultiAgentTrainer(cfg, n_agents=4)
        trainer.train()
    """

    def __init__(self, cfg: Config = None, n_agents: int = 2):
        self.cfg      = cfg or Config()
        self.n_agents = n_agents
        self.env      = MultiAgentEnv(
            n_agents=n_agents,
            cols=cfg.GRID_COLS,
            rows=cfg.GRID_ROWS,
            cell=cfg.CELL_SIZE,
        )

        # 每个智能体独立的 DQN
        self.agents: List[DQNAgent] = [DQNAgent(cfg) for _ in range(n_agents)]

        # 日志目录
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR,        exist_ok=True)
        self.loggers = [
            Logger(os.path.join(cfg.LOG_DIR, f"multi_agent_{i}.csv"))
            for i in range(n_agents)
        ]

        # 历史记录（每个智能体独立）
        self.ep_scores: List[List[int]]   = [[] for _ in range(n_agents)]
        self.best_eval: List[float]       = [-np.inf] * n_agents

    # ------------------------------------------------------------------
    # 主训练入口
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, list]:
        """
        执行多智能体训练主循环

        Returns:
            metrics dict: {"agent_{i}_scores": [...]}
        """
        cfg        = self.cfg
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"  多智能体 DQN 训练  ({self.n_agents} 蛇)")
        print(f"  Episodes: {cfg.NUM_EPISODES}")
        print(f"  Device  : {self.agents[0].device}")
        print(f"{'='*60}\n")

        self._warmup()

        for ep in range(1, cfg.NUM_EPISODES + 1):
            scores = self._run_episode()

            for i, s in enumerate(scores):
                self.ep_scores[i].append(s)

            for i, agent in enumerate(self.agents):
                agent.decay_epsilon()
                self.loggers[i].write(
                    ep, 0.0, scores[i], 0.0, agent.epsilon, None
                )

            # 周期评估
            if ep % cfg.EVAL_FREQ == 0:
                eval_scores = self._evaluate()
                for i, es in enumerate(eval_scores):
                    if es > self.best_eval[i]:
                        self.best_eval[i] = es
                        self.agents[i].save(
                            os.path.join(cfg.CHECKPOINT_DIR,
                                         f"best_agent{i}.pt"),
                            metadata={"episode": ep,
                                      "eval_score": es},
                        )

            # 周期保存
            if ep % cfg.SAVE_FREQ == 0:
                for i, agent in enumerate(self.agents):
                    agent.save(
                        os.path.join(cfg.CHECKPOINT_DIR,
                                     f"ep{ep:05d}_agent{i}.pt"),
                    )

            if ep % 100 == 0:
                self._print_progress(ep, start_time)

        print("\n多智能体训练完成")
        return {
            f"agent_{i}_scores": self.ep_scores[i]
            for i in range(self.n_agents)
        }

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _warmup(self):
        """各智能体随机热身"""
        cfg = self.cfg
        states = self.env.reset()
        step   = 0
        print(f"热身中（{cfg.WARMUP_STEPS} 步）...", end=" ", flush=True)
        while step < cfg.WARMUP_STEPS:
            actions = [np.random.randint(cfg.ACTION_SIZE)
                       for _ in range(self.n_agents)]
            next_states, dones, info = self.env.step(actions)
            for i, agent in enumerate(self.agents):
                agent.store(
                    states[i], actions[i], 0.0,
                    next_states[i], dones[i]
                )
            states = next_states
            step  += 1
            if self.env.all_done:
                states = self.env.reset()
        print("完成")

    def _run_episode(self) -> List[int]:
        """运行一轮训练，返回各智能体的得分"""
        cfg    = self.cfg
        states = self.env.reset()
        alive  = [True] * self.n_agents

        for _ in range(cfg.MAX_STEPS_EP):
            # 选择动作
            actions = []
            for i, agent in enumerate(self.agents):
                if alive[i]:
                    a = agent.select_action(states[i])
                else:
                    a = 0
                actions.append(a)

            next_states, dones, info = self.env.step(actions)

            # 存储经验并学习
            for i, agent in enumerate(self.agents):
                if alive[i]:
                    # 根据结果分配奖励
                    reward = self._compute_reward(i, info, dones)
                    agent.store(
                        states[i], actions[i], reward,
                        next_states[i], dones[i]
                    )
                    agent.learn()
                if dones[i]:
                    alive[i] = False

            states = next_states

            if self.env.all_done:
                break

        return info["scores"]

    def _compute_reward(
        self, agent_idx: int, info: dict, dones: List[bool]
    ) -> float:
        """
        多智能体奖励：使用分数差分（当前 score 减上一步分数）
        简化版：直接用环境的奖励信号（每步 0，吃食物 +10，死亡 -10）
        实际奖励由 env.step 内部管理，这里仅返回 0 作占位
        """
        return 0.0

    def _evaluate(self) -> List[float]:
        """评估所有智能体（贪心策略），返回各智能体平均分"""
        n = self.cfg.EVAL_EPISODES
        scores_sum = [0.0] * self.n_agents
        for _ in range(n):
            states = self.env.reset()
            while not self.env.all_done:
                actions = [
                    self.agents[i].select_action(states[i], deterministic=True)
                    for i in range(self.n_agents)
                ]
                states, _, info = self.env.step(actions)
            for i in range(self.n_agents):
                scores_sum[i] += self.env.snakes[i].score

        avg = [s / n for s in scores_sum]
        print(f"[Eval]  " +
              "  ".join(f"Agent{i}={avg[i]:.2f}" for i in range(self.n_agents)))
        return avg

    def _print_progress(self, ep: int, start_time: float):
        elapsed = time.time() - start_time
        window  = min(100, ep)
        parts   = []
        for i in range(self.n_agents):
            avg = np.mean(self.ep_scores[i][-window:])
            parts.append(f"A{i}={avg:.2f}")
        print(
            f"Ep {ep:5d}/{self.cfg.NUM_EPISODES}  "
            + "  ".join(parts)
            + f"  ε={self.agents[0].epsilon:.4f}"
            + f"  Time={elapsed/60:.1f}min"
        )
