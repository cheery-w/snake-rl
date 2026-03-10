"""
训练器
负责完整的训练循环、评估和检查点管理
"""

import os
import time
import numpy as np
from typing import List, Dict, Tuple

from config import Config
from env.snake_env import SnakeGame
from agent.dqn_agent import DQNAgent
from utils.utils import Logger, moving_average


class Trainer:
    """
    DQN 训练器

    职责:
        1. 执行训练主循环
        2. 定期评估智能体性能
        3. 保存最优模型和周期性 checkpoint
        4. 记录训练日志

    使用::

        cfg     = Config()
        trainer = Trainer(cfg)
        metrics = trainer.train()
    """

    def __init__(self, cfg: Config = None):
        self.cfg   = cfg or Config()
        self.game  = SnakeGame(cfg.GRID_COLS, cfg.GRID_ROWS, cfg.CELL_SIZE)
        self.agent = DQNAgent(cfg)

        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR,        exist_ok=True)
        self.logger = Logger(os.path.join(cfg.LOG_DIR, "train_log.csv"))

        # 历史记录
        self.ep_rewards: List[float] = []
        self.ep_scores:  List[int]   = []
        self.ep_losses:  List[float] = []
        self.eval_scores: List[float] = []

        self.best_eval_score = -np.inf

    # ------------------------------------------------------------------
    # 主训练入口
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, list]:
        """
        执行完整训练流程

        Returns:
            metrics dict 包含 rewards / scores / losses / eval_scores
        """
        cfg        = self.cfg
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"  贪吃蛇 DQN 训练开始")
        print(f"  Episodes : {cfg.NUM_EPISODES}")
        print(f"  Warmup   : {cfg.WARMUP_STEPS} steps")
        print(f"  Device   : {self.agent.device}")
        print(f"{'='*60}\n")

        # 热身阶段：随机动作填充经验缓冲区
        self._warmup()

        for ep in range(1, cfg.NUM_EPISODES + 1):
            ep_reward, ep_loss = self._run_episode()

            self.ep_rewards.append(ep_reward)
            self.ep_scores.append(self.game.score)
            self.ep_losses.append(ep_loss)

            # ε 衰减
            self.agent.decay_epsilon()

            # 周期性评估
            eval_score = None
            if ep % cfg.EVAL_FREQ == 0:
                eval_score = self._evaluate()
                self.eval_scores.append(eval_score)

                # 保存最优模型
                if eval_score > self.best_eval_score:
                    self.best_eval_score = eval_score
                    self.agent.save(
                        os.path.join(cfg.CHECKPOINT_DIR, "best.pt"),
                        metadata={"episode": ep, "eval_score": eval_score},
                    )

            # 定期保存 checkpoint
            if ep % cfg.SAVE_FREQ == 0:
                self.agent.save(
                    os.path.join(cfg.CHECKPOINT_DIR, f"ep_{ep:05d}.pt"),
                    metadata={"episode": ep},
                )

            # 日志记录
            self.logger.write(ep, ep_reward, self.game.score,
                              ep_loss, self.agent.epsilon, eval_score)

            # 控制台输出
            if ep % 50 == 0:
                self._print_progress(ep, start_time, eval_score)

        print(f"\n训练完成  最优评估分数: {self.best_eval_score:.2f}")
        return {
            "rewards":     self.ep_rewards,
            "scores":      self.ep_scores,
            "losses":      self.ep_losses,
            "eval_scores": self.eval_scores,
        }

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int = None,
                 render: bool = False) -> Tuple[float, float]:
        """
        评估当前策略（无探索，ε=0）

        Args:
            n_episodes: 评估轮数，默认使用 cfg.EVAL_EPISODES
            render:     是否可视化（需要 pygame）

        Returns:
            (avg_score, avg_length)
        """
        n  = n_episodes or self.cfg.EVAL_EPISODES
        renderer = None

        if render:
            from env.render import Renderer
            renderer = Renderer(self.game, fps=self.cfg.FPS, title="Snake AI - Evaluate")

        scores, lengths = [], []
        for ep in range(n):
            state = self.game.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.agent.select_action(state, deterministic=True)
                state, reward, done, _ = self.game.step(action)
                total_reward += reward
                if renderer:
                    renderer.render(episode=ep + 1, total_reward=total_reward)
            scores.append(self.game.score)
            lengths.append(self.game.snake_length)

        if renderer:
            renderer.close()

        avg_score  = float(np.mean(scores))
        avg_length = float(np.mean(lengths))
        print(f"[Eval] {n} episodes  Avg Score={avg_score:.2f}  "
              f"Avg Length={avg_length:.2f}  "
              f"Max Score={max(scores)}")
        return avg_score, avg_length

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _warmup(self):
        """用随机动作填充经验缓冲区"""
        cfg   = self.cfg
        state = self.game.reset()
        step  = 0
        print(f"热身中（{cfg.WARMUP_STEPS} 步）...", end=" ", flush=True)
        while step < cfg.WARMUP_STEPS:
            action = np.random.randint(cfg.ACTION_SIZE)
            next_state, reward, done, _ = self.game.step(action)
            self.agent.store(state, action, reward, next_state, done)
            state = next_state
            step += 1
            if done:
                state = self.game.reset()
        print("完成")

    def _run_episode(self) -> Tuple[float, float]:
        """运行一轮训练，返回 (total_reward, avg_loss)"""
        cfg        = self.cfg
        state      = self.game.reset()
        total_reward = 0.0
        losses       = []

        for _ in range(cfg.MAX_STEPS_EP):
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.game.step(action)
            self.agent.store(state, action, reward, next_state, done)
            loss = self.agent.learn()
            if loss > 0:
                losses.append(loss)
            total_reward += reward
            state = next_state
            if done:
                break

        avg_loss = float(np.mean(losses)) if losses else 0.0
        return total_reward, avg_loss

    def _evaluate(self) -> float:
        """内部评估（不渲染），返回平均分数"""
        cfg = self.cfg
        scores = []
        for _ in range(cfg.EVAL_EPISODES):
            state = self.game.reset()
            done  = False
            while not done:
                action = self.agent.select_action(state, deterministic=True)
                state, _, done, _ = self.game.step(action)
            scores.append(self.game.score)
        return float(np.mean(scores))

    def _print_progress(self, ep: int, start_time: float,
                        eval_score=None):
        """打印训练进度到控制台"""
        elapsed   = time.time() - start_time
        window    = min(50, len(self.ep_rewards))
        avg_r     = np.mean(self.ep_rewards[-window:])
        avg_s     = np.mean(self.ep_scores[-window:])
        avg_loss  = np.mean([l for l in self.ep_losses[-window:] if l > 0] or [0])

        eval_str  = f"  EvalScore={eval_score:.2f}" if eval_score is not None else ""
        print(
            f"Ep {ep:5d}/{self.cfg.NUM_EPISODES}  "
            f"AvgR={avg_r:7.2f}  AvgScore={avg_s:.2f}  "
            f"Loss={avg_loss:.4f}  ε={self.agent.epsilon:.4f}"
            f"{eval_str}  "
            f"Time={elapsed/60:.1f}min"
        )
