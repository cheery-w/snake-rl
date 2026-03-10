"""
评估器
独立评估模块，用于对训练好的模型进行系统性性能评估
"""

import os
import sys
import numpy as np
from typing import Dict, Optional

from config import Config
from env.snake_env import SnakeGame
from agent.dqn_agent import DQNAgent


class Evaluator:
    """
    智能体性能评估器

    功能:
        - 加载已训练的模型
        - 运行多局游戏，统计关键指标
        - 可选渲染可视化
        - 生成评估报告

    使用::

        evaluator = Evaluator(cfg)
        evaluator.load("checkpoints/best.pt")
        metrics = evaluator.evaluate(n_episodes=100, render=False)
        evaluator.print_report(metrics)
    """

    def __init__(self, cfg: Config = None):
        self.cfg   = cfg or Config()
        self.game  = SnakeGame(self.cfg.GRID_COLS, self.cfg.GRID_ROWS,
                               self.cfg.CELL_SIZE)
        self.agent = DQNAgent(self.cfg)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def load(self, path: str):
        """加载模型 checkpoint"""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint 不存在: {path}")
        self.agent.load(path)

    def evaluate(
        self,
        n_episodes: int = 100,
        render:     bool = False,
        fps:        Optional[int] = None,
    ) -> Dict[str, float]:
        """
        运行 n_episodes 局评估，返回汇总指标

        Args:
            n_episodes: 评估轮数
            render:     是否可视化（需要 pygame）
            fps:        渲染帧率（render=True 时有效）

        Returns:
            metrics dict::

                {
                    "avg_score":  平均得分,
                    "max_score":  最高得分,
                    "min_score":  最低得分,
                    "avg_length": 平均蛇长,
                    "max_length": 最大蛇长,
                    "avg_steps":  平均存活步数,
                }
        """
        renderer = None
        if render:
            from env.render import Renderer
            renderer = Renderer(
                self.game,
                fps=fps or self.cfg.FPS,
                title="Snake AI - Evaluate",
            )

        scores, lengths, steps_list = [], [], []

        for ep in range(n_episodes):
            state = self.game.reset()
            done  = False
            ep_steps     = 0
            total_reward = 0.0

            while not done:
                action = self.agent.select_action(state, deterministic=True)
                state, reward, done, _ = self.game.step(action)
                total_reward += reward
                ep_steps     += 1
                if renderer:
                    renderer.render(episode=ep + 1, total_reward=total_reward)

            scores.append(self.game.score)
            lengths.append(self.game.snake_length)
            steps_list.append(ep_steps)

        if renderer:
            renderer.close()

        metrics = {
            "avg_score":  float(np.mean(scores)),
            "max_score":  float(np.max(scores)),
            "min_score":  float(np.min(scores)),
            "avg_length": float(np.mean(lengths)),
            "max_length": float(np.max(lengths)),
            "avg_steps":  float(np.mean(steps_list)),
        }
        return metrics

    def print_report(self, metrics: Dict[str, float], n_episodes: int = 0):
        """打印格式化评估报告"""
        ep_str = f"  ({n_episodes} 轮)" if n_episodes else ""
        print(f"\n{'='*50}")
        print(f"  评估报告{ep_str}")
        print(f"{'='*50}")
        print(f"  平均得分  : {metrics['avg_score']:.2f}")
        print(f"  最高得分  : {metrics['max_score']:.0f}")
        print(f"  最低得分  : {metrics['min_score']:.0f}")
        print(f"  平均蛇长  : {metrics['avg_length']:.2f}")
        print(f"  最大蛇长  : {metrics['max_length']:.0f}")
        print(f"  平均步数  : {metrics['avg_steps']:.1f}")
        print(f"{'='*50}\n")
