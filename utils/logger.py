"""
日志工具模块
提供训练过程的 CSV 日志记录功能
"""

import os
import csv
from typing import Optional


class Logger:
    """
    训练日志记录器，将指标写入 CSV 文件

    记录字段:
        episode, reward, score, loss, epsilon, eval_score

    使用::

        logger = Logger("logs/train.csv")
        logger.write(ep=1, reward=5.2, score=3, loss=0.01,
                     epsilon=0.9, eval_score=None)
    """

    FIELDS = ["episode", "reward", "score", "loss", "epsilon", "eval_score"]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def write(self, episode: int, reward: float, score: int,
              loss: float, epsilon: float,
              eval_score: Optional[float] = None):
        row = {
            "episode":    episode,
            "reward":     round(reward, 4),
            "score":      score,
            "loss":       round(loss, 6),
            "epsilon":    round(epsilon, 6),
            "eval_score": round(eval_score, 4) if eval_score is not None else "",
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)
