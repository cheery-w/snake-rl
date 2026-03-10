"""
工具函数
包含随机种子设置、训练曲线绘制、CSV 日志记录
"""

import os
import csv
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 无 GUI 后端，支持服务器环境
import matplotlib.pyplot as plt
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────
# 随机种子
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """全局固定随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────
# 滑动平均
# ──────────────────────────────────────────────────────────────────────

def moving_average(data: List[float], window: int = 50) -> np.ndarray:
    """计算滑动平均（用于平滑训练曲线）"""
    if len(data) == 0:
        return np.array([])
    kernel = np.ones(window) / window
    # 使用 'same' 模式，输出长度与输入相同
    return np.convolve(data, kernel, mode="full")[:len(data)]


# ──────────────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(
    rewards:     List[float],
    scores:      List[int],
    losses:      List[float],
    eval_scores: List[float],
    save_path:   str = "logs/training_curves.png",
    show:        bool = False,
):
    """
    绘制并保存完整的训练曲线图

    四个子图:
        1. 每轮总奖励 + 滑动均值
        2. 每轮游戏得分 + 滑动均值
        3. 每轮平均 Loss
        4. 评估得分曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Snake DQN Training Curves", fontsize=14, fontweight="bold")

    episodes = np.arange(1, len(rewards) + 1)

    # ── 子图 1：奖励 ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color="#4C72B0", label="Reward")
    if len(rewards) >= 10:
        ax.plot(episodes, moving_average(rewards, 50),
                color="#4C72B0", linewidth=2, label="MA-50")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 子图 2：得分 ─────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(episodes, scores, alpha=0.3, color="#DD8452", label="Score")
    if len(scores) >= 10:
        ax.plot(episodes, moving_average(scores, 50),
                color="#DD8452", linewidth=2, label="MA-50")
    ax.set_title("Game Score (foods eaten)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 子图 3：Loss ─────────────────────────────────────────────────
    ax = axes[1, 0]
    valid_losses = [(i + 1, l) for i, l in enumerate(losses) if l > 0]
    if valid_losses:
        xs, ys = zip(*valid_losses)
        ax.plot(xs, ys, alpha=0.4, color="#55A868", label="Loss")
        ax.plot(xs, moving_average(list(ys), min(50, len(ys))),
                color="#55A868", linewidth=2, label="MA-50")
    ax.set_title("Training Loss (Huber)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 子图 4：评估得分 ──────────────────────────────────────────────
    ax = axes[1, 1]
    if eval_scores:
        eval_eps = np.linspace(1, len(rewards), len(eval_scores))
        ax.plot(eval_eps, eval_scores, marker="o", markersize=4,
                color="#C44E52", label="Eval Score")
        ax.axhline(max(eval_scores), linestyle="--", color="gray",
                   alpha=0.6, label=f"Best={max(eval_scores):.1f}")
    ax.set_title("Evaluation Score")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Score")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] 训练曲线已保存至 {save_path}")
    if show:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# CSV 日志记录器
# ──────────────────────────────────────────────────────────────────────

class Logger:
    """
    轻量级 CSV 日志记录器

    记录每轮的: episode, reward, score, loss, epsilon, eval_score
    """

    FIELDS = ["episode", "reward", "score", "loss", "epsilon", "eval_score"]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # 写入表头
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def write(self, episode: int, reward: float, score: int,
              loss: float, epsilon: float, eval_score: Optional[float]):
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


# ──────────────────────────────────────────────────────────────────────
# 其他工具
# ──────────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """将秒数格式化为可读字符串"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"
