"""
模型评估入口
加载训练好的 checkpoint，统计评估指标

用法:
    python evaluate.py --model checkpoints/best.pt
    python evaluate.py --model checkpoints/best.pt --episodes 50 --render
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from env.snake_env import SnakeGame
from agent.dqn_agent import DQNAgent
from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Snake DQN 评估")
    parser.add_argument("--model",    type=str, required=True,
                        help="checkpoint 文件路径")
    parser.add_argument("--episodes", type=int, default=50,
                        help="评估轮数")
    parser.add_argument("--render",   action="store_true",
                        help="开启 pygame 可视化")
    parser.add_argument("--fps",      type=int, default=20,
                        help="可视化帧率（仅 --render 时有效）")
    parser.add_argument("--seed",     type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.isfile(args.model):
        print(f"[Error] 模型文件不存在: {args.model}")
        sys.exit(1)

    cfg   = Config()
    cfg.FPS = args.fps

    game  = SnakeGame(cfg.GRID_COLS, cfg.GRID_ROWS, cfg.CELL_SIZE)
    agent = DQNAgent(cfg)
    ckpt  = agent.load(args.model)

    episode_info = ckpt.get("episode", "?")
    eval_score   = ckpt.get("eval_score", "?")
    print(f"[Info] Checkpoint: episode={episode_info}  "
          f"saved_eval_score={eval_score}")

    renderer = None
    if args.render:
        from env.render import Renderer
        renderer = Renderer(game, fps=cfg.FPS, title="Snake AI - Evaluate")

    scores, lengths, step_counts = [], [], []

    for ep in range(1, args.episodes + 1):
        state        = game.reset()
        done         = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, done, _ = game.step(action)
            total_reward += reward
            if renderer:
                renderer.render(episode=ep, total_reward=total_reward)

        scores.append(game.score)
        lengths.append(game.snake_length)
        step_counts.append(game.steps)

        if ep % 10 == 0:
            print(f"  Ep {ep:3d}  score={game.score}  "
                  f"length={game.snake_length}  steps={game.steps}")

    if renderer:
        renderer.close()

    # ── 汇总统计 ──────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  评估结果  ({args.episodes} episodes)")
    print(f"{'='*50}")
    print(f"  平均得分   : {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"  最高得分   : {max(scores)}")
    print(f"  平均蛇身长 : {np.mean(lengths):.2f}")
    print(f"  平均存活步 : {np.mean(step_counts):.1f}")
    print(f"  得分 >= 10 : {sum(s >= 10 for s in scores)}/{args.episodes} "
          f"({100*sum(s>=10 for s in scores)/args.episodes:.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
