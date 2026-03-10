"""
多智能体对战入口

用法:
    python multi_agent.py                         # 4 智能体观战演示
    python multi_agent.py --agents 2              # 2 智能体
    python multi_agent.py --train                 # 训练模式
    python multi_agent.py --train --agents 2 --episodes 5000
    python multi_agent.py --fps 5                 # 慢速演示
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from config import Config
from env.multi_agent_env import MultiAgentEnv
from agent.dqn_agent import DQNAgent
from ui._multi_render import MultiAgentRenderer


def parse_args():
    p = argparse.ArgumentParser(description="贪吃蛇多智能体对战")
    p.add_argument("--agents",   type=int,  default=4,
                   help="智能体数量 (2-4)")
    p.add_argument("--train",    action="store_true",
                   help="进入训练模式（不渲染）")
    p.add_argument("--episodes", type=int,  default=2000,
                   help="训练轮数（train 模式）")
    p.add_argument("--fps",      type=int,  default=15,
                   help="演示帧率")
    p.add_argument("--rounds",   type=int,  default=0,
                   help="演示轮数（0 = 无限）")
    p.add_argument("--no-load",  action="store_true",
                   help="不加载预训练权重（随机初始化）")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# 演示模式（有 pygame 渲染）
# ──────────────────────────────────────────────────────────────────────

def run_demo(args):
    cfg    = Config()
    n      = max(2, min(4, args.agents))
    env    = MultiAgentEnv(n_agents=n,
                           cols=cfg.GRID_COLS,
                           rows=cfg.GRID_ROWS,
                           cell=cfg.CELL_SIZE)
    agents = _load_agents(n, cfg, not args.no_load)

    renderer  = MultiAgentRenderer(env, fps=args.fps)
    round_num = 0

    print(f"多智能体演示  {n} 条蛇  按 Q 退出")
    try:
        while True:
            round_num += 1
            states = env.reset()
            while not env.all_done:
                actions = [
                    agents[i].select_action(states[i], deterministic=True)
                    for i in range(n)
                ]
                states, _, info = env.step(actions)
                renderer.render(round_num, info["scores"])

            scores = [s.score for s in env.snakes]
            winner = max(range(n), key=lambda i: scores[i])
            print(f"Round {round_num:4d}  Scores={scores}  "
                  f"Winner=Agent{winner}")

            if args.rounds > 0 and round_num >= args.rounds:
                break
            time.sleep(0.5)

    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        pygame.quit()


# ──────────────────────────────────────────────────────────────────────
# 训练模式
# ──────────────────────────────────────────────────────────────────────

def run_train(args):
    from trainer.multi_agent_trainer import MultiAgentTrainer
    from utils.utils import set_seed

    cfg              = Config()
    cfg.NUM_EPISODES = args.episodes
    set_seed(cfg.SEED)

    n       = max(2, min(4, args.agents))
    trainer = MultiAgentTrainer(cfg, n_agents=n)
    metrics = trainer.train()

    print("\n训练完成")
    for i in range(n):
        key = f"agent_{i}_scores"
        if key in metrics:
            import numpy as np
            scores = metrics[key]
            print(f"  Agent{i}  最终均分={np.mean(scores[-100:]):.2f}  "
                  f"最高={max(scores)}")


# ──────────────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────────────

def _load_agents(n: int, cfg: Config, load: bool) -> list:
    """创建并尝试加载各智能体权重"""
    diffs = ["easy", "medium", "hard", "expert"]
    agents = []
    for i in range(n):
        agent = DQNAgent(cfg)
        if load:
            paths = [
                f"checkpoints/best_agent{i}.pt",
                f"checkpoints/{diffs[i % 4]}.pt",
                "checkpoints/best.pt",
            ]
            for p in paths:
                if os.path.isfile(p):
                    try:
                        agent.load(p)
                    except Exception:
                        pass
                    break
        agents.append(agent)
    return agents


# ──────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.train:
        run_train(args)
    else:
        run_demo(args)


if __name__ == "__main__":
    main()
