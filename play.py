"""
游戏演示入口
用训练好的模型实时可视化贪吃蛇 AI

用法:
    python play.py --model checkpoints/best.pt          # AI 演示
    python play.py --model checkpoints/best.pt --fps 10 # 慢速演示
    python play.py --human                               # 人类手动游玩（键盘控制）
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from config import Config
from env.snake_env import SnakeGame, Direction, CLOCKWISE
from env.render import Renderer


def parse_args():
    parser = argparse.ArgumentParser(description="Snake AI 演示")
    parser.add_argument("--model",  type=str, default=None,
                        help="AI checkpoint 路径（不指定则必须加 --human）")
    parser.add_argument("--fps",    type=int, default=15,
                        help="游戏帧率")
    parser.add_argument("--human",  action="store_true",
                        help="人类手动游玩模式（WASD 或方向键）")
    parser.add_argument("--rounds", type=int, default=0,
                        help="演示轮数（0=无限）")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# AI 演示
# ──────────────────────────────────────────────────────────────────────

def run_ai(args):
    if not os.path.isfile(args.model):
        print(f"[Error] 模型文件不存在: {args.model}")
        sys.exit(1)

    from agent.dqn_agent import DQNAgent
    cfg   = Config()
    cfg.FPS = args.fps

    game  = SnakeGame(cfg.GRID_COLS, cfg.GRID_ROWS, cfg.CELL_SIZE)
    agent = DQNAgent(cfg)
    agent.load(args.model)

    renderer = Renderer(game, fps=cfg.FPS, title="Snake AI - Demo")

    round_num  = 0
    best_score = 0

    print("AI 演示中... 按 Ctrl+C 或关闭窗口退出")
    try:
        while True:
            round_num   += 1
            state        = game.reset()
            done         = False
            total_reward = 0.0

            while not done:
                action = agent.select_action(state, deterministic=True)
                state, reward, done, _ = game.step(action)
                total_reward += reward
                renderer.render(episode=round_num, total_reward=total_reward)

            best_score = max(best_score, game.score)
            print(f"Round {round_num:4d}  "
                  f"Score={game.score}  Best={best_score}  "
                  f"Length={game.snake_length}")

            if args.rounds > 0 and round_num >= args.rounds:
                break

            time.sleep(0.5)  # 回合间短暂停顿

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        renderer.close()
        print(f"\n演示结束  共 {round_num} 轮  最高得分 {best_score}")


# ──────────────────────────────────────────────────────────────────────
# 人类手动游玩
# ──────────────────────────────────────────────────────────────────────

def run_human(args):
    """
    方向键 / WASD 控制，支持暂停（Space）和重启（R）
    """
    cfg  = Config()
    cfg.FPS = max(args.fps, 10)

    game     = SnakeGame(cfg.GRID_COLS, cfg.GRID_ROWS, cfg.CELL_SIZE)
    renderer = Renderer(game, fps=cfg.FPS, title="Snake - Human Mode")

    # 方向键 / WASD → Direction
    KEY_MAP = {
        pygame.K_RIGHT: Direction.RIGHT,
        pygame.K_LEFT:  Direction.LEFT,
        pygame.K_UP:    Direction.UP,
        pygame.K_DOWN:  Direction.DOWN,
        pygame.K_d:     Direction.RIGHT,
        pygame.K_a:     Direction.LEFT,
        pygame.K_w:     Direction.UP,
        pygame.K_s:     Direction.DOWN,
    }

    def direction_to_action(current: Direction, target: Direction) -> int:
        """将绝对方向转换为相对动作（0直/1右/2左）"""
        cur_idx = CLOCKWISE.index(current)
        tgt_idx = CLOCKWISE.index(target)
        diff = (tgt_idx - cur_idx) % 4
        if diff == 0: return 0
        if diff == 1: return 1
        if diff == 3: return 2
        return 0  # 反向（忽略）

    state     = game.reset()
    next_dir  = game.direction
    round_num = 0
    done      = False
    paused    = False

    print("人类模式  WASD/方向键移动  Space=暂停  R=重置  Q=退出")

    try:
        while True:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key in KEY_MAP:
                        target = KEY_MAP[event.key]
                        # 不允许直接反向（如向右时按左）
                        cur_idx = CLOCKWISE.index(game.direction)
                        tgt_idx = CLOCKWISE.index(target)
                        if (tgt_idx - cur_idx) % 4 != 2:
                            next_dir = target
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        if paused:
                            print("⏸  已暂停（按 Space 继续）")
                    elif event.key in (pygame.K_r,):
                        state    = game.reset()
                        next_dir = game.direction
                        done     = False
                        round_num += 1
                        print(f"重置（第 {round_num+1} 局）")
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        raise SystemExit

            if paused or done:
                renderer.render(episode=round_num + 1, total_reward=0)
                continue

            action = direction_to_action(game.direction, next_dir)
            state, _, done, _ = game.step(action)
            renderer.render(episode=round_num + 1, total_reward=float(game.score))

            if done:
                print(f"游戏结束  Score={game.score}  Length={game.snake_length}  "
                      f"按 R 重新开始")

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        renderer.close()


# ──────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.human:
        run_human(args)
    elif args.model:
        run_ai(args)
    else:
        print("[Error] 请指定 --model <path> 或 --human")
        sys.exit(1)


if __name__ == "__main__":
    main()
