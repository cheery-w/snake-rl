"""
自定义规则游戏入口

用法:
    python custom_game.py                              # 默认：多食物 + 无障碍
    python custom_game.py --food 3 --obs border        # 3 食物 + 内边框墙
    python custom_game.py --food 2 --obs cross         # 2 食物 + 十字障碍
    python custom_game.py --human                      # 人类操控
    python custom_game.py --model checkpoints/best.pt  # AI 演示
    python custom_game.py --cols 30 --rows 30          # 大地图
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from config import Config
from env.custom_env import CustomEnv
from env.snake_env import Direction, CLOCKWISE


# ── 颜色 ──────────────────────────────────────────────────────────────
_BG      = (15,  15,  15)
_GRID    = (30,  30,  30)
_HEAD    = (0,   220,  80)
_BODY    = (0,   160,  60)
_FOOD    = (220,  50,  50)
_FOOD2   = (255, 160,  50)   # 第二个食物颜色
_OBS     = (100,  80,  60)   # 障碍物颜色
_TEXT    = (200, 200, 200)

_KEY_MAP = {
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_LEFT:  Direction.LEFT,
    pygame.K_UP:    Direction.UP,
    pygame.K_DOWN:  Direction.DOWN,
    pygame.K_d:     Direction.RIGHT,
    pygame.K_a:     Direction.LEFT,
    pygame.K_w:     Direction.UP,
    pygame.K_s:     Direction.DOWN,
}


def parse_args():
    p = argparse.ArgumentParser(description="自定义规则贪吃蛇")
    p.add_argument("--food",   type=int,  default=2,
                   help="同时存在的食物数量（默认 2）")
    p.add_argument("--obs",    type=str,  default="none",
                   choices=["none", "border", "cross"],
                   help="障碍物类型: none/border/cross")
    p.add_argument("--cols",   type=int,  default=20,
                   help="棋盘列数（默认 20）")
    p.add_argument("--rows",   type=int,  default=20,
                   help="棋盘行数（默认 20）")
    p.add_argument("--fps",    type=int,  default=15,
                   help="游戏帧率（默认 15）")
    p.add_argument("--human",  action="store_true",
                   help="人类手动控制")
    p.add_argument("--model",  type=str,  default=None,
                   help="AI 模型路径（不指定则用 checkpoints/best.pt）")
    p.add_argument("--reward-food",  type=float, default=10.0)
    p.add_argument("--reward-death", type=float, default=-10.0)
    return p.parse_args()


def make_env(args) -> CustomEnv:
    return CustomEnv(
        cols=args.cols,
        rows=args.rows,
        cell=20,
        multi_food=args.food,
        obstacles=args.obs,
        reward_food=args.reward_food,
        reward_death=args.reward_death,
    )


# ──────────────────────────────────────────────────────────────────────
# 渲染器
# ──────────────────────────────────────────────────────────────────────

class CustomRenderer:
    def __init__(self, env: CustomEnv, fps: int):
        pygame.init()
        self.env  = env
        self.fps  = fps
        self.cell = env.cell
        self.w    = env.width
        self.h    = env.height

        self.screen = pygame.display.set_mode((self.w, self.h + 30))
        pygame.display.set_caption("Snake — Custom Rules")
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("Consolas", 16)

    def render(self, episode: int = 0, score: int = 0):
        self._handle_events()
        self.screen.fill(_BG)
        self._draw_grid()
        self._draw_obstacles()
        self._draw_foods()
        self._draw_snake()
        self._draw_info(episode, score)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    raise SystemExit

    def _draw_grid(self):
        for x in range(0, self.w, self.cell):
            pygame.draw.line(self.screen, _GRID, (x, 0), (x, self.h))
        for y in range(0, self.h, self.cell):
            pygame.draw.line(self.screen, _GRID, (0, y), (self.w, y))

    def _draw_obstacles(self):
        for obs in self.env.obstacles:
            rect = pygame.Rect(obs[0], obs[1], self.cell, self.cell)
            pygame.draw.rect(self.screen, _OBS, rect)

    def _draw_foods(self):
        food_colors = [_FOOD, _FOOD2, (100, 220, 100), (200, 200, 50)]
        for i, f in enumerate(self.env.food_list):
            col = food_colors[i % len(food_colors)]
            cx  = f[0] + self.cell // 2
            cy  = f[1] + self.cell // 2
            pygame.draw.circle(self.screen, col, (cx, cy), self.cell // 2 - 2)

    def _draw_snake(self):
        snake = list(self.env.snake)
        for i, seg in enumerate(snake):
            color  = _HEAD if i == 0 else _BODY
            margin = 1
            rect   = pygame.Rect(
                seg[0] + margin, seg[1] + margin,
                self.cell - 2 * margin, self.cell - 2 * margin,
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def _draw_info(self, episode: int, score: int):
        info  = f"Ep: {episode}  Score: {score}  Len: {self.env.snake_length}"
        info += f"  Food: {len(self.env.food_list)}  Obs: {self.env.obstacles != []}"
        surf  = self.font.render(info, True, _TEXT)
        self.screen.blit(surf, (6, self.h + 6))


# ──────────────────────────────────────────────────────────────────────
# 游戏运行函数
# ──────────────────────────────────────────────────────────────────────

def run_ai(args):
    model_path = args.model or "checkpoints/best.pt"
    if not os.path.isfile(model_path):
        print(f"[Error] 模型不存在: {model_path}")
        sys.exit(1)

    from agent.dqn_agent import DQNAgent
    cfg   = Config()
    agent = DQNAgent(cfg)
    agent.load(model_path)

    env      = make_env(args)
    renderer = CustomRenderer(env, fps=args.fps)

    round_num  = 0
    best_score = 0
    print(f"AI 自定义游戏  食物数={args.food}  障碍={args.obs}")
    try:
        while True:
            round_num   += 1
            state        = env.reset()
            done         = False
            while not done:
                action = agent.select_action(state, deterministic=True)
                state, _, done, _ = env.step(action)
                renderer.render(round_num, env.score)
            best_score = max(best_score, env.score)
            print(f"Round {round_num:4d}  Score={env.score}  Best={best_score}")
            time.sleep(0.3)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        pygame.quit()


def run_human(args):
    env      = make_env(args)
    renderer = CustomRenderer(env, fps=max(args.fps, 8))

    def dir_to_action(cur: Direction, tgt: Direction) -> int:
        ci = CLOCKWISE.index(cur)
        ti = CLOCKWISE.index(tgt)
        d  = (ti - ci) % 4
        return 0 if d == 0 else 1 if d == 1 else 2 if d == 3 else 0

    state     = env.reset()
    next_dir  = env.direction
    round_num = 1
    done      = False
    paused    = False
    print(f"人类模式  食物={args.food}  障碍={args.obs}  WASD/方向键  Space=暂停  R=重置")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key in _KEY_MAP:
                        tgt = _KEY_MAP[event.key]
                        ci  = CLOCKWISE.index(env.direction)
                        ti  = CLOCKWISE.index(tgt)
                        if (ti - ci) % 4 != 2:
                            next_dir = tgt
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        state = env.reset()
                        next_dir = env.direction
                        done = False
                        round_num += 1
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        raise SystemExit

            if paused or done:
                renderer.render(round_num, env.score)
                continue

            action = dir_to_action(env.direction, next_dir)
            state, _, done, _ = env.step(action)
            renderer.render(round_num, env.score)

            if done:
                print(f"死亡  Score={env.score}  按 R 重新开始")

    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        pygame.quit()


def main_gui(fps: int = 15):
    """由 main.py 调用的 GUI 入口（使用默认参数）"""
    import types
    args = types.SimpleNamespace(
        food=2, obs="none", cols=20, rows=20, fps=fps,
        human=False, model=None,
        reward_food=10.0, reward_death=-10.0,
    )
    run_ai(args) if os.path.isfile("checkpoints/best.pt") else run_human(args)


def main():
    args = parse_args()
    if args.human:
        run_human(args)
    else:
        run_ai(args)


if __name__ == "__main__":
    main()
