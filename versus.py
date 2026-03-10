"""
人机对战入口

支持两种对战模式:
    simultaneous (同时对战): 人类与 AI 在同一棋盘上同时游戏，争夺食物
    turns        (轮流对战): 人类与 AI 交替游戏，比较各自得分

用法:
    python versus.py --model checkpoints/best.pt
    python versus.py --model checkpoints/best.pt --mode simultaneous
    python versus.py --model checkpoints/best.pt --mode turns --rounds 3
    python versus.py --model checkpoints/best.pt --fps 8
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
import numpy as np

from config import Config
from env.snake_env import SnakeGame, Direction, CLOCKWISE
from env.versus_env import VersusEnv
from agent.dqn_agent import DQNAgent


# ── 颜色常量 ──────────────────────────────────────────────────────────

_BG        = (15,  15,  15)
_GRID      = (35,  35,  35)
_AI_HEAD   = (0,   220, 80)
_AI_BODY   = (0,   160, 60)
_HU_HEAD   = (50,  130, 255)
_HU_BODY   = (30,  85,  200)
_FOOD      = (220, 50,  50)
_TEXT      = (220, 220, 220)
_DIM       = (100, 100, 100)
_GOLD      = (255, 215, 0)

# 键盘映射（同时对战和轮流对战共用）
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


# ══════════════════════════════════════════════════════════════════════
# 同时对战渲染器
# ══════════════════════════════════════════════════════════════════════

class VersusRenderer:
    """
    同时对战专用渲染器

    在同一棋盘上以不同颜色显示两条蛇:
        绿色 = AI    蓝色 = 玩家
    """

    def __init__(self, env: VersusEnv, fps: int = 10):
        pygame.init()
        self.env   = env
        self.fps   = fps
        self.cell  = env.cell
        self.w     = env.width
        self.h     = env.height

        self.screen = pygame.display.set_mode((self.w, self.h + 60))
        pygame.display.set_caption("Snake — Human vs AI")
        self.clock  = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 15)
        self.font_m = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)

    def render(self, round_num: int = 1):
        """渲染当前帧"""
        self.screen.fill(_BG)
        self._draw_grid()
        self._draw_food()
        self._draw_snake(self.env.snake1, _AI_HEAD, _AI_BODY,
                         self.env.done1, label="A")
        self._draw_snake(self.env.snake2, _HU_HEAD, _HU_BODY,
                         self.env.done2, label="Y")
        self._draw_hud(round_num)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def show_result(self, message: str):
        """在画面上叠加显示对战结果"""
        overlay = pygame.Surface((self.w, self.h + 60), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        msg_surf = self.font_l.render(message, True, _GOLD)
        sub_surf = self.font_s.render(
            "按任意键继续  /  Q 退出", True, _TEXT)
        self.screen.blit(msg_surf,
                         msg_surf.get_rect(center=(self.w // 2,
                                                   self.h // 2 - 22)))
        self.screen.blit(sub_surf,
                         sub_surf.get_rect(center=(self.w // 2,
                                                   self.h // 2 + 18)))
        pygame.display.flip()

    def close(self):
        pygame.quit()

    # ── 私有绘制方法 ────────────────────────────────────────────────

    def _draw_grid(self):
        for x in range(0, self.w, self.cell):
            pygame.draw.line(self.screen, _GRID, (x, 0), (x, self.h))
        for y in range(0, self.h, self.cell):
            pygame.draw.line(self.screen, _GRID, (0, y), (self.w, y))

    def _draw_food(self):
        fx, fy = self.env.food
        cx = fx + self.cell // 2
        cy = fy + self.cell // 2
        pygame.draw.circle(self.screen, _FOOD, (cx, cy),
                           self.cell // 2 - 2)

    def _draw_snake(self, snake, head_col, body_col, dead: bool, label: str):
        segs = list(snake)
        for i, seg in enumerate(segs):
            color = head_col if i == 0 else body_col
            if dead:
                color = tuple(max(0, c - 100) for c in color)
            m = 1
            rect = pygame.Rect(seg[0] + m, seg[1] + m,
                                self.cell - 2 * m, self.cell - 2 * m)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # 头部标签
        if segs:
            hx = segs[0][0] + self.cell // 2
            hy = segs[0][1] + self.cell // 2
            lbl = self.font_s.render(label, True, (0, 0, 0))
            self.screen.blit(lbl, lbl.get_rect(center=(hx, hy)))

    def _draw_hud(self, round_num: int):
        """底部 HUD 信息栏"""
        y0 = self.h + 5

        # AI 分数（左）
        ai_col  = _AI_HEAD if not self.env.done1 else _DIM
        ai_surf = self.font_m.render(f"AI: {self.env.score1}", True, ai_col)
        self.screen.blit(ai_surf, (10, y0))

        # 中间：回合 + 操作提示
        rd_surf = self.font_s.render(f"Round {round_num}", True, _DIM)
        ht_surf = self.font_s.render(
            "WASD/Arrow=Move  Space=Pause  Q=Quit", True, _DIM)
        self.screen.blit(rd_surf,
                         rd_surf.get_rect(center=(self.w // 2, y0 + 8)))
        self.screen.blit(ht_surf,
                         ht_surf.get_rect(center=(self.w // 2, y0 + 28)))

        # 玩家分数（右）
        hu_col  = _HU_HEAD if not self.env.done2 else _DIM
        hu_surf = self.font_m.render(f"You: {self.env.score2}", True, hu_col)
        self.screen.blit(hu_surf,
                         hu_surf.get_rect(right=self.w - 10, top=y0))


# ══════════════════════════════════════════════════════════════════════
# 模式 1：同时对战
# ══════════════════════════════════════════════════════════════════════

def run_simultaneous(args, agent: DQNAgent, cfg: Config):
    """
    同时对战主循环

    人类（蓝色）与 AI（绿色）在同一棋盘上竞争，
    先全部死亡时结束，按得分判定胜负
    """
    env      = VersusEnv(cfg.GRID_COLS, cfg.GRID_ROWS, cfg.CELL_SIZE)
    renderer = VersusRenderer(env, fps=args.fps)

    wins_ai    = 0
    wins_human = 0
    draws      = 0
    round_num  = 0

    print("\n同时对战模式")
    print("  绿色 A = AI    蓝色 Y = 你")
    print("  WASD / 方向键 控制    Space=暂停    Q=退出\n")

    try:
        while True:
            round_num += 1
            state_ai, _ = env.reset()
            next_dir    = env.dir2   # 人类蛇初始方向
            paused      = False

            # ── 回合主循环 ────────────────────────────────────────
            while not (env.done1 and env.done2):
                # 事件处理
                quit_game = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_game = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key in _KEY_MAP:
                            target  = _KEY_MAP[event.key]
                            cur_idx = CLOCKWISE.index(env.dir2)
                            tgt_idx = CLOCKWISE.index(target)
                            if (tgt_idx - cur_idx) % 4 != 2:
                                next_dir = target
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            quit_game = True

                if quit_game:
                    raise SystemExit

                if paused:
                    renderer.render(round_num)
                    continue

                # AI 动作（仅在存活时计算新状态）
                if not env.done1:
                    state_ai  = env._build_state(
                        env.head1, env.dir1, env.snake1, env.snake2)
                    action_ai = agent.select_action(state_ai, deterministic=True)
                else:
                    action_ai = 0

                # 人类动作（绝对方向 → 相对动作）
                if not env.done2:
                    cur_idx = CLOCKWISE.index(env.dir2)
                    tgt_idx = CLOCKWISE.index(next_dir)
                    diff    = (tgt_idx - cur_idx) % 4
                    action_human = (0 if diff == 0 else
                                    1 if diff == 1 else
                                    2 if diff == 3 else 0)
                else:
                    action_human = 0

                state_ai, _, _, _, _ = env.step(action_ai, action_human)
                renderer.render(round_num)

            # ── 判定胜负 ──────────────────────────────────────────
            if   env.done1 and not env.done2:
                winner = "You Win!";      wins_human += 1
            elif env.done2 and not env.done1:
                winner = "AI Wins!";      wins_ai    += 1
            elif env.score1 > env.score2:
                winner = "AI Wins (Score)!"; wins_ai    += 1
            elif env.score2 > env.score1:
                winner = "You Win (Score)!"; wins_human += 1
            else:
                winner = "Draw!";         draws      += 1

            print(f"Round {round_num:3d}  "
                  f"AI={env.score1}  You={env.score2}  → {winner}")
            renderer.show_result(winner)

            # 等待按键继续
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            raise SystemExit
                        waiting = False
                renderer.clock.tick(30)

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        renderer.close()
        _print_summary(wins_ai, wins_human, draws)


# ══════════════════════════════════════════════════════════════════════
# 模式 2：轮流对战
# ══════════════════════════════════════════════════════════════════════

def run_turns(args, agent: DQNAgent, cfg: Config):
    """
    轮流对战主循环

    每轮依次: 人类游戏 → AI 游戏 → 比较分数
    最终统计总胜场数
    """
    game = SnakeGame(cfg.GRID_COLS, cfg.GRID_ROWS, cfg.CELL_SIZE)
    from env.render import Renderer
    renderer = Renderer(game, fps=args.fps, title="Snake — Turns Mode")

    human_scores: list = []
    ai_scores:    list = []

    print("\n轮流对战模式")
    print(f"  共 {args.rounds} 轮  WASD/方向键控制  Q=退出\n")

    try:
        for rd in range(1, args.rounds + 1):
            print(f"── 第 {rd} 轮 ──")

            # 人类回合
            pygame.display.set_caption(
                f"Round {rd}/{args.rounds}  Your Turn — WASD/Arrow keys")
            h_score = _play_human_round(game, renderer, rd)
            human_scores.append(h_score)
            print(f"  你的得分: {h_score}")

            _pause_between(renderer, game, 1.0)

            # AI 回合
            pygame.display.set_caption(
                f"Round {rd}/{args.rounds}  AI's Turn")
            a_score = _play_ai_round(game, renderer, agent, rd)
            ai_scores.append(a_score)
            print(f"  AI 得分:  {a_score}")

            # 本轮结论
            if   h_score > a_score: print("  → 本轮你赢了！")
            elif a_score > h_score: print("  → 本轮 AI 赢了！")
            else:                   print("  → 本轮平局")
            print()

            _pause_between(renderer, game, 1.5)

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        renderer.close()

    # ── 汇总 ──────────────────────────────────────────────────────
    n = min(len(human_scores), len(ai_scores))
    if n == 0:
        return
    h_wins = sum(1 for h, a in zip(human_scores, ai_scores) if h > a)
    a_wins = sum(1 for h, a in zip(human_scores, ai_scores) if a > h)
    ties   = n - h_wins - a_wins

    print(f"\n{'='*45}")
    print(f"  轮流对战结果  ({n} 轮)")
    print(f"  你   均分={np.mean(human_scores):.1f}  "
          f"最高={max(human_scores)}  胜场={h_wins}")
    print(f"  AI   均分={np.mean(ai_scores):.1f}  "
          f"最高={max(ai_scores)}  胜场={a_wins}")
    print(f"  平局: {ties}")
    print(f"{'='*45}\n")


# ── 轮流模式辅助函数 ────────────────────────────────────────────────

def _play_human_round(game: SnakeGame, renderer, round_num: int) -> int:
    """人类玩一局，返回得分"""
    state    = game.reset()
    next_dir = game.direction
    done     = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key in _KEY_MAP:
                    target  = _KEY_MAP[event.key]
                    cur_idx = CLOCKWISE.index(game.direction)
                    tgt_idx = CLOCKWISE.index(target)
                    if (tgt_idx - cur_idx) % 4 != 2:
                        next_dir = target
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    raise SystemExit

        # 绝对方向 → 相对动作
        cur_idx = CLOCKWISE.index(game.direction)
        tgt_idx = CLOCKWISE.index(next_dir)
        diff    = (tgt_idx - cur_idx) % 4
        action  = (0 if diff == 0 else
                   1 if diff == 1 else
                   2 if diff == 3 else 0)

        state, _, done, _ = game.step(action)
        renderer.render(episode=round_num, total_reward=float(game.score))

    return game.score


def _play_ai_round(game: SnakeGame, renderer,
                   agent: DQNAgent, round_num: int) -> int:
    """AI 玩一局，返回得分"""
    state = game.reset()
    done  = False
    total = 0.0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    raise SystemExit

        action = agent.select_action(state, deterministic=True)
        state, reward, done, _ = game.step(action)
        total += reward
        renderer.render(episode=round_num, total_reward=total)

    return game.score


def _pause_between(renderer, game, seconds: float):
    """在两局之间短暂暂停，保持窗口响应"""
    deadline = time.time() + seconds
    while time.time() < deadline:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    raise SystemExit
        renderer.render(episode=0, total_reward=0)


# ══════════════════════════════════════════════════════════════════════
# 共用工具
# ══════════════════════════════════════════════════════════════════════

def _print_summary(wins_ai: int, wins_human: int, draws: int):
    total = wins_ai + wins_human + draws
    if total == 0:
        return
    print(f"\n{'='*40}")
    print(f"  对战统计  ({total} 轮)")
    print(f"  AI 胜  : {wins_ai:3d}  ({wins_ai/total*100:.0f}%)")
    print(f"  你 胜  : {wins_human:3d}  ({wins_human/total*100:.0f}%)")
    print(f"  平局   : {draws:3d}")
    print(f"{'='*40}\n")


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="贪吃蛇 人机对战")
    p.add_argument("--model", type=str, required=True,
                   help="AI 模型 checkpoint 路径")
    p.add_argument("--mode",  type=str, default="simultaneous",
                   choices=["simultaneous", "turns"],
                   help="对战模式: simultaneous(同时) / turns(轮流)")
    p.add_argument("--rounds", type=int, default=3,
                   help="turns 模式下的对战轮数（默认 3）")
    p.add_argument("--fps",    type=int, default=8,
                   help="游戏帧率（默认 8，便于人类操作）")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"[Error] 模型文件不存在: {args.model}")
        sys.exit(1)

    cfg   = Config()
    agent = DQNAgent(cfg)
    agent.load(args.model)

    if args.mode == "simultaneous":
        run_simultaneous(args, agent, cfg)
    else:
        run_turns(args, agent, cfg)


if __name__ == "__main__":
    main()
