"""
主入口 — 贪吃蛇 AI
用法: python main.py [--fps 15]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from config import Config
from ui.main_menu import MainMenu, MENU_AI_DEMO, MENU_VS_AI, MENU_EXIT
from ui.difficulty_menu import DifficultyMenu, BACK
from env.snake_env import SnakeGame, Direction, CLOCKWISE
from ui.ui_utils import get_chinese_font

# ── 返回码 ─────────────────────────────────────────────────────────────
RET_MENU    = "menu"
RET_QUIT    = "quit"
RET_RESTART = "restart"

# ── 窗口尺寸 ───────────────────────────────────────────────────────────
MENU_W, MENU_H = 500, 640

# ── 调色板 ─────────────────────────────────────────────────────────────
_BG   = (15,  15,  15)
_HEAD = (0,   220, 80)
_BODY = (0,   160, 60)
_FOOD = (220, 50,  50)
_TEXT = (220, 220, 220)
_DIM  = (100, 100, 100)


# ──────────────────────────────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="贪吃蛇 AI")
    p.add_argument("--fps",   type=int, default=15, help="游戏帧率（默认 15）")
    p.add_argument("--model", type=str, default=None, help="AI 模型路径")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# 查找模型
# ──────────────────────────────────────────────────────────────────────

def _find_model(difficulty=None):
    base = os.path.dirname(os.path.abspath(__file__))
    ck   = os.path.join(base, "checkpoints")
    candidates = []
    if difficulty:
        candidates.append(os.path.join(ck, f"{difficulty}.pt"))
    candidates.append(os.path.join(ck, "best.pt"))
    if os.path.isdir(ck):
        for f in sorted(os.listdir(ck)):
            if f.endswith(".pt"):
                candidates.append(os.path.join(ck, f))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


# ──────────────────────────────────────────────────────────────────────
# 窗口工具
# ──────────────────────────────────────────────────────────────────────

def _resize(w, h, title="贪吃蛇 AI"):
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(title)
    return screen


# ──────────────────────────────────────────────────────────────────────
# 单侧棋盘绘制助手
# ──────────────────────────────────────────────────────────────────────

def _draw_board(screen, game, ox, oy, cell, cols, rows, label, label_color,
                label_font, info_font):
    gw = cols * cell
    gh = rows * cell
    pygame.draw.rect(screen, (20, 20, 20), (ox, oy, gw, gh))
    for x in range(0, gw + 1, cell):
        pygame.draw.line(screen, (35, 35, 35), (ox + x, oy), (ox + x, oy + gh))
    for y in range(0, gh + 1, cell):
        pygame.draw.line(screen, (35, 35, 35), (ox, oy + y), (ox + gw, oy + y))
    fx, fy = game.food
    pygame.draw.circle(screen, _FOOD,
                       (ox + fx + cell // 2, oy + fy + cell // 2), cell // 2 - 2)
    for i, seg in enumerate(game.snake):
        color = _HEAD if i == 0 else _BODY
        rect  = pygame.Rect(ox + seg[0] + 1, oy + seg[1] + 1, cell - 2, cell - 2)
        pygame.draw.rect(screen, color, rect, border_radius=3)
    ls = label_font.render(label, True, label_color)
    screen.blit(ls, ls.get_rect(center=(ox + gw // 2, oy - 18)))
    sc = info_font.render(f"得分：{game.score}", True, _TEXT)
    screen.blit(sc, sc.get_rect(center=(ox + gw // 2, oy + gh + 14)))


# ──────────────────────────────────────────────────────────────────────
# 叠加层：提示信息
# ──────────────────────────────────────────────────────────────────────

def show_tip(screen, lines, clock, font):
    """全屏提示，按任意键返回菜单"""
    w, h = screen.get_size()
    screen.fill(_BG)
    for i, line in enumerate(lines):
        surf = font.render(line, True, _TEXT)
        screen.blit(surf, surf.get_rect(center=(w // 2, h // 2 - 20 + i * 32)))
    hint = font.render("按任意键返回菜单", True, _DIM)
    screen.blit(hint, hint.get_rect(center=(w // 2, h // 2 + 80)))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return RET_QUIT
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                return RET_MENU
        clock.tick(30)


# ──────────────────────────────────────────────────────────────────────
# AI 演示模式
# ──────────────────────────────────────────────────────────────────────

def run_ai_demo(fps, model_path, screen):
    """AI 自动演示。  Esc=返回菜单  Q=退出程序"""
    clock = pygame.time.Clock()
    font  = get_chinese_font(16)

    if not model_path:
        return show_tip(screen,
                        ["未找到训练好的模型！", "请先运行 python train.py"],
                        clock, font)

    from agent.dqn_agent import DQNAgent
    cfg   = Config()
    game  = SnakeGame(cfg.GRID_COLS, cfg.GRID_ROWS, cfg.CELL_SIZE)
    agent = DQNAgent(cfg)
    try:
        agent.load(model_path)
    except Exception as e:
        return show_tip(screen, [f"模型加载失败：{e}"], clock, font)

    cell = cfg.CELL_SIZE
    cols = cfg.GRID_COLS
    rows = cfg.GRID_ROWS
    gw   = cols * cell
    gh   = rows * cell

    def _draw(episode, total_reward):
        screen.fill(_BG)
        for x in range(0, gw + 1, cell):
            pygame.draw.line(screen, (30, 30, 30), (x, 0), (x, gh))
        for y in range(0, gh + 1, cell):
            pygame.draw.line(screen, (30, 30, 30), (0, y), (gw, y))
        fx, fy = game.food
        pygame.draw.circle(screen, _FOOD,
                           (fx + cell // 2, fy + cell // 2), cell // 2 - 2)
        for i, seg in enumerate(game.snake):
            color = _HEAD if i == 0 else _BODY
            rect  = pygame.Rect(seg[0] + 1, seg[1] + 1, cell - 2, cell - 2)
            pygame.draw.rect(screen, color, rect, border_radius=3)
        info = (f"第 {episode} 局  "
                f"得分：{game.score}  "
                f"长度：{game.snake_length}  "
                f"奖励：{total_reward:.1f}")
        surf = font.render(info, True, _TEXT)
        screen.blit(surf, (6, gh + 6))
        pygame.display.flip()

    round_num  = 0
    best_score = 0
    print(f"AI 演示  模型={model_path}  Esc=返回菜单  Q=退出")

    while True:
        round_num   += 1
        state        = game.reset()
        done         = False
        total_reward = 0.0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return RET_QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return RET_MENU
                    if event.key == pygame.K_q:
                        return RET_QUIT

            action = agent.select_action(state, deterministic=True)
            state, reward, done, _ = game.step(action)
            total_reward += reward
            _draw(round_num, total_reward)
            clock.tick(fps)

        best_score = max(best_score, game.score)
        print(f"Round {round_num:4d}  Score={game.score}  Best={best_score}")

        # 局间停顿，仍响应按键
        t0 = pygame.time.get_ticks()
        while pygame.time.get_ticks() - t0 < 500:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return RET_QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return RET_MENU
                    if event.key == pygame.K_q:
                        return RET_QUIT
            clock.tick(30)


# ──────────────────────────────────────────────────────────────────────
# 人机对战模式
# ──────────────────────────────────────────────────────────────────────

def run_vs_ai(fps, difficulty, model_path, screen):
    """人机对战（左=玩家，右=AI）。  Esc=返回菜单  Q=退出  R=重玩  Space=暂停"""
    clock      = pygame.time.Clock()
    font       = get_chinese_font(16)
    label_font = get_chinese_font(18, bold=True)

    if not model_path:
        return show_tip(screen,
                        [f"未找到 {difficulty} 难度的模型！",
                         "请先运行 python train.py"],
                        clock, font)

    from agent.dqn_agent import DQNAgent
    cfg      = Config()
    ai_agent = DQNAgent(cfg)
    try:
        ai_agent.load(model_path)
    except Exception as e:
        return show_tip(screen, [f"模型加载失败：{e}"], clock, font)

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

    def dir_to_action(cur, tgt):
        ci = CLOCKWISE.index(cur)
        ti = CLOCKWISE.index(tgt)
        d  = (ti - ci) % 4
        return 0 if d == 0 else 1 if d == 1 else 2 if d == 3 else 0

    cell = cfg.CELL_SIZE
    cols = cfg.GRID_COLS
    rows = cfg.GRID_ROWS
    gw   = cols * cell
    gh   = rows * cell
    sw, sh = screen.get_size()
    gap  = 60
    total_w = gw * 2 + gap
    lox  = (sw - total_w) // 2        # left board origin x
    rox  = lox + gw + gap             # right board origin x
    boy  = 50                         # board origin y

    human_game = SnakeGame(cols, rows, cell)
    ai_game    = SnakeGame(cols, rows, cell)

    def redraw(round_num):
        screen.fill((10, 10, 10))
        title_surf = font.render(
            f"第 {round_num} 局  |  难度：{difficulty}  |  "
            "WASD/方向键  Space=暂停  R=重玩  Esc=菜单  Q=退出",
            True, (140, 140, 140))
        screen.blit(title_surf, title_surf.get_rect(center=(sw // 2, 22)))
        _draw_board(screen, human_game, lox, boy, cell, cols, rows,
                    "玩  家", (100, 220, 100), label_font, font)
        _draw_board(screen, ai_game, rox, boy, cell, cols, rows,
                    "人  工  智  能", (100, 160, 255), label_font, font)
        pygame.display.flip()

    round_num = 0

    while True:
        round_num += 1
        h_state    = human_game.reset()
        ai_state   = ai_game.reset()
        h_next_dir = human_game.direction
        h_done     = False
        ai_done    = False
        paused     = False
        restart    = False

        while not (h_done and ai_done):
            restart = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return RET_QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return RET_QUIT
                    elif event.key == pygame.K_ESCAPE:
                        return RET_MENU
                    elif event.key == pygame.K_r:
                        restart = True
                        break
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key in KEY_MAP and not h_done:
                        tgt = KEY_MAP[event.key]
                        ci  = CLOCKWISE.index(human_game.direction)
                        ti  = CLOCKWISE.index(tgt)
                        if (ti - ci) % 4 != 2:
                            h_next_dir = tgt

            if restart:
                break

            if paused:
                redraw(round_num)
                clock.tick(30)
                continue

            if not h_done:
                h_action = dir_to_action(human_game.direction, h_next_dir)
                h_state, _, h_done, _ = human_game.step(h_action)
            if not ai_done:
                ai_action = ai_agent.select_action(ai_state, deterministic=True)
                ai_state, _, ai_done, _ = ai_game.step(ai_action)

            redraw(round_num)
            clock.tick(fps)

        if restart:
            continue

        # ── 局结束：显示结果 ──
        hs, as_ = human_game.score, ai_game.score
        if hs > as_:
            winner = "玩家获胜！"
        elif as_ > hs:
            winner = "AI 获胜！"
        else:
            winner = "平  局！"

        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))
        for i, line in enumerate([
            winner,
            f"玩家得分：{hs}    AI 得分：{as_}",
            "R=重玩    Esc=返回菜单    Q=退出",
        ]):
            surf = label_font.render(line, True, _TEXT)
            screen.blit(surf, surf.get_rect(center=(sw // 2, sh // 2 - 28 + i * 34)))
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return RET_QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        return RET_MENU
                    elif event.key == pygame.K_q:
                        return RET_QUIT
            clock.tick(30)


# ──────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    pygame.init()
    screen = _resize(MENU_W, MENU_H)

    state      = "menu"
    difficulty = "medium"

    while state not in ("quit", None):

        # ── 主菜单 ──
        if state == "menu":
            if screen.get_size() != (MENU_W, MENU_H):
                screen = _resize(MENU_W, MENU_H)
            choice = MainMenu(screen).run()
            if choice is None or choice == MENU_EXIT:
                state = "quit"
            elif choice == MENU_AI_DEMO:
                state = "ai_demo"
            elif choice == MENU_VS_AI:
                state = "diff_select"

        # ── 难度选择 ──
        elif state == "diff_select":
            if screen.get_size() != (MENU_W, MENU_H):
                screen = _resize(MENU_W, MENU_H)
            r = DifficultyMenu(screen).run()
            if r is None or r == "quit":
                state = "quit"
            elif r == BACK:
                state = "menu"
            else:
                difficulty = r
                state      = "vs_ai"

        # ── AI 演示 ──
        elif state == "ai_demo":
            cfg    = Config()
            gw     = cfg.GRID_COLS * cfg.CELL_SIZE
            gh     = cfg.GRID_ROWS * cfg.CELL_SIZE + 30
            screen = _resize(gw, gh, "AI 演示")
            ret    = run_ai_demo(args.fps, args.model or _find_model(), screen)
            state  = "menu" if ret == RET_MENU else "quit"

        # ── 人机对战 ──
        elif state == "vs_ai":
            cfg    = Config()
            gw     = cfg.GRID_COLS * cfg.CELL_SIZE
            gh     = cfg.GRID_ROWS * cfg.CELL_SIZE
            vs_w   = gw * 2 + 120
            vs_h   = gh + 80
            screen = _resize(vs_w, vs_h, f"人机对战 — {difficulty}")
            ret    = run_vs_ai(args.fps, difficulty,
                               args.model or _find_model(difficulty), screen)
            state  = "menu" if ret == RET_MENU else "quit"

    pygame.quit()
    print("再见！")


if __name__ == "__main__":
    main()
