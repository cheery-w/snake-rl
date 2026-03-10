"""
多智能体专用渲染器（内部模块）
"""

import pygame
from typing import List

from env.multi_agent_env import MultiAgentEnv


_BG   = (15, 15, 15)
_GRID = (30, 30, 30)
_FOOD = (220, 50, 50)
_TEXT = (200, 200, 200)
_DIM  = (80, 80, 80)


class MultiAgentRenderer:
    """
    多智能体游戏渲染器（支持 2-4 条蛇，不同颜色）
    """

    def __init__(self, env: MultiAgentEnv, fps: int = 15):
        pygame.init()
        self.env  = env
        self.fps  = fps
        self.cell = env.cell
        self.w    = env.width
        self.h    = env.height

        self.screen = pygame.display.set_mode((self.w, self.h + 50))
        pygame.display.set_caption("Snake — Multi-Agent")
        self.clock  = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)

    def render(self, round_num: int, scores: List[int]):
        self._handle_events()
        self.screen.fill(_BG)
        self._draw_grid()
        self._draw_food()
        self._draw_snakes()
        self._draw_hud(round_num, scores)
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

    def _draw_food(self):
        fx, fy = self.env.food
        cx = fx + self.cell // 2
        cy = fy + self.cell // 2
        pygame.draw.circle(self.screen, _FOOD, (cx, cy), self.cell // 2 - 2)

    def _draw_snakes(self):
        colors = self.env.AGENT_COLORS
        for i, snake in enumerate(self.env.snakes):
            hc, bc = colors[i % len(colors)]
            segs   = list(snake.body)
            for j, seg in enumerate(segs):
                color = hc if j == 0 else bc
                if not snake.alive:
                    color = tuple(max(0, c - 80) for c in color)
                m    = 1
                rect = pygame.Rect(seg[0] + m, seg[1] + m,
                                   self.cell - 2 * m, self.cell - 2 * m)
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
            # 标签
            if segs:
                lbl  = self.font_s.render(str(i), True, (0, 0, 0))
                hx   = segs[0][0] + self.cell // 2
                hy   = segs[0][1] + self.cell // 2
                self.screen.blit(lbl, lbl.get_rect(center=(hx, hy)))

    def _draw_hud(self, round_num: int, scores: List[int]):
        y0 = self.h + 6
        colors = self.env.AGENT_COLORS
        x  = 8
        for i, sc in enumerate(scores):
            alive = self.env.snakes[i].alive
            col   = colors[i % len(colors)][0] if alive else _DIM
            surf  = self.font_m.render(f"A{i}:{sc}", True, col)
            self.screen.blit(surf, (x, y0))
            x += surf.get_width() + 16

        rd = self.font_s.render(f"Round {round_num}", True, _DIM)
        self.screen.blit(rd, rd.get_rect(right=self.w - 8, top=y0 + 4))
