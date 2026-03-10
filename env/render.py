"""
游戏可视化渲染器（基于 pygame）
与游戏逻辑完全解耦，仅负责画面渲染
"""

import pygame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .snake_env import SnakeGame


class Renderer:
    """
    贪吃蛇 pygame 渲染器

    使用方法::

        renderer = Renderer(game, fps=20)
        renderer.render()   # 每帧调用一次
        renderer.close()    # 退出时调用
    """

    # ── 调色板 ──────────────────────────────────────────────
    BG        = (15,  15,  15)   # 背景
    GRID_LINE = (35,  35,  35)   # 网格线
    HEAD      = (0,   220, 80)   # 蛇头
    BODY      = (0,   160, 60)   # 蛇身
    FOOD      = (220, 50,  50)   # 食物
    TEXT      = (220, 220, 220)  # 文字

    def __init__(self, game: "SnakeGame", fps: int = 20, title: str = "Snake AI"):
        pygame.init()
        self.game   = game
        self.fps    = fps
        self.cell   = game.cell
        self.width  = game.width
        self.height = game.height

        self.screen = pygame.display.set_mode((self.width, self.height + 30))
        pygame.display.set_caption(title)
        self.clock  = pygame.time.Clock()
        # 使用宋体显示中文信息
        from ui.ui_utils import get_chinese_font
        self.font   = get_chinese_font(16)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def render(self, episode: int = 0, total_reward: float = 0.0):
        """渲染当前帧"""
        self._handle_events()
        self.screen.fill(self.BG)
        self._draw_grid()
        self._draw_food()
        self._draw_snake()
        self._draw_info(episode, total_reward)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        """关闭 pygame 窗口"""
        pygame.quit()

    # ------------------------------------------------------------------
    # 私有绘制方法
    # ------------------------------------------------------------------

    def _draw_grid(self):
        for x in range(0, self.width, self.cell):
            pygame.draw.line(self.screen, self.GRID_LINE, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell):
            pygame.draw.line(self.screen, self.GRID_LINE, (0, y), (self.width, y))

    def _draw_snake(self):
        snake = list(self.game.snake)
        for i, seg in enumerate(snake):
            color  = self.HEAD if i == 0 else self.BODY
            margin = 1
            rect   = pygame.Rect(
                seg[0] + margin,
                seg[1] + margin,
                self.cell - 2 * margin,
                self.cell - 2 * margin,
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # 在蛇头上画眼睛
        if snake:
            self._draw_eyes(snake[0])

    def _draw_eyes(self, head: list):
        """在蛇头上绘制两个小眼睛"""
        bs = self.cell
        cx = head[0] + bs // 2
        cy = head[1] + bs // 2
        offset = bs // 5
        for dx, dy in [(-offset, -offset), (offset, -offset)]:
            pygame.draw.circle(self.screen, (0, 0, 0), (cx + dx, cy + dy), 2)

    def _draw_food(self):
        fx, fy = self.game.food
        # 绘制圆形食物
        cx = fx + self.cell // 2
        cy = fy + self.cell // 2
        r  = self.cell // 2 - 2
        pygame.draw.circle(self.screen, self.FOOD, (cx, cy), r)

    def _draw_info(self, episode: int, total_reward: float):
        """在底部状态栏显示游戏信息（宋体）"""
        info = (
            f"第 {episode} 局  "
            f"得分：{self.game.score}  "
            f"长度：{self.game.snake_length}  "
            f"奖励：{total_reward:.1f}"
        )
        surf = self.font.render(info, True, self.TEXT)
        self.screen.blit(surf, (6, self.height + 6))

    def _handle_events(self):
        """处理窗口关闭事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit("Window closed by user.")
