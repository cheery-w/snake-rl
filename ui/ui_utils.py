"""
界面工具函数
封装常用 pygame 绘制操作：按钮、文本、背景、动画贪吃蛇
"""

import math
import pygame
from typing import Tuple, Optional, List


# ── 宋体字体工具 ─────────────────────────────────────────────────────

def get_chinese_font(size: int, bold: bool = False) -> pygame.font.Font:
    """
    获取宋体中文字体（Windows SimSun / macOS STSong / 跨平台兜底）

    Args:
        size: 字号（像素）
        bold: 是否粗体

    Returns:
        可正常渲染中文的 pygame.font.Font
    """
    candidates = [
        "SimSun",      # Windows 宋体
        "宋体",         # Windows 中文名
        "simsun",      # 小写变体
        "STSong",      # macOS 宋体
        "FangSong",    # 仿宋（备用）
        "STFangsong",  # macOS 仿宋
        "Noto Serif CJK SC",   # Linux
        "WenQuanYi Micro Hei", # Linux 备用
    ]
    # 用系统字体列表来判断某个字体名是否真实存在
    available = {f.lower() for f in pygame.font.get_fonts()}
    for name in candidates:
        if name.lower().replace(" ", "") in available or \
                name.lower() in available:
            return pygame.font.SysFont(name, size, bold=bold)
    # 兜底：系统默认字体
    return pygame.font.SysFont(None, size, bold=bold)


# ── 全局调色板 ────────────────────────────────────────────────────────
BG_DARK       = (12,  12,  20)    # 深蓝黑背景
BG_PANEL      = (22,  22,  38)    # 面板背景
ACCENT_GREEN  = (0,   200,  80)   # 主绿色（品牌色）
ACCENT_BLUE   = (50,  130, 255)   # 蓝色高亮
ACCENT_GOLD   = (255, 215,   0)   # 金色
ACCENT_RED    = (220,  50,  50)   # 红色
TEXT_PRIMARY  = (230, 230, 240)   # 主文字
TEXT_SECONDARY= (140, 140, 160)   # 次级文字
BTN_NORMAL    = (30,   50,  80)   # 按钮默认
BTN_HOVER     = (50,   80, 130)   # 按钮悬停
BTN_ACTIVE    = (0,   160,  60)   # 按钮激活（选中）
BTN_BORDER    = (60,   90, 140)   # 按钮边框
DISABLED      = (60,   60,  80)   # 禁用状态


class Button:
    """
    可点击按钮组件

    属性:
        rect      — 矩形区域
        label     — 显示文本
        tag       — 任意标识符（用于区分按钮）
        enabled   — 是否可点击
        selected  — 是否处于选中状态（用于单选组）
    """

    def __init__(
        self,
        rect:    Tuple[int, int, int, int],
        label:   str,
        tag:     object = None,
        enabled: bool   = True,
        font:    pygame.font.Font = None,
    ):
        self.rect     = pygame.Rect(rect)
        self.label    = label
        self.tag      = tag if tag is not None else label
        self.enabled  = enabled
        self.selected = False
        self._font    = font   # None 时由 draw() 接收

    def draw(
        self,
        surface: pygame.Surface,
        mouse_pos: Tuple[int, int],
        font: pygame.font.Font,
    ):
        """绘制按钮（根据状态自动切换颜色）"""
        f = self._font or font

        if not self.enabled:
            bg_col = DISABLED
            bd_col = DISABLED
        elif self.selected:
            bg_col = BTN_ACTIVE
            bd_col = ACCENT_GREEN
        elif self.rect.collidepoint(mouse_pos):
            bg_col = BTN_HOVER
            bd_col = ACCENT_BLUE
        else:
            bg_col = BTN_NORMAL
            bd_col = BTN_BORDER

        pygame.draw.rect(surface, bg_col,  self.rect, border_radius=8)
        pygame.draw.rect(surface, bd_col,  self.rect, width=2, border_radius=8)

        text_col = TEXT_SECONDARY if not self.enabled else TEXT_PRIMARY
        surf = f.render(self.label, True, text_col)
        surface.blit(surf, surf.get_rect(center=self.rect.center))

    def is_clicked(
        self, event: pygame.event.Event
    ) -> bool:
        """检测鼠标点击事件是否命中此按钮"""
        return (
            self.enabled
            and event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )


class Panel:
    """半透明圆角面板"""

    def __init__(
        self,
        rect:         Tuple[int, int, int, int],
        color:        Tuple[int, int, int]       = BG_PANEL,
        border_color: Optional[Tuple[int, int, int]] = BTN_BORDER,
        alpha:        int = 220,
        radius:       int = 12,
    ):
        self.rect         = pygame.Rect(rect)
        self.color        = color
        self.border_color = border_color
        self.alpha        = alpha
        self.radius       = radius

    def draw(self, surface: pygame.Surface):
        surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        r = (*self.color, self.alpha)
        pygame.draw.rect(surf, r,
                         pygame.Rect(0, 0, *self.rect.size),
                         border_radius=self.radius)
        surface.blit(surf, self.rect.topleft)
        if self.border_color:
            pygame.draw.rect(surface, self.border_color, self.rect,
                             width=2, border_radius=self.radius)


# ──────────────────────────────────────────────────────────────────────
# 文字绘制
# ──────────────────────────────────────────────────────────────────────

def draw_text(
    surface: pygame.Surface,
    text:    str,
    font:    pygame.font.Font,
    color:   Tuple[int, int, int],
    center:  Tuple[int, int],
):
    """在指定中心坐标绘制文字"""
    surf = font.render(text, True, color)
    surface.blit(surf, surf.get_rect(center=center))


def draw_text_left(
    surface: pygame.Surface,
    text:    str,
    font:    pygame.font.Font,
    color:   Tuple[int, int, int],
    topleft: Tuple[int, int],
):
    surf = font.render(text, True, color)
    surface.blit(surf, topleft)


# ──────────────────────────────────────────────────────────────────────
# 背景：动态网格 + 滚动贪吃蛇装饰
# ──────────────────────────────────────────────────────────────────────

class BackgroundAnimator:
    """
    主菜单动态背景

    绘制暗色网格 + 两条缓慢移动的"装饰蛇"
    """

    CELL  = 20
    SPEED = 3   # 格/秒（通过 tick 控制）

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self._tick = 0

        # 两条装饰蛇（颜色、初始位置）
        self._snakes = [
            {
                "segs":  [[20 * i, 40] for i in range(8, -1, -1)],
                "dir":   (1, 0),
                "color": (0, 80, 30),
                "timer": 0,
            },
            {
                "segs":  [[width - 20 * i, height - 60]
                          for i in range(8, -1, -1)],
                "dir":   (-1, 0),
                "color": (20, 40, 100),
                "timer": 0,
            },
        ]

    def update(self):
        """每帧调用，推进动画"""
        self._tick += 1
        # 每 8 帧移动一步
        if self._tick % 8 != 0:
            return
        bs = self.CELL
        for snake in self._snakes:
            dx, dy = snake["dir"]
            head   = snake["segs"][0]
            nx, ny = head[0] + dx * bs, head[1] + dy * bs

            # 越界时随机转向或折返
            if nx < 0 or nx >= self.w or ny < 0 or ny >= self.h:
                # 尝试转向
                possible = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                import random
                random.shuffle(possible)
                for nd in possible:
                    nx2 = head[0] + nd[0] * bs
                    ny2 = head[1] + nd[1] * bs
                    if 0 <= nx2 < self.w and 0 <= ny2 < self.h:
                        dx, dy   = nd
                        nx, ny   = nx2, ny2
                        snake["dir"] = nd
                        break
                else:
                    continue

            snake["segs"].insert(0, [nx, ny])
            snake["segs"].pop()

    def draw(self, surface: pygame.Surface):
        surface.fill(BG_DARK)
        self._draw_grid(surface)
        for snake in self._snakes:
            for i, seg in enumerate(snake["segs"]):
                alpha = max(30, 80 - i * 8)
                col   = (*snake["color"], alpha)
                s     = pygame.Surface((self.CELL - 2, self.CELL - 2),
                                       pygame.SRCALPHA)
                s.fill(col)
                surface.blit(s, (seg[0] + 1, seg[1] + 1))

    def _draw_grid(self, surface: pygame.Surface):
        col = (25, 25, 40)
        for x in range(0, self.w, self.CELL):
            pygame.draw.line(surface, col, (x, 0), (x, self.h))
        for y in range(0, self.h, self.CELL):
            pygame.draw.line(surface, col, (0, y), (self.w, y))


# ──────────────────────────────────────────────────────────────────────
# 标题动画（颜色渐变）
# ──────────────────────────────────────────────────────────────────────

def get_title_color(tick: int) -> Tuple[int, int, int]:
    """基于帧数产生循环渐变颜色"""
    t = tick * 0.03
    r = int(127 + 127 * math.sin(t))
    g = int(127 + 127 * math.sin(t + 2.094))
    b = int(127 + 127 * math.sin(t + 4.189))
    return (r, g, b)


# ──────────────────────────────────────────────────────────────────────
# 等待按键继续
# ──────────────────────────────────────────────────────────────────────

def wait_for_key(clock: pygame.time.Clock, fps: int = 30) -> bool:
    """
    阻塞等待任意按键，返回 True（按键）/ False（关闭窗口）
    """
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                return True
            if (event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1):
                return True
        clock.tick(fps)
