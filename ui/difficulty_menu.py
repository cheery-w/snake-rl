"""
难度选择界面
全部文字使用宋体（SimSun）
"""

import pygame
from typing import Optional

from ui.ui_utils import (
    get_chinese_font,
    BackgroundAnimator,
    TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_GOLD,
    BTN_NORMAL, BTN_HOVER, BTN_BORDER,
)

DIFF_EASY   = "easy"
DIFF_MEDIUM = "medium"
DIFF_HARD   = "hard"
DIFF_EXPERT = "expert"
BACK        = "back"

_INFO = {
    DIFF_EASY:   {"汉字": "简  单", "color": (60,200,100),  "stars": 1, "desc": "基础 DQN，适合初次体验"},
    DIFF_MEDIUM: {"汉字": "中  等", "color": (100,160,255), "stars": 2, "desc": "标准 Double DQN，具备基本策略"},
    DIFF_HARD:   {"汉字": "困  难", "color": (255,160,50),  "stars": 3, "desc": "Dueling+Double DQN，追食效率高"},
    DIFF_EXPERT: {"汉字": "专  家", "color": (220,50,80),   "stars": 4, "desc": "最强模型，接近最优策略"},
}
_ORDER = [DIFF_EASY, DIFF_MEDIUM, DIFF_HARD, DIFF_EXPERT]


class DifficultyMenu:
    """难度选择界面（宋体）"""

    def __init__(self, screen: pygame.Surface):
        self.screen   = screen
        self.clock    = pygame.time.Clock()
        self.tick     = 0
        self.w, self.h = screen.get_size()
        self.bg       = BackgroundAnimator(self.w, self.h)
        self.selected = DIFF_MEDIUM

        self.font_title   = get_chinese_font(38, bold=True)
        self.font_btn     = get_chinese_font(24, bold=True)
        self.font_desc    = get_chinese_font(16)
        self.font_star    = get_chinese_font(18)
        self.font_hint    = get_chinese_font(14)

        self._rects = {}
        self._rect_confirm = None
        self._rect_back    = None
        self._build_rects()

    def _build_rects(self):
        cx = self.w // 2
        # 4个难度按钮，2列2行
        bw, bh, gx, gy = 200, 52, 220, 68
        sy = 200
        for i, diff in enumerate(_ORDER):
            col = i % 2
            row = i // 2
            x = cx - gx//2 - bw//2 + col * gx + 10
            y = sy + row * gy
            self._rects[diff] = pygame.Rect(x, y, bw, bh)
        # 确认和返回按钮
        self._rect_confirm = pygame.Rect(cx - 110, self.h - 140, 220, 50)
        self._rect_back    = pygame.Rect(cx - 90,  self.h - 78,  180, 38)

    def run(self) -> Optional[str]:
        while True:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return BACK
                    if event.key in (pygame.K_q,):
                        return "quit"
                    if event.key == pygame.K_RETURN:
                        return self.selected
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for diff, rect in self._rects.items():
                        if rect.collidepoint(mouse):
                            self.selected = diff
                    if self._rect_confirm.collidepoint(mouse):
                        return self.selected
                    if self._rect_back.collidepoint(mouse):
                        return BACK

            self.tick += 1
            self.bg.update()
            self._draw(mouse)
            pygame.display.flip()
            self.clock.tick(30)

    def _draw(self, mouse):
        self.bg.draw(self.screen)
        self._draw_title()
        self._draw_buttons(mouse)
        self._draw_desc_panel()
        self._draw_confirm(mouse)
        self._draw_footer()

    def _draw_title(self):
        cx = self.w // 2
        import math
        t   = self.tick * 0.025
        col = (int(100+100*math.sin(t)), int(180+60*math.sin(t+2.1)), int(80+80*math.sin(t+4.2)))
        surf = self.font_title.render("选择  AI  难度", True, col)
        self.screen.blit(surf, surf.get_rect(center=(cx, 80)))
        pygame.draw.line(self.screen, (50,80,120), (cx-130,112), (cx+130,112), 1)
        sub = self.font_hint.render("选择后点击「确认开始」进入人机对战", True, TEXT_SECONDARY)
        self.screen.blit(sub, sub.get_rect(center=(cx, 138)))

    def _draw_buttons(self, mouse):
        for diff, rect in self._rects.items():
            info    = _INFO[diff]
            sel     = (diff == self.selected)
            hovered = rect.collidepoint(mouse)
            col     = info["color"]
            if sel:
                bg_col = tuple(max(0, c-80) for c in col)
                bd_col = col
                tx_col = col
            elif hovered:
                bg_col = (40,60,90)
                bd_col = (100,140,200)
                tx_col = TEXT_PRIMARY
            else:
                bg_col = BTN_NORMAL
                bd_col = BTN_BORDER
                tx_col = TEXT_SECONDARY
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=10)
            pygame.draw.rect(self.screen, bd_col, rect, width=2, border_radius=10)
            # 文字
            ts = self.font_btn.render(info["汉字"], True, tx_col)
            self.screen.blit(ts, ts.get_rect(center=(rect.centerx - 20, rect.centery)))
            # 星星
            stars = "★" * info["stars"] + "☆" * (4 - info["stars"])
            ss = self.font_star.render(stars, True, ACCENT_GOLD if sel else (100,90,50))
            self.screen.blit(ss, (rect.right - ss.get_width() - 6, rect.centery - ss.get_height()//2))

    def _draw_desc_panel(self):
        cx   = self.w // 2
        info = _INFO.get(self.selected)
        if not info:
            return
        # 半透明面板
        pane = pygame.Surface((340, 60), pygame.SRCALPHA)
        pane.fill((22, 22, 38, 200))
        self.screen.blit(pane, (cx - 170, 348))
        pygame.draw.rect(self.screen, BTN_BORDER,
                         pygame.Rect(cx-170, 348, 340, 60), 2, border_radius=8)
        sel_surf = self.font_btn.render(
            f"当前选择：{info['汉字'].strip()}", True, info["color"])
        self.screen.blit(sel_surf, sel_surf.get_rect(center=(cx, 366)))
        ds = self.font_desc.render(info["desc"], True, TEXT_SECONDARY)
        self.screen.blit(ds, ds.get_rect(center=(cx, 392)))

    def _draw_confirm(self, mouse):
        for rect, label, is_back in [
            (self._rect_confirm, "确 认 开 始  ▶", False),
            (self._rect_back,    "← 返回主菜单",   True),
        ]:
            hov = rect.collidepoint(mouse)
            if is_back:
                bg = (40,25,25) if not hov else (70,35,35)
                bd = (100,60,60) if not hov else (180,80,80)
                tx = (160,100,100) if not hov else (220,140,140)
            else:
                bg = (0,100,40) if not hov else (0,140,60)
                bd = (0,180,80) if not hov else (0,220,100)
                tx = (180,255,180) if not hov else (220,255,220)
            pygame.draw.rect(self.screen, bg, rect, border_radius=10)
            pygame.draw.rect(self.screen, bd, rect, width=2, border_radius=10)
            font = self.font_btn if not is_back else self.font_desc
            ts = font.render(label, True, tx)
            self.screen.blit(ts, ts.get_rect(center=rect.center))

    def _draw_footer(self):
        hints = [
            "点击选择难度  |  Enter 确认  |  Esc 返回菜单  |  Q 退出程序",
        ]
        for i, h in enumerate(hints):
            surf = self.font_hint.render(h, True, (55,60,80))
            self.screen.blit(surf, surf.get_rect(center=(self.w//2, self.h - 22 + i*18)))
