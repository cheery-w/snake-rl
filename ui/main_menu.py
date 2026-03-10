"""
主菜单界面
全部文字使用宋体（SimSun），三个选项：AI演示 / 人机对战 / 退出
"""

import math
import pygame
from typing import Optional

from ui.ui_utils import (
    get_chinese_font,
    BackgroundAnimator,
    BG_DARK, TEXT_PRIMARY, TEXT_SECONDARY,
    BTN_NORMAL, BTN_HOVER, BTN_BORDER,
)

MENU_AI_DEMO = "ai_demo"
MENU_VS_AI   = "vs_ai"
MENU_EXIT    = "exit"


class MainMenu:
    def __init__(self, screen: pygame.Surface):
        self.screen   = screen
        self.clock    = pygame.time.Clock()
        self.tick     = 0
        self.w, self.h = screen.get_size()
        self.bg       = BackgroundAnimator(self.w, self.h)
        self._hovered = -1

        self.font_title = get_chinese_font(52, bold=True)
        self.font_sub   = get_chinese_font(18)
        self.font_btn   = get_chinese_font(26, bold=True)
        self.font_hint  = get_chinese_font(14)

        self._items = [
            ("AI 演示",  MENU_AI_DEMO,  "观看训练好的 AI 自动游玩"),
            ("人机对战", MENU_VS_AI,    "选择 AI 难度，与 AI 同台竞技"),
            ("退    出", MENU_EXIT,     "关闭游戏"),
        ]
        self._rects = []
        self._build_rects()

    def _build_rects(self):
        cx = self.w // 2
        btn_w, btn_h, gap, start_y = 300, 54, 72, 260
        self._rects = [
            pygame.Rect(cx - btn_w // 2, start_y + i * gap, btn_w, btn_h)
            for i in range(len(self._items))
        ]

    def run(self) -> Optional[str]:
        while True:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return MENU_EXIT
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i, rect in enumerate(self._rects):
                        if rect.collidepoint(mouse):
                            return self._items[i][1]

            self._hovered = next(
                (i for i, r in enumerate(self._rects) if r.collidepoint(mouse)), -1)
            self.tick += 1
            self.bg.update()
            self._draw(mouse)
            pygame.display.flip()
            self.clock.tick(30)

    def _draw(self, mouse):
        self.bg.draw(self.screen)
        self._draw_title()
        self._draw_buttons()
        self._draw_desc()
        self._draw_footer()

    def _draw_title(self):
        cx  = self.w // 2
        t   = self.tick * 0.025
        col = (int(100+100*math.sin(t)), int(200+50*math.sin(t+2.1)), int(80+80*math.sin(t+4.2)))
        surf = self.font_title.render("贪吃蛇  人工智能", True, col)
        self.screen.blit(surf, surf.get_rect(center=(cx, 100)))
        pygame.draw.line(self.screen, (50,80,120), (cx-170,136), (cx+170,136), 1)
        sub = self.font_sub.render("强化学习  ·  Deep Q-Network  ·  PPO", True, TEXT_SECONDARY)
        self.screen.blit(sub, sub.get_rect(center=(cx, 160)))

    def _draw_buttons(self):
        num_font = get_chinese_font(14, bold=True)
        for i, (label, tag, _) in enumerate(self._items):
            rect    = self._rects[i]
            hovered = self._hovered == i
            if tag == MENU_EXIT:
                bg_col = (90,30,30) if hovered else (50,20,20)
                bd_col = (200,80,80) if hovered else (120,50,50)
                tx_col = (240,160,160) if hovered else (200,120,120)
            else:
                bg_col = BTN_HOVER if hovered else BTN_NORMAL
                bd_col = (100,160,255) if hovered else BTN_BORDER
                tx_col = TEXT_PRIMARY if hovered else TEXT_SECONDARY
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=10)
            pygame.draw.rect(self.screen, bd_col, rect, width=2, border_radius=10)
            # 序号圆
            nx, ny = rect.left + 26, rect.centery
            pygame.draw.circle(self.screen, bd_col, (nx, ny), 14)
            ns = num_font.render(str(i+1), True, (0,0,0))
            self.screen.blit(ns, ns.get_rect(center=(nx, ny)))
            # 文字
            ts = self.font_btn.render(label, True, tx_col)
            self.screen.blit(ts, ts.get_rect(center=(rect.centerx+10, rect.centery)))

    def _draw_desc(self):
        if self._hovered < 0:
            return
        _, _, desc = self._items[self._hovered]
        rect = self._rects[self._hovered]
        surf = self.font_hint.render(desc, True, (120,140,160))
        self.screen.blit(surf, surf.get_rect(center=(self.w//2, rect.bottom+14)))

    def _draw_footer(self):
        hint = self.font_hint.render(
            "鼠标点击选择  |  Q 退出程序", True, (55,60,80))
        self.screen.blit(hint, hint.get_rect(center=(self.w//2, self.h-22)))
