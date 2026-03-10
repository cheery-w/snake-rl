"""
在线对战客户端
连接服务器，接收游戏状态并渲染，发送玩家动作

用法:
    python network/client.py --host 127.0.0.1 --port 9999 --name 我的名字
"""

import socket
import threading
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame

from network.protocol import (
    recv_message, send_message,
    MSG_STATE, MSG_RESULT, MSG_DISCONNECT,
    make_join, make_action,
)
from env.snake_env import Direction, CLOCKWISE


# ── 颜色 ──────────────────────────────────────────────────────────────
_BG     = (15,  15,  15)
_GRID   = (30,  30,  30)
_FOOD   = (220,  50,  50)
_TEXT   = (200, 200, 200)
_DIM    = ( 80,  80,  80)
_P0_H   = (  0, 220,  80)
_P0_B   = (  0, 160,  60)
_P1_H   = ( 50, 130, 255)
_P1_B   = ( 30,  85, 200)
_GOLD   = (255, 215,   0)

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


class OnlineClient:
    """
    在线对战客户端

    职责:
        1. 连接服务器并登录
        2. 独立线程持续接收服务器状态
        3. 主线程处理键盘输入 + 渲染
        4. 将玩家动作异步发送给服务器
    """

    CELL   = 20
    COLS   = 20
    ROWS   = 20
    W      = COLS * CELL
    H      = ROWS * CELL

    def __init__(
        self,
        host:        str = "127.0.0.1",
        port:        int = 9999,
        player_name: str = "Player",
        fps:         int = 30,
    ):
        self.host        = host
        self.port        = port
        self.player_name = player_name
        self.fps         = fps

        self.sock:       socket.socket = None
        self.player_id:  int           = -1
        self.connected:  bool          = False

        # 最新游戏状态（由接收线程写入）
        self._state_lock = threading.Lock()
        self._last_state = None
        self._game_over  = False
        self._result     = None

        # 玩家输入
        self._next_dir:  Direction = Direction.RIGHT
        self._cur_dir:   Direction = Direction.RIGHT

    # ------------------------------------------------------------------
    # 连接 & 通信
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.connected = True

            # 发送 JOIN
            self.sock.sendall(make_join(self.player_name))

            # 等待 welcome
            msg = recv_message(self.sock)
            if msg and msg.get("type") == "welcome":
                self.player_id = msg.get("player_id", 0)
                print(f"[Client] 已连接  player_id={self.player_id}")
            return True
        except Exception as e:
            print(f"[Client] 连接失败: {e}")
            return False

    def _recv_loop(self):
        """持续接收服务器消息"""
        while self.connected:
            msg = recv_message(self.sock)
            if msg is None:
                self.connected = False
                break
            t = msg.get("type")
            if t == MSG_STATE:
                with self._state_lock:
                    self._last_state = msg
            elif t == MSG_RESULT:
                with self._state_lock:
                    self._result    = msg
                    self._game_over = True

    def _send_action(self, action: int):
        """异步发送动作"""
        if self.connected:
            send_message(self.sock, "action", {"action": action})

    # ------------------------------------------------------------------
    # 主循环（pygame）
    # ------------------------------------------------------------------

    def run(self):
        if not self.connect():
            return

        pygame.init()
        screen = pygame.display.set_mode((self.W, self.H + 60))
        pygame.display.set_caption(f"Snake Online — {self.player_name}")
        clock  = pygame.time.Clock()
        font_s = pygame.font.SysFont("Consolas", 14)
        font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        font_l = pygame.font.SysFont("Consolas", 30, bold=True)

        # 启动接收线程
        recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        recv_thread.start()

        # 发送 READY
        send_message(self.sock, "ready")

        running = True
        while running:
            # ── 事件处理 ──────────────────���───────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in _KEY_MAP:
                        target  = _KEY_MAP[event.key]
                        cur_idx = CLOCKWISE.index(self._cur_dir)
                        tgt_idx = CLOCKWISE.index(target)
                        if (tgt_idx - cur_idx) % 4 != 2:
                            self._next_dir = target
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

            # ── 计算动作 ──────────────────────────────────────────
            cur_idx = CLOCKWISE.index(self._cur_dir)
            tgt_idx = CLOCKWISE.index(self._next_dir)
            diff    = (tgt_idx - cur_idx) % 4
            action  = (0 if diff == 0 else
                       1 if diff == 1 else
                       2 if diff == 3 else 0)
            self._send_action(action)

            # ── 渲染 ──────────────────────────────────────────────
            with self._state_lock:
                state  = self._last_state
                result = self._result

            screen.fill(_BG)
            self._draw_grid(screen)

            if state:
                self._draw_state(screen, state, font_s, font_m)

            if result:
                self._draw_result(screen, result, font_l, font_s)

            if not self.connected:
                surf = font_m.render("服务器断开连接", True, (220, 80, 80))
                screen.blit(surf, surf.get_rect(center=(self.W // 2, self.H // 2)))

            pygame.display.flip()
            clock.tick(self.fps)

        # 断开连接
        if self.connected:
            send_message(self.sock, MSG_DISCONNECT)
        self.sock.close()
        pygame.quit()

    # ------------------------------------------------------------------
    # 绘制辅助
    # ------------------------------------------------------------------

    def _draw_grid(self, screen):
        for x in range(0, self.W, self.CELL):
            pygame.draw.line(screen, _GRID, (x, 0), (x, self.H))
        for y in range(0, self.H, self.CELL):
            pygame.draw.line(screen, _GRID, (0, y), (self.W, y))

    def _draw_state(self, screen, state: dict, font_s, font_m):
        food = state.get("food", [0, 0])
        cx   = food[0] + self.CELL // 2
        cy   = food[1] + self.CELL // 2
        pygame.draw.circle(screen, _FOOD, (cx, cy), self.CELL // 2 - 2)

        snakes  = state.get("snakes", [])
        dones   = state.get("dones",  [False, False])
        scores  = state.get("scores", [0, 0])
        colors  = [(_P0_H, _P0_B), (_P1_H, _P1_B)]

        for i, segs in enumerate(snakes):
            if i >= len(colors):
                break
            hc, bc = colors[i]
            dead   = dones[i] if i < len(dones) else False
            for j, seg in enumerate(segs):
                color = hc if j == 0 else bc
                if dead:
                    color = tuple(max(0, c - 80) for c in color)
                m    = 1
                rect = pygame.Rect(seg[0] + m, seg[1] + m,
                                   self.CELL - 2 * m, self.CELL - 2 * m)
                pygame.draw.rect(screen, color, rect, border_radius=3)

        # HUD
        y0     = self.H + 8
        labels = ["AI" if self.player_id != 0 else "你",
                  "你" if self.player_id == 1 else "对手"]
        cols2  = [_P0_H, _P1_H]
        for i, (lbl, sc) in enumerate(zip(labels, scores)):
            c    = cols2[i] if not dones[i] else _DIM
            surf = font_m.render(f"{lbl}: {sc}", True, c)
            screen.blit(surf, (10 + i * 200, y0))

    def _draw_result(self, screen, result: dict, font_l, font_s):
        winner = result.get("winner", -1)
        scores = result.get("scores", [0, 0])
        if winner == self.player_id:
            msg = "你赢了！"
            col = _GOLD
        else:
            msg = "你输了"
            col = (180, 80, 80)

        overlay = pygame.Surface((self.W, self.H + 60), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        surf = font_l.render(msg, True, col)
        screen.blit(surf, surf.get_rect(center=(self.W // 2, self.H // 2 - 20)))
        sc_s = font_s.render(f"比分: {scores[0]} : {scores[1]}", True, _TEXT)
        screen.blit(sc_s, sc_s.get_rect(center=(self.W // 2, self.H // 2 + 24)))
        hint = font_s.render("按 Q 退出", True, _DIM)
        screen.blit(hint, hint.get_rect(center=(self.W // 2, self.H // 2 + 54)))


def parse_args():
    p = argparse.ArgumentParser(description="贪吃蛇在线对战客户端")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--name", type=str, default="Player",
                   help="玩家昵称")
    p.add_argument("--fps",  type=int, default=30)
    return p.parse_args()


def main():
    args   = parse_args()
    client = OnlineClient(args.host, args.port, args.name, args.fps)
    client.run()


if __name__ == "__main__":
    main()
