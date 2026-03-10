"""
在线对战服务器
处理多个客户端连接，同步游戏状态，主持游戏逻辑

用法:
    python network/server.py
    python network/server.py --host 0.0.0.0 --port 9999 --players 2
"""

import socket
import threading
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from network.protocol import (
    recv_message, send_message,
    MSG_JOIN, MSG_READY, MSG_ACTION, MSG_PING, MSG_DISCONNECT,
    make_state, make_result,
)
from env.versus_env import VersusEnv
from config import Config


class ClientSession:
    """管理单个客户端连接"""

    def __init__(self, conn: socket.socket, addr, player_id: int):
        self.conn      = conn
        self.addr      = addr
        self.player_id = player_id
        self.name      = f"Player{player_id}"
        self.ready     = False
        self.action    = 0     # 最新动作
        self.alive     = True
        self._lock     = threading.Lock()

    def send(self, msg_type: str, payload=None) -> bool:
        return send_message(self.conn, msg_type, payload)

    def close(self):
        self.alive = False
        try:
            self.conn.close()
        except Exception:
            pass


class GameServer:
    """
    贪吃蛇在线对战服务器

    流程:
        1. 等待 n_players 个客户端连接并发送 JOIN
        2. 所有客户端准备就绪后开始游戏
        3. 每帧收集所有客户端的 ACTION，推进游戏状态
        4. 广播新状态给所有客户端
        5. 游戏结束后发送 RESULT，支持再次开局
    """

    def __init__(
        self,
        host:      str = "0.0.0.0",
        port:      int = 9999,
        n_players: int = 2,
        fps:       int = 10,
    ):
        self.host      = host
        self.port      = port
        self.n_players = n_players
        self.fps       = fps
        self.tick_time = 1.0 / fps

        self.sessions: list = []
        self.cfg       = Config()
        self._stop     = False

    def start(self):
        """启动服务器主循环"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(self.n_players)
            print(f"[Server] 监听 {self.host}:{self.port}  "
                  f"等待 {self.n_players} 位玩家...")

            while not self._stop:
                # 等待足够的客户端连接
                self.sessions.clear()
                while len(self.sessions) < self.n_players:
                    try:
                        srv.settimeout(1.0)
                        conn, addr = srv.accept()
                    except socket.timeout:
                        continue
                    pid = len(self.sessions)
                    sess = ClientSession(conn, addr, pid)
                    self.sessions.append(sess)
                    threading.Thread(
                        target=self._handle_client,
                        args=(sess,),
                        daemon=True,
                    ).start()
                    print(f"[Server] 玩家 {pid} 已连接: {addr}")

                print("[Server] 所有玩家已连接，等待就绪...")
                self._wait_ready()
                print("[Server] 游戏开始！")
                self._run_game()

    def _handle_client(self, sess: ClientSession):
        """客户端接收线程"""
        while sess.alive:
            msg = recv_message(sess.conn)
            if msg is None:
                print(f"[Server] 玩家 {sess.player_id} 断开连接")
                sess.alive = False
                break

            t = msg.get("type")
            if t == MSG_JOIN:
                sess.name  = msg.get("name", sess.name)
                sess.ready = False
                sess.send("welcome", {"player_id": sess.player_id,
                                      "n_players": self.n_players})
            elif t == MSG_READY:
                sess.ready = True
            elif t == MSG_ACTION:
                sess.action = int(msg.get("action", 0))
            elif t == MSG_PING:
                sess.send("pong")
            elif t == MSG_DISCONNECT:
                sess.alive = False
                break

    def _wait_ready(self, timeout: float = 30.0):
        """等待所有玩家发送 READY"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(s.ready for s in self.sessions):
                return
            time.sleep(0.1)
        # 超时强制开始
        print("[Server] 等待超时，强制开始")

    def _run_game(self):
        """游戏主循环"""
        env    = VersusEnv(self.cfg.GRID_COLS, self.cfg.GRID_ROWS,
                           self.cfg.CELL_SIZE)
        states = env.reset()
        step   = 0

        while True:
            t0      = time.time()
            actions = [s.action for s in self.sessions]

            s1, s2, d1, d2, info = env.step(actions[0], actions[1])
            step += 1

            # 广播状态
            snakes_data = [
                [list(seg) for seg in env.snake1],
                [list(seg) for seg in env.snake2],
            ]
            state_bytes = make_state(
                snakes=snakes_data,
                food=list(env.food),
                scores=[info["score_ai"], info["score_human"]],
                dones=[d1, d2],
                step=step,
            )
            for sess in self.sessions:
                if sess.alive:
                    try:
                        sess.conn.sendall(state_bytes)
                    except Exception:
                        sess.alive = False

            # 检查结束
            if d1 and d2:
                scores  = [info["score_ai"], info["score_human"]]
                winner  = 0 if scores[0] >= scores[1] else 1
                result  = make_result(winner=winner, scores=scores)
                for sess in self.sessions:
                    if sess.alive:
                        try:
                            sess.conn.sendall(result)
                        except Exception:
                            pass
                print(f"[Server] 游戏结束  Step={step}  "
                      f"Scores={scores}  Winner=Player{winner}")
                break

            # 检查断线
            if any(not s.alive for s in self.sessions):
                print("[Server] 有玩家断线，终止当前对局")
                break

            # 控制帧率
            elapsed = time.time() - t0
            sleep   = self.tick_time - elapsed
            if sleep > 0:
                time.sleep(sleep)


def parse_args():
    p = argparse.ArgumentParser(description="贪吃蛇在线对战服务器")
    p.add_argument("--host",    type=str, default="0.0.0.0")
    p.add_argument("--port",    type=int, default=9999)
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--fps",     type=int, default=10)
    return p.parse_args()


def main():
    args   = parse_args()
    server = GameServer(args.host, args.port, args.players, args.fps)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[Server] 已停止")


if __name__ == "__main__":
    main()
