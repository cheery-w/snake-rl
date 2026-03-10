"""
在线对战入口

用法:
    # 启动服务器（一台机器上运行一次）
    python online_play.py --server

    # 连接服务器（每位玩家各运行一次）
    python online_play.py --host 127.0.0.1 --name 我的名字

    # 服务器和客户端同机测试
    python online_play.py --server &
    python online_play.py --host 127.0.0.1 --name Player1
"""

import argparse
import sys
import os
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="贪吃蛇在线对战")
    p.add_argument("--server",  action="store_true",
                   help="以服务器模式启动")
    p.add_argument("--host",    type=str, default="127.0.0.1",
                   help="服务器地址（客户端模式）")
    p.add_argument("--port",    type=int, default=9999)
    p.add_argument("--name",    type=str, default="Player",
                   help="玩家昵称")
    p.add_argument("--players", type=int, default=2,
                   help="服务器等待的玩家数（服务器模式）")
    p.add_argument("--fps",     type=int, default=10,
                   help="游戏帧率")
    return p.parse_args()


def run_server(args):
    from network.server import GameServer
    server = GameServer(
        host="0.0.0.0",
        port=args.port,
        n_players=args.players,
        fps=args.fps,
    )
    print(f"服务器启动  端口={args.port}  等待 {args.players} 位玩家")
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n服务器已停止")


def run_client(args):
    from network.client import OnlineClient
    client = OnlineClient(
        host=args.host,
        port=args.port,
        player_name=args.name,
        fps=30,
    )
    client.run()


def main_client():
    """由 main.py 调用的默认客户端入口"""
    import types
    args = types.SimpleNamespace(
        host="127.0.0.1", port=9999, name="Player", fps=30
    )
    run_client(args)


def main():
    args = parse_args()
    if args.server:
        run_server(args)
    else:
        run_client(args)


if __name__ == "__main__":
    main()
