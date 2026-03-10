"""
网络通信协议
定义客户端 ↔ 服务器之间的消息格式（JSON over TCP）
"""

import json
import struct
from typing import Any, Dict


# ── 消息类型常量 ───────────────────────────────────────────────────────
MSG_JOIN      = "join"       # 客户端加入游戏
MSG_READY     = "ready"      # 客户端就绪
MSG_ACTION    = "action"     # 客户端发送动作
MSG_STATE     = "state"      # 服务器推送游戏状态
MSG_RESULT    = "result"     # 对局结束结果
MSG_ERROR     = "error"      # 错误消息
MSG_PING      = "ping"       # 心跳
MSG_PONG      = "pong"       # 心跳响应
MSG_DISCONNECT = "disconnect" # 断开连接


def encode(msg_type: str, payload: Dict[str, Any] = None) -> bytes:
    """
    将消息编码为带长度前缀的字节流

    格式: [4字节长度 big-endian][JSON字节]

    Args:
        msg_type: 消息类型（见 MSG_* 常量）
        payload:  消息体字典

    Returns:
        带长度前缀的字节数据
    """
    data = {"type": msg_type}
    if payload:
        data.update(payload)
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    header = struct.pack(">I", len(body))
    return header + body


def decode(raw: bytes) -> Dict[str, Any]:
    """
    解码消息（不含长度前缀）

    Args:
        raw: JSON 字节数据

    Returns:
        消息字典，包含 "type" 键
    """
    return json.loads(raw.decode("utf-8"))


def recv_message(sock) -> Dict[str, Any]:
    """
    从 socket 读取一条完整消息（阻塞）

    Args:
        sock: socket 对象

    Returns:
        消息字典；连接断开时返回 None
    """
    try:
        header = _recv_exact(sock, 4)
        if header is None:
            return None
        length = struct.unpack(">I", header)[0]
        if length == 0 or length > 1_048_576:   # 最大 1MB 防护
            return None
        body = _recv_exact(sock, length)
        if body is None:
            return None
        return decode(body)
    except Exception:
        return None


def send_message(sock, msg_type: str, payload: Dict[str, Any] = None) -> bool:
    """
    通过 socket 发送一条消息

    Returns:
        True 表示发送成功
    """
    try:
        sock.sendall(encode(msg_type, payload))
        return True
    except Exception:
        return False


def _recv_exact(sock, n: int):
    """精确读取 n 字节"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ── 标准消息构造函数 ───────────────────────────────────────────────────

def make_join(player_name: str = "Player") -> bytes:
    return encode(MSG_JOIN, {"name": player_name})


def make_action(action: int) -> bytes:
    return encode(MSG_ACTION, {"action": action})


def make_state(
    snakes:    list,
    food:      list,
    scores:    list,
    dones:     list,
    step:      int,
) -> bytes:
    return encode(MSG_STATE, {
        "snakes": snakes,
        "food":   food,
        "scores": scores,
        "dones":  dones,
        "step":   step,
    })


def make_result(winner: int, scores: list) -> bytes:
    return encode(MSG_RESULT, {"winner": winner, "scores": scores})
