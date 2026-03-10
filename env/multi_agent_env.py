"""
多智能体对战环境
支持 2-4 条蛇在同一棋盘上同时竞争

颜色约定（供渲染器使用）:
    蛇 0 → 绿色   蛇 1 → 蓝色   蛇 2 → 橙色   蛇 3 → 紫色
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional

from env.snake_env import Direction, CLOCKWISE


# 每条蛇的初始方向和起始位置偏移（相对于棋盘中心）
_INIT_CONFIGS = [
    # (x_frac, y_frac, initial_direction)
    (0.25, 0.50, Direction.RIGHT),   # 蛇0：左侧向右
    (0.75, 0.50, Direction.LEFT),    # 蛇1：右侧向左
    (0.50, 0.25, Direction.DOWN),    # 蛇2：上方向下
    (0.50, 0.75, Direction.UP),      # 蛇3：下方向上
]


class Snake:
    """单条蛇的状态容器"""

    __slots__ = ("head", "body", "direction", "score", "steps", "alive")

    def __init__(self, head: list, direction: Direction):
        self.head      = head[:]
        self.direction = direction
        bs             = None   # 在 MultiAgentEnv._init_snake 中设置
        self.body      = deque([head[:]])   # body[0] 是头
        self.score     = 0
        self.steps     = 0
        self.alive     = True


class MultiAgentEnv:
    """
    多智能体贪吃蛇对战环境

    支持 2–4 个智能体，每条蛇独立控制，共享同一棋盘和食物。

    状态空间（每条蛇 11 维，与 SnakeGame 相同）:
        [危险直/右/左, 方向左/右/上/下, 食物左/右/上/下]
        危险检测包含所有其他蛇的蛇身

    动作空间（每条蛇 3 个相对动作）:
        0=直走  1=右转  2=左转

    使用::

        env = MultiAgentEnv(n_agents=4)
        states = env.reset()         # List[np.ndarray], 长度 = n_agents

        while not env.all_done:
            actions = [agent_i.select_action(s) for i, s in enumerate(states)]
            states, dones, info = env.step(actions)
    """

    AGENT_COLORS = [
        ((0,   220,  80), (0,   160,  60)),    # 绿 (head, body)
        ((50,  130, 255), (30,   85, 200)),    # 蓝
        ((255, 140,   0), (200, 100,   0)),    # 橙
        ((180,   0, 180), (130,   0, 130)),    # 紫
    ]

    def __init__(
        self,
        n_agents: int = 2,
        cols:     int = 20,
        rows:     int = 20,
        cell:     int = 20,
    ):
        assert 2 <= n_agents <= 4, "n_agents 必须在 2-4 之间"
        self.n_agents = n_agents
        self.cols     = cols
        self.rows     = rows
        self.cell     = cell
        self.width    = cols * cell
        self.height   = rows * cell

        self.snakes: List[Snake] = []
        self.food:   List[int]   = [0, 0]
        self.reset()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def reset(self) -> List[np.ndarray]:
        """重置环境，返回各智能体的初始状态"""
        bs = self.cell
        self.snakes = []

        for i in range(self.n_agents):
            xf, yf, init_dir = _INIT_CONFIGS[i]
            cx = int(round(xf * (self.cols - 1))) * bs
            cy = int(round(yf * (self.rows - 1))) * bs

            # 确保起点不超边界
            cx = max(2 * bs, min(cx, (self.cols - 3) * bs))
            cy = max(2 * bs, min(cy, (self.rows - 3) * bs))

            snake = Snake([cx, cy], init_dir)
            # 初始长度为 3
            if init_dir == Direction.RIGHT:
                offsets = [[0, 0], [-bs, 0], [-2 * bs, 0]]
            elif init_dir == Direction.LEFT:
                offsets = [[0, 0], [bs, 0],  [2 * bs, 0]]
            elif init_dir == Direction.DOWN:
                offsets = [[0, 0], [0, -bs], [0, -2 * bs]]
            else:   # UP
                offsets = [[0, 0], [0, bs],  [0, 2 * bs]]

            snake.body = deque([[cx + dx, cy + dy] for dx, dy in offsets])
            snake.head = list(snake.body[0])
            self.snakes.append(snake)

        self._place_food()
        return [self._build_state(i) for i in range(self.n_agents)]

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[bool], dict]:
        """
        所有存活蛇同步执行动作

        Args:
            actions: 长度为 n_agents 的动作列表（已死亡蛇的动作被忽略）

        Returns:
            (states, dones, info)
            states: List[np.ndarray]，每条蛇的下一状态
            dones:  List[bool]
            info:   {"scores": List[int]}
        """
        assert len(actions) == self.n_agents

        # ── 计算新头部（不立即移动）─────────────────────────────────
        new_heads: List[Optional[list]] = []
        for i, s in enumerate(self.snakes):
            if not s.alive:
                new_heads.append(None)
                continue
            new_heads.append(self._calc_new_head(s.head, s.direction, actions[i]))

        # ── 检测哪条蛇吃到食物 ────────────────────────────────────────
        ate = [False] * self.n_agents
        food_eaters = [i for i, h in enumerate(new_heads)
                       if h is not None and h == self.food]
        if len(food_eaters) == 1:
            ate[food_eaters[0]] = True
        elif len(food_eaters) > 1:
            # 多蛇同时到食物 → 随机选一条吃到
            winner = random.choice(food_eaters)
            ate[winner] = True

        # ── 移动各蛇 ──────────────────────────────────────────────────
        for i, s in enumerate(self.snakes):
            if not s.alive:
                continue
            s.direction = self._update_dir(s.direction, actions[i])
            s.steps    += 1
            if not ate[i]:
                s.body.pop()
            s.body.appendleft(new_heads[i][:])
            s.head = new_heads[i]

        # ── 死亡检测 ──────────────────────────────────────────────────
        for i, s in enumerate(self.snakes):
            if not s.alive:
                continue
            if self._is_dead(i):
                s.alive = False

        # ── 得分与食物刷新 ────────────────────────────────────────────
        for i, s in enumerate(self.snakes):
            if ate[i] and s.alive:
                s.score += 1
                self._place_food()
                break

        states = [self._build_state(i) for i in range(self.n_agents)]
        dones  = [not s.alive for s in self.snakes]
        info   = {"scores": [s.score for s in self.snakes]}
        return states, dones, info

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def all_done(self) -> bool:
        return all(not s.alive for s in self.snakes)

    @property
    def state_size(self) -> int:
        return 11

    @property
    def action_size(self) -> int:
        return 3

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _is_dead(self, idx: int) -> bool:
        """检测第 idx 条蛇是否死亡"""
        s    = self.snakes[idx]
        h    = s.head
        x, y = h

        # 越界
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        # 撞自身（跳过头部 [0]）
        body_list = list(s.body)
        if h in body_list[1:]:
            return True

        # 撞其他蛇（含对方蛇头）
        for j, other in enumerate(self.snakes):
            if j == idx or not other.alive:
                continue
            if h in list(other.body):
                return True

        # 步数超限（防止死循环）
        if s.steps > 100 * len(s.body):
            return True

        return False

    def _calc_new_head(
        self, head: list, direction: Direction, action: int
    ) -> list:
        new_dir = self._update_dir(direction, action)
        bs = self.cell
        x, y = head
        if   new_dir == Direction.RIGHT: x += bs
        elif new_dir == Direction.LEFT:  x -= bs
        elif new_dir == Direction.DOWN:  y += bs
        elif new_dir == Direction.UP:    y -= bs
        return [x, y]

    def _update_dir(self, direction: Direction, action: int) -> Direction:
        idx = CLOCKWISE.index(direction)
        if   action == 1: idx = (idx + 1) % 4
        elif action == 2: idx = (idx - 1) % 4
        return CLOCKWISE[idx]

    def _place_food(self):
        """在所有蛇体外随机放置食物"""
        occupied = set()
        for s in self.snakes:
            occupied |= {tuple(seg) for seg in s.body}
        while True:
            x = random.randint(0, self.cols - 1) * self.cell
            y = random.randint(0, self.rows - 1) * self.cell
            if (x, y) not in occupied:
                self.food = [x, y]
                return

    def _point_is_dangerous(self, point: list, snake_idx: int) -> bool:
        """判断某坐标对第 snake_idx 条蛇是否危险"""
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        own_body = list(self.snakes[snake_idx].body)
        if point in own_body[1:]:
            return True
        for j, other in enumerate(self.snakes):
            if j == snake_idx or not other.alive:
                continue
            if point in list(other.body):
                return True
        return False

    def _build_state(self, idx: int) -> np.ndarray:
        """构建第 idx 条蛇的 11 维状态向量"""
        s  = self.snakes[idx]
        bs = self.cell
        h  = s.head
        d  = s.direction

        pt_l = [h[0] - bs, h[1]]
        pt_r = [h[0] + bs, h[1]]
        pt_u = [h[0],      h[1] - bs]
        pt_d = [h[0],      h[1] + bs]

        dir_l = (d == Direction.LEFT)
        dir_r = (d == Direction.RIGHT)
        dir_u = (d == Direction.UP)
        dir_d = (d == Direction.DOWN)

        def danger(pt):
            return self._point_is_dangerous(pt, idx)

        danger_straight = (
            (dir_r and danger(pt_r)) or
            (dir_l and danger(pt_l)) or
            (dir_u and danger(pt_u)) or
            (dir_d and danger(pt_d))
        )
        danger_right = (
            (dir_u and danger(pt_r)) or
            (dir_d and danger(pt_l)) or
            (dir_l and danger(pt_u)) or
            (dir_r and danger(pt_d))
        )
        danger_left = (
            (dir_d and danger(pt_r)) or
            (dir_u and danger(pt_l)) or
            (dir_r and danger(pt_u)) or
            (dir_l and danger(pt_d))
        )

        food = self.food
        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l), int(dir_r), int(dir_u), int(dir_d),
            int(food[0] < h[0]),
            int(food[0] > h[0]),
            int(food[1] < h[1]),
            int(food[1] > h[1]),
        ]
        return np.array(state, dtype=np.float32)
