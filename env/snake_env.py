"""
贪吃蛇游戏环境（纯逻辑，不含渲染）
提供与 OpenAI Gym 风格一致的接口
"""

import numpy as np
import random
from collections import deque
from enum import Enum
from typing import Tuple, List, Optional


class Direction(Enum):
    RIGHT = 0
    DOWN  = 1
    LEFT  = 2
    UP    = 3

# 顺时针方向列表，用于相对动作转换
CLOCKWISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


class SnakeGame:
    """
    贪吃蛇核心环境

    动作空间（3个离散动作）:
        0 - 直走（保持当前方向）
        1 - 右转（顺时针 90°）
        2 - 左转（逆时针 90°）

    状态空间（11 维二值向量）:
        [0] 直走方向有危险
        [1] 右转方向有危险
        [2] 左转方向有危险
        [3] 当前方向：左
        [4] 当前方向：右
        [5] 当前方向：上
        [6] 当前方向：下
        [7] 食物在左
        [8] 食物在右
        [9] 食物在上
        [10] 食物在下
    """

    def __init__(self, cols: int = 20, rows: int = 20, cell: int = 20):
        self.cols = cols
        self.rows = rows
        self.cell = cell
        self.width  = cols * cell
        self.height = rows * cell
        self.reset()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """重置游戏，返回初始状态向量"""
        cx = (self.cols // 2) * self.cell
        cy = (self.rows // 2) * self.cell

        self.direction = Direction.RIGHT
        self.head      = [cx, cy]
        # 初始蛇身长度为 3
        self.snake = deque([
            [cx,                    cy],
            [cx - self.cell,        cy],
            [cx - 2 * self.cell,    cy],
        ])
        self.score      = 0
        self.steps      = 0
        self._place_food()
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步

        Args:
            action: 0=直走, 1=右转, 2=左转

        Returns:
            next_state, reward, done, info
        """
        self.steps += 1
        old_dist = self._manhattan_dist()

        # 移动蛇头
        self._move(action)
        self.snake.appendleft(self.head[:])

        reward = 0.0
        done   = False

        # 死亡判定
        if self._is_collision() or self.steps > 100 * len(self.snake):
            reward = -10.0
            done   = True
            return self._build_state(), reward, done, {"score": self.score}

        # 吃到食物
        if self.head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()
            # 距离塑形奖励
            new_dist = self._manhattan_dist()
            reward = 0.1 if new_dist < old_dist else -0.1

        return self._build_state(), reward, done, {"score": self.score}

    # ------------------------------------------------------------------
    # 状态构建
    # ------------------------------------------------------------------

    def _build_state(self) -> np.ndarray:
        h  = self.head
        d  = self.direction
        bs = self.cell

        # 四个相邻格
        pt_l = [h[0] - bs, h[1]]
        pt_r = [h[0] + bs, h[1]]
        pt_u = [h[0],      h[1] - bs]
        pt_d = [h[0],      h[1] + bs]

        dir_l = (d == Direction.LEFT)
        dir_r = (d == Direction.RIGHT)
        dir_u = (d == Direction.UP)
        dir_d = (d == Direction.DOWN)

        # 三个相对方向的危险检测
        danger_straight = (
            (dir_r and self._is_collision(pt_r)) or
            (dir_l and self._is_collision(pt_l)) or
            (dir_u and self._is_collision(pt_u)) or
            (dir_d and self._is_collision(pt_d))
        )
        danger_right = (
            (dir_u and self._is_collision(pt_r)) or
            (dir_d and self._is_collision(pt_l)) or
            (dir_l and self._is_collision(pt_u)) or
            (dir_r and self._is_collision(pt_d))
        )
        danger_left = (
            (dir_d and self._is_collision(pt_r)) or
            (dir_u and self._is_collision(pt_l)) or
            (dir_r and self._is_collision(pt_u)) or
            (dir_l and self._is_collision(pt_d))
        )

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l), int(dir_r), int(dir_u), int(dir_d),
            int(self.food[0] < h[0]),   # 食物在左
            int(self.food[0] > h[0]),   # 食物在右
            int(self.food[1] < h[1]),   # 食物在上（y轴向下）
            int(self.food[1] > h[1]),   # 食物在下
        ]
        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _place_food(self):
        """在不与蛇身重叠的位置随机放置食物"""
        snake_set = {tuple(s) for s in self.snake}
        while True:
            x = random.randint(0, self.cols - 1) * self.cell
            y = random.randint(0, self.rows - 1) * self.cell
            if (x, y) not in snake_set:
                self.food = [x, y]
                return

    def _is_collision(self, point: Optional[List[int]] = None) -> bool:
        """检测碰撞（越界或撞到蛇身）"""
        p = self.head if point is None else point
        x, y = p
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # 跳过蛇头自身（索引 0）
        body = list(self.snake)
        if p in body[1:]:
            return True
        return False

    def _move(self, action: int):
        """根据相对动作更新方向，然后移动蛇头"""
        idx = CLOCKWISE.index(self.direction)
        if   action == 1: idx = (idx + 1) % 4   # 右转
        elif action == 2: idx = (idx - 1) % 4   # 左转
        # action == 0: 直走，方向不变
        self.direction = CLOCKWISE[idx]

        bs = self.cell
        x, y = self.head
        if   self.direction == Direction.RIGHT: x += bs
        elif self.direction == Direction.LEFT:  x -= bs
        elif self.direction == Direction.DOWN:  y += bs
        elif self.direction == Direction.UP:    y -= bs
        self.head = [x, y]

    def _manhattan_dist(self) -> int:
        return abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        return 11

    @property
    def action_size(self) -> int:
        return 3

    @property
    def snake_length(self) -> int:
        return len(self.snake)
