"""
人机对战环境
两条蛇共享同一棋盘，同步执行动作

颜色约定（供渲染器使用）:
    蛇1 (AI)    → 绿色
    蛇2 (Human) → 蓝色
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List, Optional

from env.snake_env import Direction, CLOCKWISE


class VersusEnv:
    """
    人机对战环境 — 两条蛇同时游戏

    状态空间（每条蛇 11 维，与 SnakeGame 相同）:
        [危险直/右/左, 方向左/右/上/下, 食物左/右/上/下]
        危险检测包含对方蛇身，使 AI 能感知对手位置

    动作空间（每条蛇 3 个相对动作）:
        0=直走  1=右转  2=左转

    使用::

        env = VersusEnv()
        state_ai, state_human = env.reset()

        while not (env.done1 and env.done2):
            action_ai    = agent.select_action(state_ai)
            action_human = player.get_action(env.dir2)
            state_ai, state_human, d1, d2, info = env.step(action_ai, action_human)
    """

    def __init__(self, cols: int = 20, rows: int = 20, cell: int = 20):
        self.cols   = cols
        self.rows   = rows
        self.cell   = cell
        self.width  = cols * cell
        self.height = rows * cell
        self.reset()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        重置环境，两条蛇分别从棋盘左右两侧出发

        Returns:
            (state_ai, state_human)
        """
        bs = self.cell

        # 蛇1 (AI): 从左 1/4 处出发，向右移动
        cx1 = (self.cols // 4) * bs
        cy  = (self.rows // 2) * bs
        self.dir1  = Direction.RIGHT
        self.head1 = [cx1, cy]
        self.snake1 = deque([
            [cx1,           cy],
            [cx1 - bs,      cy],
            [cx1 - 2 * bs,  cy],
        ])

        # 蛇2 (Human): 从右 3/4 处出发，向左移动
        cx2 = (self.cols * 3 // 4) * bs
        self.dir2  = Direction.LEFT
        self.head2 = [cx2, cy]
        self.snake2 = deque([
            [cx2,           cy],
            [cx2 + bs,      cy],
            [cx2 + 2 * bs,  cy],
        ])

        self.score1 = 0
        self.score2 = 0
        self.steps1 = 0
        self.steps2 = 0
        self.done1  = False
        self.done2  = False

        self._place_food()

        s1 = self._build_state(self.head1, self.dir1, self.snake1, self.snake2)
        s2 = self._build_state(self.head2, self.dir2, self.snake2, self.snake1)
        return s1, s2

    def step(
        self,
        action_ai: int,
        action_human: int,
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        """
        两条蛇同步执行动作

        Args:
            action_ai:    AI 动作    (0=直走, 1=右转, 2=左转)
            action_human: 人类动作   (0=直走, 1=右转, 2=左转)

        Returns:
            (state_ai, state_human, done_ai, done_human, info)
            info = {"score_ai": int, "score_human": int}
        """
        # 已全部结束时直接返回
        if self.done1 and self.done2:
            s1 = self._build_state(self.head1, self.dir1, self.snake1, self.snake2)
            s2 = self._build_state(self.head2, self.dir2, self.snake2, self.snake1)
            return s1, s2, True, True, self._info()

        # ── 计算新头部坐标（不移动蛇）────────────────────────────────
        new_head1 = (None if self.done1 else
                     self._calc_new_head(self.head1, self.dir1, action_ai))
        new_head2 = (None if self.done2 else
                     self._calc_new_head(self.head2, self.dir2, action_human))

        # ── 预判食物 ────────────────────────────────────────────────
        ate1 = (not self.done1) and (new_head1 == self.food)
        ate2 = (not self.done2) and (new_head2 == self.food)

        # 双方同时到达食物时随机决定
        if ate1 and ate2:
            if random.random() < 0.5:
                ate2 = False
            else:
                ate1 = False

        # ── 移动蛇1 ─────────────────────────────────────────────────
        if not self.done1:
            self.dir1 = self._update_dir(self.dir1, action_ai)
            self.steps1 += 1
            if not ate1:
                self.snake1.pop()
            self.snake1.appendleft(new_head1[:])
            self.head1 = new_head1

        # ── 移动蛇2 ─────────────────────────────────────────────────
        if not self.done2:
            self.dir2 = self._update_dir(self.dir2, action_human)
            self.steps2 += 1
            if not ate2:
                self.snake2.pop()
            self.snake2.appendleft(new_head2[:])
            self.head2 = new_head2

        # ── 死亡检测 ─────────────────────────────────────────────────
        if not self.done1:
            self.done1 = self._is_dead(self.head1, self.snake1,
                                       self.snake2, self.steps1)
        if not self.done2:
            self.done2 = self._is_dead(self.head2, self.snake2,
                                       self.snake1, self.steps2)

        # 头碰头：双方同时死亡
        if (not self.done1 and not self.done2 and
                self.head1 is not None and self.head2 is not None and
                self.head1 == self.head2):
            self.done1 = True
            self.done2 = True

        # ── 得分与食物刷新 ────────────────────────────────────────────
        if ate1 and not self.done1:
            self.score1 += 1
            self._place_food()
        elif ate2 and not self.done2:
            self.score2 += 1
            self._place_food()

        s1 = self._build_state(self.head1, self.dir1, self.snake1, self.snake2)
        s2 = self._build_state(self.head2, self.dir2, self.snake2, self.snake1)
        return s1, s2, self.done1, self.done2, self._info()

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _info(self) -> dict:
        return {"score_ai": self.score1, "score_human": self.score2}

    def _calc_new_head(self, head: list, direction: Direction,
                       action: int) -> list:
        """根据相对动作计算下一帧的头部坐标"""
        new_dir = self._update_dir(direction, action)
        bs = self.cell
        x, y = head
        if   new_dir == Direction.RIGHT: x += bs
        elif new_dir == Direction.LEFT:  x -= bs
        elif new_dir == Direction.DOWN:  y += bs
        elif new_dir == Direction.UP:    y -= bs
        return [x, y]

    def _update_dir(self, direction: Direction, action: int) -> Direction:
        """根据相对动作更新绝对方向"""
        idx = CLOCKWISE.index(direction)
        if   action == 1: idx = (idx + 1) % 4   # 右转
        elif action == 2: idx = (idx - 1) % 4   # 左转
        return CLOCKWISE[idx]

    def _is_dead(self, head: list, own_snake: deque,
                 other_snake: deque, steps: int) -> bool:
        """检查蛇是否死亡（越界 / 撞自身 / 撞对方 / 超步）"""
        x, y = head
        # 越界
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # 撞自身（跳过头部 [0]）
        body = list(own_snake)
        if head in body[1:]:
            return True
        # 撞对方蛇身（含对方头部）
        if head in list(other_snake):
            return True
        # 步数超限（防止无限循环）
        if steps > 100 * len(own_snake):
            return True
        return False

    def _place_food(self):
        """在不与任何蛇重叠的格子随机放置食物"""
        occupied = ({tuple(s) for s in self.snake1} |
                    {tuple(s) for s in self.snake2})
        while True:
            x = random.randint(0, self.cols - 1) * self.cell
            y = random.randint(0, self.rows - 1) * self.cell
            if (x, y) not in occupied:
                self.food = [x, y]
                return

    def _point_is_dangerous(self, point: list,
                             own_snake: deque,
                             other_snake: deque) -> bool:
        """判断某坐标对某条蛇是否危险（含对方蛇身）"""
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        body = list(own_snake)
        if point in body[1:]:
            return True
        if point in list(other_snake):
            return True
        return False

    def _build_state(self, head: list, direction: Direction,
                     own_snake: deque,
                     other_snake: deque) -> np.ndarray:
        """构建 11 维状态向量（与 SnakeGame 格式一致）"""
        bs = self.cell
        h  = head
        d  = direction

        pt_l = [h[0] - bs, h[1]]
        pt_r = [h[0] + bs, h[1]]
        pt_u = [h[0],      h[1] - bs]
        pt_d = [h[0],      h[1] + bs]

        dir_l = (d == Direction.LEFT)
        dir_r = (d == Direction.RIGHT)
        dir_u = (d == Direction.UP)
        dir_d = (d == Direction.DOWN)

        def danger(pt):
            return self._point_is_dangerous(pt, own_snake, other_snake)

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

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        return 11

    @property
    def action_size(self) -> int:
        return 3
