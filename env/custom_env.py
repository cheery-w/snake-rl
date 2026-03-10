"""
自定义规则游戏环境
在标准 SnakeGame 基础上支持：
    - 自定义网格大小
    - 多个食物
    - 障碍物（静态墙）
    - 可配置的奖励机制
    - 可配置的最大步数倍数
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional

from env.snake_env import SnakeGame, Direction, CLOCKWISE


class CustomEnv(SnakeGame):
    """
    可自定义规则的贪吃蛇环境

    在 SnakeGame 基础上扩展:
        multi_food    同时存在多个食物（每个都给奖励）
        obstacles     静态障碍物格子列表（撞到即死）
        reward_food   吃食物的奖励（默认 10.0）
        reward_death  死亡惩罚（默认 -10.0）
        step_limit    最大步数 = step_limit_factor * 蛇长度

    使用::

        rules = CustomRules(
            multi_food=2,
            obstacles="border",   # "none" | "border" | "cross" | list[list[int]]
            reward_food=15.0,
        )
        env = CustomEnv(rules=rules)
        state = env.reset()
        state, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        cols:              int   = 20,
        rows:              int   = 20,
        cell:              int   = 20,
        multi_food:        int   = 1,
        obstacles:                 "str | List[List[int]]" = "none",
        reward_food:       float = 10.0,
        reward_death:      float = -10.0,
        reward_closer:     float = 0.1,
        reward_farther:    float = -0.1,
        step_limit_factor: int   = 100,
    ):
        """
        Args:
            cols/rows/cell:   网格尺寸
            multi_food:       同时在场的食物数量（≥1）
            obstacles:        障碍物布局
                              "none"   - 无障碍
                              "border" - 内边框墙（留 2 格空间）
                              "cross"  - 十字障碍
                              List[[x,y], ...] - 自定义坐标列表（像素坐标）
            reward_*:         奖励参数
            step_limit_factor: 最大步数 = factor * 蛇身长度
        """
        # 先调用父类 __init__（会调用 reset，暂时用空障碍）
        self._obstacles_raw    = obstacles
        self._multi_food       = max(1, multi_food)
        self._reward_food      = reward_food
        self._reward_death     = reward_death
        self._reward_closer    = reward_closer
        self._reward_farther   = reward_farther
        self._step_limit_factor = step_limit_factor

        # 延迟构建障碍物（需要知道 cell/cols/rows）
        self._obstacles: List[List[int]] = []

        super().__init__(cols, rows, cell)

    # ------------------------------------------------------------------
    # 重写 reset
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """重置游戏，初始化障碍物和多食物"""
        state = super().reset()

        # 重建障碍物（保证 cell/cols/rows 已就绪）
        self._obstacles = self._build_obstacles(self._obstacles_raw)

        # 多食物：食物列表（父类已放了 1 个）
        self._food_list: List[List[int]] = [self.food[:]]
        for _ in range(self._multi_food - 1):
            self._food_list.append(self._place_extra_food())

        return self._build_state()

    # ------------------------------------------------------------------
    # 重写 step（使用自定义奖励和障碍物）
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        old_dist = self._nearest_food_dist()

        self._move(action)
        self.snake.appendleft(self.head[:])

        reward = 0.0
        done   = False

        # 死亡：碰墙、碰自身、碰障碍、超步
        if (self._is_collision()
                or self._hit_obstacle()
                or self.steps > self._step_limit_factor * len(self.snake)):
            reward = self._reward_death
            done   = True
            return self._build_state(), reward, done, {"score": self.score}

        # 吃到食物（检测所有食物）
        eaten_idx = self._check_food_eaten()
        if eaten_idx >= 0:
            self.score += 1
            reward = self._reward_food
            self._respawn_food(eaten_idx)
            # 同步 self.food 指向第一个食物（父类使用）
            self.food = self._food_list[0]
        else:
            self.snake.pop()
            new_dist = self._nearest_food_dist()
            reward = (self._reward_closer if new_dist < old_dist
                      else self._reward_farther)

        return self._build_state(), reward, done, {"score": self.score}

    # ------------------------------------------------------------------
    # 碰撞检测扩展
    # ------------------------------------------------------------------

    def _hit_obstacle(self, point: Optional[List[int]] = None) -> bool:
        p = self.head if point is None else point
        return p in self._obstacles

    def _is_collision(self, point: Optional[List[int]] = None) -> bool:
        """覆盖父类，额外检测障碍物"""
        if super()._is_collision(point):
            return True
        p = self.head if point is None else point
        return p in self._obstacles

    # ------------------------------------------------------------------
    # 食物管理
    # ------------------------------------------------------------------

    def _check_food_eaten(self) -> int:
        """返回被吃掉食物的索引，-1 表示没吃到"""
        for i, f in enumerate(self._food_list):
            if self.head == f:
                return i
        return -1

    def _respawn_food(self, idx: int):
        """重新放置被吃掉的食物"""
        self._food_list[idx] = self._place_extra_food()
        self.food = self._food_list[0]

    def _place_extra_food(self) -> List[int]:
        """在空格随机放置一个食物"""
        snake_set = {tuple(s) for s in self.snake}
        food_set  = {tuple(f) for f in self._food_list}
        obs_set   = {tuple(o) for o in self._obstacles}
        occupied  = snake_set | food_set | obs_set
        while True:
            x = random.randint(0, self.cols - 1) * self.cell
            y = random.randint(0, self.rows - 1) * self.cell
            if (x, y) not in occupied:
                return [x, y]

    def _nearest_food_dist(self) -> int:
        """蛇头到最近食物的曼哈顿距离"""
        if not hasattr(self, "_food_list") or not self._food_list:
            return abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
        return min(
            abs(self.head[0] - f[0]) + abs(self.head[1] - f[1])
            for f in self._food_list
        )

    # ------------------------------------------------------------------
    # 障碍物构建
    # ------------------------------------------------------------------

    def _build_obstacles(self, spec) -> List[List[int]]:
        """根据 spec 构建障碍物坐标列表"""
        if spec == "none" or spec is None:
            return []

        bs = self.cell

        if spec == "border":
            obs = []
            # 内边框（距边界 1 格）
            gap = 4 * bs   # 4 格宽的无障碍区（供蛇转身）
            for x in range(0, self.width, bs):
                for y in [2 * bs, self.height - 3 * bs]:
                    if abs(x - self.width // 2) > gap // 2:
                        obs.append([x, y])
            for y in range(0, self.height, bs):
                for x in [2 * bs, self.width - 3 * bs]:
                    if abs(y - self.height // 2) > gap // 2:
                        obs.append([x, y])
            return obs

        if spec == "cross":
            obs = []
            mid_x = (self.cols // 2) * bs
            mid_y = (self.rows // 2) * bs
            arm   = 3   # 十字臂长（格数）
            for i in range(-arm, arm + 1):
                obs.append([mid_x + i * bs, mid_y])
                obs.append([mid_x, mid_y + i * bs])
            return obs

        if isinstance(spec, list):
            return [list(p) for p in spec]

        return []

    # ------------------------------------------------------------------
    # 覆盖 _place_food（父类 reset 时用）
    # ------------------------------------------------------------------

    def _place_food(self):
        """父类 reset 调用时放置主食物"""
        snake_set = {tuple(s) for s in self.snake} if hasattr(self, "snake") else set()
        obs_set   = {tuple(o) for o in self._obstacles} if self._obstacles else set()
        occupied  = snake_set | obs_set
        while True:
            x = random.randint(0, self.cols - 1) * self.cell
            y = random.randint(0, self.rows - 1) * self.cell
            if (x, y) not in occupied:
                self.food = [x, y]
                return

    # ------------------------------------------------------------------
    # 属性（供渲染器访问）
    # ------------------------------------------------------------------

    @property
    def food_list(self) -> List[List[int]]:
        """所有食物坐标"""
        return getattr(self, "_food_list", [self.food])

    @property
    def obstacles(self) -> List[List[int]]:
        """障碍物坐标列表"""
        return self._obstacles
