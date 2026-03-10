"""
人类玩家游戏环境
在 SnakeGame 基础上集成 pygame 键盘输入，专为人类玩家设计
"""

import pygame
from env.snake_env import SnakeGame, Direction, CLOCKWISE


class HumanEnv(SnakeGame):
    """
    面向人类玩家的贪吃蛇环境

    继承自 SnakeGame，额外提供:
        - 键盘事件处理（方向键 / WASD）
        - 暂停 / 恢复（Space）
        - 重置请求检测（R）
        - 退出请求检测（Q / Esc）

    典型用法::

        env = HumanEnv()
        state = env.reset()

        while True:
            env.handle_events()
            if env.quit_requested:
                break
            if env.reset_requested:
                state = env.reset()
                continue
            if env.paused:
                renderer.render()
                continue
            action = env.get_human_action()
            state, reward, done, info = env.step(action)
            renderer.render()
    """

    # 按键 → 绝对方向
    KEY_MAP = {
        pygame.K_RIGHT: Direction.RIGHT,
        pygame.K_LEFT:  Direction.LEFT,
        pygame.K_UP:    Direction.UP,
        pygame.K_DOWN:  Direction.DOWN,
        pygame.K_d:     Direction.RIGHT,
        pygame.K_a:     Direction.LEFT,
        pygame.K_w:     Direction.UP,
        pygame.K_s:     Direction.DOWN,
    }

    def __init__(self, cols: int = 20, rows: int = 20, cell: int = 20):
        super().__init__(cols, rows, cell)
        self._desired_dir: Direction = None
        self.paused          = False
        self.quit_requested  = False
        self.reset_requested = False

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def reset(self):
        state = super().reset()
        self._desired_dir    = None
        self.paused          = False
        self.reset_requested = False
        return state

    def handle_events(self):
        """
        消费 pygame 事件队列，更新控制标志

        应在每帧的开头、step() 之前调用一次
        """
        self.reset_requested = False   # 每帧清空一次

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                return

            if event.type == pygame.KEYDOWN:
                if event.key in self.KEY_MAP:
                    target = self.KEY_MAP[event.key]
                    # 禁止 180° 直接反向
                    cur_idx = CLOCKWISE.index(self.direction)
                    tgt_idx = CLOCKWISE.index(target)
                    if (tgt_idx - cur_idx) % 4 != 2:
                        self._desired_dir = target

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self.reset_requested = True

                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.quit_requested = True

    def get_human_action(self) -> int:
        """
        根据最近一次有效按键返回相对动作

        Returns:
            int: 0=直走, 1=右转, 2=左转
        """
        if self._desired_dir is None:
            return 0   # 无输入时保持直走

        cur_idx = CLOCKWISE.index(self.direction)
        tgt_idx = CLOCKWISE.index(self._desired_dir)
        diff = (tgt_idx - cur_idx) % 4
        if diff == 0: return 0
        if diff == 1: return 1
        if diff == 3: return 2
        return 0
