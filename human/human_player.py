"""
人类玩家控制器
处理键盘输入，将按键映射为蛇的相对动作
"""

import pygame
from env.snake_env import Direction, CLOCKWISE


class HumanPlayer:
    """
    将键盘输入转换为贪吃蛇相对动作的控制器

    支持按键:
        - 方向键（↑ ↓ ← →）
        - WASD
        - Space : 暂停 / 继续
        - R     : 请求重置（调用方负责处理 reset_requested 标志）
        - Q / Esc : 请求退出

    使用::

        player = HumanPlayer()
        # 在每帧的开头调用
        player.process_events(current_snake_direction)
        if player.quit_requested:
            break
        action = player.get_action(current_snake_direction)
    """

    # 按键 → 绝对方向映射
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

    def __init__(self):
        self._desired_dir: Direction = None   # 玩家期望的下一步方向
        self.paused        = False
        self.reset_requested = False
        self.quit_requested  = False

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def process_events(self, current_dir: Direction):
        """
        消费 pygame 事件队列，更新内部状态标志

        Args:
            current_dir: 蛇当前移动方向（用于过滤 180° 反向操作）
        """
        self.reset_requested = False   # 每帧重置一次

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                return

            if event.type == pygame.KEYDOWN:
                if event.key in self.KEY_MAP:
                    target = self.KEY_MAP[event.key]
                    # 禁止直接 180° 反向
                    cur_idx = CLOCKWISE.index(current_dir)
                    tgt_idx = CLOCKWISE.index(target)
                    if (tgt_idx - cur_idx) % 4 != 2:
                        self._desired_dir = target

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self.reset_requested = True
                    self._desired_dir    = None

                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.quit_requested = True

    def get_action(self, current_dir: Direction) -> int:
        """
        根据玩家最近一次有效按键返回相对动作

        Args:
            current_dir: 蛇当前移动方向

        Returns:
            int: 0=直走, 1=右转, 2=左转
        """
        if self._desired_dir is None:
            return 0   # 无输入时保持直走

        return self._dir_to_action(current_dir, self._desired_dir)

    def reset(self):
        """重置控制器状态（新局开始时调用）"""
        self._desired_dir    = None
        self.paused          = False
        self.reset_requested = False
        self.quit_requested  = False

    # ------------------------------------------------------------------
    # 静态工具
    # ------------------------------------------------------------------

    @staticmethod
    def _dir_to_action(current: Direction, target: Direction) -> int:
        """将绝对方向转换为相对动作（0直/1右转/2左转）"""
        cur_idx = CLOCKWISE.index(current)
        tgt_idx = CLOCKWISE.index(target)
        diff = (tgt_idx - cur_idx) % 4
        if diff == 0: return 0
        if diff == 1: return 1
        if diff == 3: return 2
        return 0   # 180°（已在 process_events 中过滤，此处保底直走）
