"""
智能体工厂
根据算法类型和难度级别创建对应的智能体，支持加载预训练权重
"""

import os
from typing import Optional

from config import Config
from utils.config_manager import ConfigManager


# 难度 → 对应的 checkpoint 文件名约定
DIFFICULTY_CHECKPOINT_MAP = {
    "easy":   "checkpoints/easy.pt",
    "medium": "checkpoints/medium.pt",
    "hard":   "checkpoints/hard.pt",
    "expert": "checkpoints/expert.pt",
}

# 如果找不到难度专属 checkpoint，则回退到最优模型
FALLBACK_CHECKPOINT = "checkpoints/best.pt"


class AgentFactory:
    """
    智能体工厂

    根据算法（dqn / ppo）和难度（easy/medium/hard/expert）
    创建并返回已配置好的智能体对象。

    可选加载预训练权重：
        - 优先尝试 `checkpoints/<difficulty>.pt`
        - 其次尝试 `checkpoints/best.pt`
        - 都不存在时返回随机初始化的智能体（适合训练）

    使用::

        agent = AgentFactory.create(algorithm="dqn", difficulty="hard")
        action = agent.select_action(state, deterministic=True)
    """

    @staticmethod
    def create(
        algorithm:   str = "dqn",
        difficulty:  str = "medium",
        load_model:  bool = True,
        model_path:  Optional[str] = None,
        cfg:         Optional[Config] = None,
    ):
        """
        创建智能体

        Args:
            algorithm:   "dqn" 或 "ppo"
            difficulty:  "easy" / "medium" / "hard" / "expert"
            load_model:  是否尝试加载预训练权重
            model_path:  显式指定权重路径（优先级最高）
            cfg:         基础 Config；None 时自动创建

        Returns:
            DQNAgent 或 PPOAgent 实例
        """
        algorithm  = algorithm.lower()
        difficulty = difficulty.lower()

        # ── 构建配置 ─────────────────────────────────────────────
        cm = ConfigManager(cfg)
        cm.apply_difficulty(difficulty)
        config = cm.get_config()

        # ── 创建智能体 ────────────────────────────────────────────
        if algorithm == "dqn":
            from agent.dqn_agent import DQNAgent
            agent = DQNAgent(config)
        elif algorithm == "ppo":
            from agent.ppo_agent import PPOAgent
            agent = PPOAgent(config)
        else:
            raise ValueError(
                f"未知算法 '{algorithm}'，可选: dqn / ppo"
            )

        # ── 加载权重 ──────────────────────────────────────────────
        if load_model:
            path = model_path or AgentFactory._find_checkpoint(difficulty)
            if path and os.path.isfile(path):
                agent.load(path)
                print(f"[AgentFactory] 已加载权重: {path}")
            else:
                print(f"[AgentFactory] 未找到预训练权重，使用随机初始化")

        print(f"[AgentFactory] 创建完成: algorithm={algorithm}  "
              f"difficulty={difficulty}")
        return agent

    @staticmethod
    def _find_checkpoint(difficulty: str) -> Optional[str]:
        """按优先级查找可用的 checkpoint 路径"""
        candidates = [
            DIFFICULTY_CHECKPOINT_MAP.get(difficulty, ""),
            FALLBACK_CHECKPOINT,
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                return p
        return None

    @staticmethod
    def list_options():
        """打印所有可用选项"""
        print("算法: dqn, ppo")
        print("难度: easy, medium, hard, expert")
        ConfigManager.list_difficulties()
