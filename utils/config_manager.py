"""
配置管理器
支持从 JSON 文件加载/保存配置，并提供运行时动态修改接口
"""

import json
import os
from typing import Any

from config import Config


class ConfigManager:
    """
    配置管理器

    封装 Config 对象，额外提供:
        - 从 JSON 文件加载配置
        - 将当前配置保存为 JSON
        - 按键名动态读写配置项
        - 内置难度预设

    使用::

        cm = ConfigManager()
        cfg = cm.get_config()          # 获取当前 Config 对象

        cm.apply_difficulty("easy")    # 应用难度预设
        cm.save("my_config.json")      # 保存为 JSON
        cm.load("my_config.json")      # 从 JSON 载入
    """

    # ── 难度预设 ────────────────────────────────────────────────────
    DIFFICULTY_PRESETS = {
        "easy": {
            "HIDDEN_SIZES":  [128, 128],
            "NUM_EPISODES":  500,
            "EPS_END":       0.05,
            "EPS_DECAY":     0.990,
            "LR":            1e-3,
            "_description":  "简单：网络较小，训练轮数少",
        },
        "medium": {
            "HIDDEN_SIZES":  [256, 256],
            "NUM_EPISODES":  2_000,
            "EPS_END":       0.01,
            "EPS_DECAY":     0.995,
            "LR":            1e-3,
            "_description":  "中等：标准 DQN 配置",
        },
        "hard": {
            "HIDDEN_SIZES":  [256, 256, 256],
            "NUM_EPISODES":  5_000,
            "EPS_END":       0.005,
            "EPS_DECAY":     0.997,
            "LR":            5e-4,
            "_description":  "困难：三层网络，更多训练",
        },
        "expert": {
            "HIDDEN_SIZES":  [512, 512, 256],
            "NUM_EPISODES":  10_000,
            "EPS_END":       0.001,
            "EPS_DECAY":     0.999,
            "LR":            3e-4,
            "USE_DUELING":   True,
            "USE_DOUBLE":    True,
            "_description":  "专家：大网络，超长训练，Double+Dueling",
        },
    }

    def __init__(self, config: Config = None):
        self._cfg = config or Config()

    # ── 核心接口 ────────────────────────────────────────────────────

    def get_config(self) -> Config:
        """返回当前 Config 对象"""
        return self._cfg

    def get(self, key: str, default: Any = None) -> Any:
        """按键名读取配置项"""
        return getattr(self._cfg, key, default)

    def set(self, key: str, value: Any):
        """按键名动态修改配置项"""
        if not hasattr(self._cfg, key):
            raise KeyError(f"Config has no attribute '{key}'")
        setattr(self._cfg, key, value)

    # ── 难度预设 ────────────────────────────────────────────────────

    def apply_difficulty(self, difficulty: str):
        """
        应用难度预设，覆盖对应的超参数

        Args:
            difficulty: "easy" | "medium" | "hard" | "expert"
        """
        difficulty = difficulty.lower()
        if difficulty not in self.DIFFICULTY_PRESETS:
            raise ValueError(
                f"未知难度 '{difficulty}'，可选: "
                f"{list(self.DIFFICULTY_PRESETS.keys())}"
            )
        preset = self.DIFFICULTY_PRESETS[difficulty]
        for k, v in preset.items():
            if k.startswith("_"):
                continue
            setattr(self._cfg, k, v)
        print(f"[ConfigManager] 已应用难度预设: {difficulty}  "
              f"— {preset.get('_description', '')}")

    @classmethod
    def list_difficulties(cls):
        """打印所有可用难度"""
        for name, p in cls.DIFFICULTY_PRESETS.items():
            print(f"  {name:8s}  {p.get('_description', '')}")

    # ── JSON 持久化 ─────────────────────────────────────────────────

    def save(self, path: str):
        """将当前配置保存为 JSON 文件"""
        data = {
            k: v for k, v in vars(self._cfg.__class__).items()
            if not k.startswith("_") and not callable(v)
        }
        # 实例属性优先级更高（可能被命令行覆盖）
        data.update({
            k: v for k, v in vars(self._cfg).items()
            if not k.startswith("_")
        })
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[ConfigManager] 配置已保存至 {path}")

    def load(self, path: str):
        """从 JSON 文件加载配置，覆盖对应项"""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(self._cfg, k):
                setattr(self._cfg, k, v)
        print(f"[ConfigManager] 已从 {path} 加载配置（{len(data)} 项）")

    # ── 工具 ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """返回当前配置的摘要字符串"""
        lines = ["[Config Summary]"]
        cls_attrs = {
            k: v for k, v in vars(self._cfg.__class__).items()
            if not k.startswith("_") and not callable(v)
        }
        inst_attrs = {
            k: v for k, v in vars(self._cfg).items()
            if not k.startswith("_")
        }
        merged = {**cls_attrs, **inst_attrs}
        for k, v in sorted(merged.items()):
            lines.append(f"  {k:20s} = {v}")
        return "\n".join(lines)
