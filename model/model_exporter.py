"""
模型导出工具
支持将训练好的 PyTorch 模型导出为:
    - TorchScript (.pt)
    - ONNX (.onnx)
"""

import os
import torch
import torch.nn as nn
from typing import Optional


class ModelExporter:
    """
    模型导出工具

    将 PyTorch nn.Module 导出为 TorchScript 或 ONNX 格式，
    便于在不依赖 Python / PyTorch 的环境中部署。

    使用::

        from agent.dqn_agent import DQNAgent
        from config import Config
        from model.model_exporter import ModelExporter

        agent = DQNAgent(Config())
        agent.load("checkpoints/best.pt")

        exporter = ModelExporter(agent.online_net, state_size=11)
        exporter.to_torchscript("exports/snake_dqn.pt")
        exporter.to_onnx("exports/snake_dqn.onnx")
    """

    def __init__(
        self,
        model:      nn.Module,
        state_size: int,
        device:     str = "cpu",
    ):
        """
        Args:
            model:      已训练的 nn.Module（DQNModel / PPOModel 等）
            state_size: 输入状态维度
            device:     导出时使用的设备（建议 "cpu" 以确保兼容性）
        """
        self.model      = model.eval().to(device)
        self.state_size = state_size
        self.device     = device

        # 示例输入（用于 trace / ONNX export）
        self._dummy = torch.zeros(1, state_size, device=device)

    # ------------------------------------------------------------------
    # TorchScript 导出
    # ------------------------------------------------------------------

    def to_torchscript(self, path: str, method: str = "trace") -> str:
        """
        导出为 TorchScript 格式

        Args:
            path:   输出文件路径（.pt）
            method: "trace"（推荐）或 "script"

        Returns:
            实际写入路径
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with torch.no_grad():
            if method == "trace":
                scripted = torch.jit.trace(self.model, self._dummy)
            else:
                scripted = torch.jit.script(self.model)

        scripted.save(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"[Exporter] TorchScript 已保存: {path}  ({size_kb:.1f} KB)")
        return path

    # ------------------------------------------------------------------
    # ONNX 导出
    # ------------------------------------------------------------------

    def to_onnx(
        self,
        path:         str,
        opset_version: int = 11,
        dynamic_batch: bool = True,
    ) -> str:
        """
        导出为 ONNX 格式

        Args:
            path:           输出文件路径（.onnx）
            opset_version:  ONNX 算子集版本
            dynamic_batch:  是否支持动态 batch size

        Returns:
            实际写入路径
        """
        try:
            import onnx  # noqa: F401
        except ImportError:
            print("[Exporter] 警告: 未安装 onnx，跳过 ONNX 导出")
            print("           安装命令: pip install onnx")
            return ""

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} \
            if dynamic_batch else None

        with torch.no_grad():
            torch.onnx.export(
                self.model,
                self._dummy,
                path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )

        size_kb = os.path.getsize(path) / 1024
        print(f"[Exporter] ONNX 已保存: {path}  ({size_kb:.1f} KB)")

        # 验证导出结果
        self._validate_onnx(path)
        return path

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _validate_onnx(self, path: str):
        """验证 ONNX 模型完整性"""
        try:
            import onnx
            model = onnx.load(path)
            onnx.checker.check_model(model)
            print(f"[Exporter] ONNX 验证通过: {path}")
        except Exception as e:
            print(f"[Exporter] ONNX 验证失败: {e}")

    def benchmark(self, n_runs: int = 1000) -> float:
        """
        简单推理速度基准测试

        Args:
            n_runs: 推理次数

        Returns:
            平均推理时间（毫秒）
        """
        import time
        self.model.eval()
        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = self.model(self._dummy)
            # 计时
            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = self.model(self._dummy)
            elapsed = (time.perf_counter() - t0) * 1000 / n_runs
        print(f"[Exporter] 推理速度: {elapsed:.4f} ms / step  "
              f"({1000/elapsed:.0f} steps/s)")
        return elapsed


# ──────────────────────────────────────────────────────────────────────
# 便捷函数
# ──────────────────────────────────────────────────────────────────────

def export_dqn_agent(
    agent,
    export_dir:    str = "exports",
    formats:       Optional[list] = None,
):
    """
    一键导出 DQNAgent 在线网络

    Args:
        agent:      DQNAgent 实例
        export_dir: 导出目录
        formats:    ["torchscript", "onnx"]（None 时全部导出）
    """
    if formats is None:
        formats = ["torchscript", "onnx"]

    exporter = ModelExporter(
        agent.online_net,
        state_size=agent.cfg.STATE_SIZE,
    )

    results = {}
    if "torchscript" in formats:
        p = os.path.join(export_dir, "snake_dqn.pt")
        results["torchscript"] = exporter.to_torchscript(p)
    if "onnx" in formats:
        p = os.path.join(export_dir, "snake_dqn.onnx")
        results["onnx"] = exporter.to_onnx(p)

    return results


def export_ppo_agent(
    agent,
    export_dir: str = "exports",
    formats:    Optional[list] = None,
):
    """
    一键导出 PPOAgent 的 Actor（策略网络）骨干

    只导出 Actor 相关权重，推理时只需前向传播至 actor 层
    """
    if formats is None:
        formats = ["torchscript", "onnx"]

    # 包装：只导出 backbone + actor 部分（argmax 动作）
    class ActorOnly(nn.Module):
        def __init__(self, ppo_model):
            super().__init__()
            self.backbone = ppo_model.backbone
            self.actor    = ppo_model.actor

        def forward(self, x):
            return self.actor(self.backbone(x)).argmax(dim=-1)

    actor_net = ActorOnly(agent.model)
    exporter  = ModelExporter(actor_net, state_size=agent.cfg.STATE_SIZE)

    results = {}
    if "torchscript" in formats:
        p = os.path.join(export_dir, "snake_ppo_actor.pt")
        results["torchscript"] = exporter.to_torchscript(p)
    if "onnx" in formats:
        p = os.path.join(export_dir, "snake_ppo_actor.onnx")
        results["onnx"] = exporter.to_onnx(p)

    return results
