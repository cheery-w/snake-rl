"""
训练入口
用法:
    python train.py                          # 默认配置训练
    python train.py --episodes 3000          # 自定义轮数
    python train.py --resume checkpoints/best.pt  # 续训
    python train.py --no-dueling --no-double # 使用标准 DQN
"""

import argparse
import sys
import os

# 确保从项目根目录运行时能正确导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from trainer.trainer import Trainer
from utils.utils import set_seed, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Snake DQN 训练")

    # 训练参数
    parser.add_argument("--episodes",   type=int,   default=None,
                        help="训练轮数（覆盖 Config.NUM_EPISODES）")
    parser.add_argument("--lr",         type=float, default=None,
                        help="学习率")
    parser.add_argument("--batch-size", type=int,   default=None,
                        help="批大小")
    parser.add_argument("--gamma",      type=float, default=None,
                        help="折扣因子")

    # 算法开关
    parser.add_argument("--no-dueling", action="store_true",
                        help="禁用 Dueling DQN")
    parser.add_argument("--no-double",  action="store_true",
                        help="禁用 Double DQN")

    # 续训
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 路径续训")

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true",
                        help="训练结束后不绘制曲线")

    return parser.parse_args()


def main():
    args = parse_args()

    # 应用命令行覆盖
    cfg = Config()
    if args.episodes  is not None: cfg.NUM_EPISODES  = args.episodes
    if args.lr        is not None: cfg.LR            = args.lr
    if args.batch_size is not None: cfg.BATCH_SIZE   = args.batch_size
    if args.gamma     is not None: cfg.GAMMA         = args.gamma
    if args.no_dueling: cfg.USE_DUELING = False
    if args.no_double:  cfg.USE_DOUBLE  = False
    cfg.SEED = args.seed

    set_seed(cfg.SEED)

    # 构建训练器
    trainer = Trainer(cfg)

    # 续训
    if args.resume:
        if not os.path.isfile(args.resume):
            print(f"[Error] checkpoint 不存在: {args.resume}")
            sys.exit(1)
        trainer.agent.load(args.resume)

    # 开始训练
    metrics = trainer.train()

    # 绘制训练曲线
    if not args.no_plot:
        plot_training_curves(
            metrics["rewards"],
            metrics["scores"],
            metrics["losses"],
            metrics["eval_scores"],
            save_path=os.path.join(cfg.LOG_DIR, "training_curves.png"),
        )

    # 最终评估
    print("\n── 最终评估（贪心策略）──")
    trainer.evaluate(n_episodes=20)


if __name__ == "__main__":
    main()
