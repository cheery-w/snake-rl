# 基于强化学习的贪吃蛇 AI

使用 **Double DQN + Dueling DQN** 训练贪吃蛇智能体，并支持人机对战。

---

## 项目结构

```
snake_rl/
├── config.py               # 全局超参数配置
├── train.py                # 训练入口
├── evaluate.py             # 评估入口
├── play.py                 # AI 演示 / 人类手动游玩
├── versus.py               # 人机对战入口
├── requirements.txt        # 依赖库
│
├── env/                    # 游戏环境
│   ├── snake_env.py        # 核心游戏逻辑（纯 Python，无渲染）
│   ├── human_env.py        # 人类玩家环境（键盘输入集成）
│   ├── versus_env.py       # 人机对战环境（双蛇共享棋盘）
│   └── render.py           # pygame 渲染器
│
├── agent/                  # 智能体
│   ├── dqn_agent.py        # DQN 智能体（Double DQN）
│   └── memory.py           # 经验回放缓冲区
│
├── model/                  # 神经网络
│   └── dqn_model.py        # DQN / Dueling DQN 网络定义
│
├── trainer/                # 训练与评估
│   ├── trainer.py          # 训练器（主循环 + 检查点）
│   └── evaluator.py        # 独立评估器
│
├── human/                  # 人类玩家控制器
│   └── human_player.py     # 键盘输入 → 动作映射
│
├── utils/                  # 工具
│   ├── utils.py            # 绘图、日志、种子工具
│   └── logger.py           # CSV 训练日志记录器
│
├── checkpoints/            # 保存的模型权重
└── logs/                   # 训练日志与曲线图
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
cd snake_rl
python train.py                          # 使用默认配置训练
python train.py --episodes 5000          # 指定训练轮数
python train.py --resume checkpoints/best.pt  # 从检查点恢复训练
python train.py --no-dueling --no-double      # 使用标准 DQN
```

训练过程中每 50 轮打印一次进度，每 100 轮保存一次检查点，每 50 轮评估一次最优模型。

### 3. 观看 AI 演示

```bash
python play.py --model checkpoints/best.pt        # AI 自动游玩
python play.py --model checkpoints/best.pt --fps 5 # 慢速演示
python play.py --model checkpoints/best.pt --rounds 10  # 演示 10 局
```

### 4. 人类手动游玩

```bash
python play.py --human          # 默认帧率
python play.py --human --fps 15 # 自定义帧率
```

操作键：`WASD` 或方向键移动，`Space` 暂停，`R` 重置，`Q` / `Esc` 退出。

### 5. 评估模型性能

```bash
python evaluate.py --model checkpoints/best.pt
python evaluate.py --model checkpoints/best.pt --episodes 200 --render
```

### 6. 人机对战

```bash
# 同时对战（双蛇共享棋盘）
python versus.py --model checkpoints/best.pt
python versus.py --model checkpoints/best.pt --mode simultaneous --fps 8

# 轮流对战（各自游玩，比较得分）
python versus.py --model checkpoints/best.pt --mode turns
python versus.py --model checkpoints/best.pt --mode turns --rounds 5
```

对战操作：`WASD` 或方向键，`Space` 暂停，`Q` 退出。

---

## 算法说明

### 状态空间（11 维二值向量）

| 维度  | 含义                          |
|-------|-------------------------------|
| 0     | 直走方向有危险                |
| 1     | 右转方向有危险                |
| 2     | 左转方向有危险                |
| 3–6   | 当前方向独热编码（左/右/上/下）|
| 7–10  | 食物相对位置（左/右/上/下）   |

### 动作空间（3 个相对动作）

| 动作 | 含义   |
|------|--------|
| 0    | 直走   |
| 1    | 右转   |
| 2    | 左转   |

### 奖励设计

| 事件     | 奖励    |
|----------|---------|
| 吃到食物 | +10     |
| 死亡     | -10     |
| 接近食物 | +0.1    |
| 远离食物 | -0.1    |

### 网络架构（Dueling DQN）

```
输入层 (11) → 全连接 (256) → 全连接 (256)
                    ↙               ↘
        值流 V(s)          优势流 A(s,a)
                    ↘               ↙
              Q(s,a) = V(s) + A(s,a) - mean(A)
```

### 训练技巧

- **Double DQN**：用在线网络选动作，用目标网络估值，减少 Q 值高估
- **经验回放**：100,000 容量的环形缓冲区，打破样本相关性
- **目标网络**：每 1,000 步硬更新一次，稳定训练目标
- **ε-贪心衰减**：从 1.0 以 0.995 倍率衰减至 0.01
- **Huber 损失**：对异常值比 MSE 更鲁棒
- **梯度裁剪**：最大范数 10，防止梯度爆炸

---

## 对战模式说明

### 同时对战（simultaneous）

- 绿色（A）= AI，蓝色（Y）= 你
- 双方在同一 20×20 棋盘上同时游戏，争夺同一食物
- 危险包括：墙壁、自身、对方蛇身
- 两蛇均死亡后结算，按得分（或死亡顺序）判定胜负

### 轮流对战（turns）

- 你先玩，AI 后玩，各自独立游戏
- 每轮比较两方得分，多轮后统计胜场

---

## 配置说明

修改 `config.py` 调整超参数：

```python
class Config:
    GRID_COLS   = 20       # 网格列数
    GRID_ROWS   = 20       # 网格行数
    NUM_EPISODES = 2000    # 训练轮数
    LR           = 1e-3    # 学习率
    GAMMA        = 0.99    # 折扣因子
    BATCH_SIZE   = 64      # 批大小
    USE_DUELING  = True    # 使用 Dueling DQN
    USE_DOUBLE   = True    # 使用 Double DQN
    ...
```

---

## 依赖

```
torch>=2.0.0
numpy>=1.24.0
pygame>=2.1.0
matplotlib>=3.7.0
```
