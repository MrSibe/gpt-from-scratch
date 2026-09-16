# mrsibe-llm

从可读的字符级 GPT 出发，逐步学习 Transformer、训练和推理优化。
模型是 Pre-LayerNorm + 绝对位置编码 + GELU FFN 的 decoder-only GPT，支持手写因果注意力与 SDPA。

| 文件 | 作用 |
| --- | --- |
| `model.py` | 模型与自回归生成 |
| `data.py` | 字符编码、数据划分、batch 采样 |
| `train.py` | 训练、验证指标、保存最佳模型 |
| `generate.py` | 加载 `best.pt` 生成文本 |
| `benchmark.py` | 独立的训练计算测速 |
| `log.py` | 运行目录、配置快照、指标 CSV |
| `plot.py` | 多实验训练曲线对比 |

## 环境

Python 3.12+，使用 uv：

```bash
uv sync
```

Linux/Windows 配置了 PyTorch CUDA 13.0 包源，GPU 需要匹配的驱动。
脚本默认在 CUDA 可用时使用 GPU，也可用 `--device cpu` 强制 CPU。

## 训练与生成

```bash
uv run python train.py \
  --device cpu --block-size 32 --batch-size 4 \
  --n-layer 1 --n-head 2 --n-embd 32 \
  --max-iters 10 --eval-interval 10 --eval-iters 2 \
  --run-name smoke

# 目录名以训练时打印的为准
uv run python generate.py \
  --device cpu --ckpt runs/<时间戳>-smoke/best.pt \
  --prompt "To be" --tokens 100 --seed 1337
```

- 默认数据 `input.txt`，不存在时联网下载 Tiny Shakespeare；`--data` 可指定本地 UTF-8 文本。
- 前 90% / 后 10% 为训练集 / 验证集，两部分各需至少 `block_size + 2` 个字符。
- 实验目录固定为 `runs/<本地时间>-<run-name>/`，重名自动追加后缀，不覆盖旧实验。
- 生成时 prompt 需非空且字符都在训练词表内，`temperature` 需大于 0。
- `n-embd` 必须整除 `n-head`；`run-name` 不能包含路径分隔符。
- 完整默认训练：`uv run python train.py`；显存不足先减小 `--batch-size`。

### 实验变量

`train.py` 与 `benchmark.py` 共有：

- `--attention manual|sdpa`：默认 manual。SDPA 使用因果模式，评估时关闭 dropout；
  后端由 PyTorch 自动选择，**不保证使用 FlashAttention**。
- `--dtype fp32|fp16|bf16`：默认 fp32。低精度仅支持 CUDA，用 autocast，权重保持 FP32；
  FP16 启用 GradScaler（梯度溢出会跳过该步更新），BF16 需设备支持。
- `--compile`：只编译模型的 forward/backward，不含 Python 主循环和优化器。

一次只改一个主要变量，先验正确性再测性能。不同精度/后端不保证逐位一致；
`step` 是迭代次数，`tokens_seen` 才是实际处理的训练 token 数。

## 训练记录

```text
runs/<时间戳>-<run-name>/
  config.json    模型/训练配置、数据 SHA-256、代码版本、运行环境
  metrics.csv    每步训练 loss；评估点批量写入
  best.pt        验证 loss 最低的权重、配置和词表
  summary.json   best loss、best step、总时间、全程峰值显存
```

| CSV 列 | 含义 |
| --- | --- |
| `step` / `tokens_seen` | 训练迭代 / 累计训练 token（不含评估） |
| `train_loss_step` | 当前 batch 的训练 loss，含 dropout |
| `lr` | 学习率 |
| `val_loss` | 固定验证样本的平均 loss，仅评估点记录 |
| `grad_norm` | 更新前的全局梯度 L2 范数，仅评估点计算；FP16 先 unscale，不裁剪 |
| `wall_time_s` | 含评估、保存和编译冷启动的累计时间，仅评估点记录 |

- 普通训练步不读取 CUDA 标量，loss 先留在设备上，到评估点才批量写盘；
  中断会丢失最近一段未写入的 loss（数据传输和 FP16 scaler 仍可能同步）。
- 只评估 val，用固定 seed 的独立生成器复用同一批验证样本，最后一步也评估。
- 峰值显存在训练循环前重置（不含模型初始化），覆盖训练、评估和编译全程，CPU 为 null。
- 只保存 `best.pt`，无优化器/RNG 状态，不支持续训。加载用 `weights_only=True`，
  编译训练也保存标准权重名。

## 独立测速

```bash
uv run python benchmark.py \
  --device cuda --attention sdpa --dtype bf16 \
  --batch-size 16 --block-size 256 \
  --warmup 50 --steps 100 --repeats 5 --run-name sdpa-bf16
```

- 使用固定的、已在设备上的合成 `(x, y)`，测的是训练计算，**不是端到端吞吐**。
- 包含 forward、backward、AdamW 和 AMP scaler；排除采样、CPU→GPU 传输、评估、日志和保存。
- 只在每个测量窗口两端显式同步，窗口内部不读取 loss、不计时。
- 默认连续测 5 个窗口、每窗口 100 步，报告吞吐中位数/最小值/最大值和平均单步时间；
  严谨对照还应重复启动进程并控制温度、功耗和后台负载。
- `warmup_time_s` 含首次编译，不是纯编译耗时；warmup 太短或发生重编译会影响结果。
- 显存峰值在 warmup 后重置，包含驻留的模型和优化器状态，排除 warmup 的瞬时峰值。
- 输出 `config.json`、`benchmark.csv`、`summary.json`，不生成权重。`--vocab-size` 默认 65，
  与真实训练比较时应匹配词表、模型、batch 和上下文长度。

## 绘图

```bash
uv run python plot.py
uv run python plot.py runs/a runs/b --smooth 50 --out runs/comparison.png
```

只读取含 `metrics.csv` 的训练目录（benchmark 目录自动跳过），
输出验证 loss 对训练 token（含淡色训练曲线）和验证 loss 对累计时间两张子图。
性能窗口数据见 `benchmark.csv`，深入排查瓶颈时再用 PyTorch Profiler。
