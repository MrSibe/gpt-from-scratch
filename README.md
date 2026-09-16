# mrsibe-llm

从可读的字符级 GPT 出发，逐步学习 Transformer 架构、训练优化和推理优化。
当前实现是手写因果注意力、Pre-LayerNorm、绝对位置编码和 GELU FFN 的 decoder-only 模型。

- [学习与优化路线](docs/ROADMAP.md)：阶段目标、实验顺序、指标和验收标准。
- `model.py`：模型和自回归生成。
- `data.py`：字符编码、数据划分和 batch 采样。
- `train.py` / `generate.py`：训练和生成入口。
- `runlog.py`：实验目录、配置快照和指标 CSV。
- `plot.py`：对比多个实验的曲线。
- `tests/test_smoke.py`：正确性、实验记录与训练→保存→生成闭环测试。

## 环境

Python 3.12+，使用 uv 安装依赖：

```bash
uv sync
```

当前项目在 Linux/Windows 上配置了 PyTorch CUDA 13.0 包源；使用 GPU 需要匹配的驱动。
脚本默认在 CUDA 可用时使用 GPU，也可以通过 `--device cpu` 强制使用 CPU。

## 快速验证闭环

下面是小模型短跑，只验证流程，不代表训练质量。默认数据文件不存在时会下载 Tiny Shakespeare，需要联网。
若 `out/smoke` 已存在，目录名会自动变成 `out/smoke-2` 等，请以打印出的路径为准。

```bash
uv run python train.py \
  --device cpu --block-size 32 --batch-size 4 \
  --n-layer 1 --n-head 2 --n-embd 32 \
  --max-iters 10 --eval-interval 10 --eval-iters 2 \
  --out-dir out/smoke

uv run python generate.py \
  --device cpu --ckpt out/smoke/best.pt \
  --prompt "To be" --tokens 100 --seed 1337
```

使用 `--data /path/to/corpus.txt` 可以指定本地 UTF-8 文本。
当前按前 90% / 后 10% 划分训练集和验证集，每部分都需要至少 `block_size + 2` 个字符。
生成 prompt 必须非空，且所有字符都在训练词表内；temperature 应大于零。

完整默认配置的训练：

```bash
uv run python train.py
```

不指定 `--out-dir` 时，实验目录是 `runs/<本地时间>-gpt<n_layer>x<n_embd>/`，例如
`runs/20260916-170355-gpt6x384/`；该路径会直接打印出来。
默认模型比短跑配置大得多；显存不足时，先减小 `--batch-size`，必要时减小上下文和模型规模。
更多参数见 `uv run python train.py --help` 和 `uv run python generate.py --help`。

## 实验记录

每次训练都会写入一个独立目录（默认在 `runs/` 下，已被 git 忽略）：

```text
runs/20260916-170355-gpt6x384/
  config.json    模型/训练配置、数据摘要、代码版本和运行环境
  metrics.csv    每步一行的指标，可直接用 pandas 或表格软件读取
  best.pt        验证 loss 最低的权重，供 generate.py 使用
  summary.json   训练结束时的汇总
```

- 目录已存在时自动改用 `-2`、`-3` 后缀，不会覆盖旧实验；实际路径在启动时打印。
- `metrics.csv` 的列定义见 `runlog.py` 中的 `METRIC_FIELDS`，每行对应一次参数更新：

  | 列 | 含义 |
  | --- | --- |
  | `step` / `tokens_seen` | 已完成的参数更新次数 / 累计训练 token（不含评估） |
  | `train_loss_step` | 当前步的训练 loss |
  | `train_loss_eval` / `val_loss` | 评估时的平均训练/验证 loss，只在评估步有值 |
  | `lr` / `grad_norm` | 学习率 / 裁剪前梯度范数（尚未启用裁剪） |
  | `step_time_s` / `train_tokens_per_sec` | 单步时间 / 该步训练吞吐 |
  | `peak_memory_mb` | 该步（含该步内的评估）CUDA 峰值 allocated 显存，CPU 运行时为空 |
  | `eval_time_s` / `wall_time_s` | 该次评估耗时 / 从训练开始累计的时间 |

- 评估使用独立随机数流，并在每个评估点复用同一组固定 batch：

  - 改 `--eval-interval` 不会改变训练数据顺序，两次实验的逐步训练指标可以逐位对比；
  - 同 `--seed` 的实验使用完全相同的评估样本。
- `wall_time_s` 是从训练开始累计的时间，包含评估和保存；续训时在已有数值上继续累加，
  因此在同一个实验目录内始终单调不减。只有 `step_time_s` / `train_tokens_per_sec` 是纯训练指标。
- `last.pt` 在每个评估点写入，体积约为 `best.pt` 的两倍以上（含 AdamW 的一阶/二阶矩）。
  它不进 `step_time_s`，但会计入 `wall_time_s`；评估间隔很小时写盘开销会明显。

### 续训

```bash
uv run python train.py --resume runs/20260916-170355-gpt6x384/last.pt --max-iters 10000
```

- `--max-iters` 是包含已完成步数在内的**总步数目标**，必须大于 checkpoint 里的 `iter`。
- 恢复内容：模型权重、AdamW 优化器状态、全局与训练生成器的随机数状态、累计训练时间、best loss。
  指标追加到同一个 `metrics.csv`，不新建目录，也不重复表头。
- 以 checkpoint 为准（命令行给不同值时只提示、不生效）：模型结构、`block_size`、
  `batch_size`、`seed`。可以自由调整的是 `--max-iters`、`--eval-interval`、`--eval-iters`。
- 数据文件的 SHA-256 或词表不一致时会拒绝续训，避免静默换数据。
- 保存的随机数状态对应“最后一个训练步结束之后”：评估使用独立随机数流，不会推进训练随机数流。
  同一次实验内续训可以复现连续训练的逐步指标（CPU 测试校验到 1e-9；GPU 上不同进程之间
  本身存在 1e-6 级非确定性，不能要求逐位相同）。

### 绘图

```bash
uv run python plot.py                        # 对比 runs/ 下全部实验
uv run python plot.py runs/a runs/b          # 只对比指定实验
uv run python plot.py --smooth 50 --out runs/latest.png
```

输出四张子图：验证 loss 对训练 token、验证 loss 对累计时间、逐步训练吞吐、逐步峰值显存。
坐标轴标签使用英文，避免缺少中文字体的系统显示方块。
图例取自目录名，并带上 `config.json` 中的参数量。

## Checkpoint 约定

- `best.pt` 保存验证 loss 最好的模型；`last.pt` 保存可续训的完整状态。
- `config` 保存为普通字典；两种文件都用 `torch.load(..., weights_only=True)` 加载。
- `iter` 表示已完成的参数更新次数；评估发生在更新后，最后一步也会评估。
- `best.pt` 不含 optimizer / RNG，只用于推理；续训请用 `last.pt`。
- 修复前将 `GPTConfig` 对象直接写入文件的旧 checkpoint 不兼容当前加载方式。建议重新训练；如需保留旧权重，应仅对自己信任的旧文件做离线格式迁移，不要通过关闭安全加载来打开来源不明的文件。

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

测试使用 CPU 和临时本地语料，不下载数据、不依赖 GPU，也不会覆盖 `runs/` 中的记录。
覆盖前向/反向、因果性、小 batch 过拟合、保存加载一致性、导入无副作用、实验记录内容、
评估随机数隔离（改变评估频率不改变训练指标）、续训一致性（续训与连续训练的逐步指标一致、
拒绝换数据、累计时间单调）和 CLI 生成闭环，以及绘图脚本的冒烟测试。
