# mrsibe-llm

用于学习大模型训练、推理与优化的可读 GPT 实验项目。
主线使用 **TinyStories V2 GPT-4 + byte-level BPE**，保留字符级路径用于快速排错。
模型是 Pre-LayerNorm + 绝对位置编码 + GELU FFN 的 decoder-only GPT，支持手写因果注意力与 SDPA。
不引入 Trainer 框架，训练核心在 `train.py` 的 `get_lr()` 和 `train_step()`。
模型默认参数统一定义在 `model.py` 的 `GPTConfig`，训练超参数默认值在 `train.py`，各脚本通过 CLI 覆盖。

| 文件 | 作用 |
| --- | --- |
| `model.py` | 模型与自回归生成 |
| `prepare.py` | 仅在训练故事上拟合 BPE，离线写入 token 文件 |
| `tokenizer.py` | 字符 / BPE 编解码与 checkpoint 序列化 |
| `data.py` | 数据源配置、字符数据 / BPE memmap、随机窗口采样 |
| `train.py` | 梯度累积、调度、裁剪、验证、保存最佳模型 |
| `generate.py` | 加载 `best.pt` 生成文本 |
| `benchmark.py` | 独立的训练计算测速 |
| `profiler.py` | 训练算子耗时、显存事件、结构归因与 trace |
| `log.py` / `plot.py` | 运行记录 / 多次训练曲线对比 |
| `tests/` | 数据、训练数值、日志、生成和测速的回归测试 |

## 环境

Python 3.12+，使用 uv：

```bash
uv sync
```

命令示例使用 Bash；Windows 可在 WSL 中执行，或改写为对应的 PowerShell 语法。
Linux/Windows 配置了 PyTorch CUDA 13.0 包源，GPU 需要匹配的驱动。
`train.py`、`benchmark.py`、`profiler.py` 默认在 CUDA 可用时使用 GPU，以 BF16 autocast 计算，权重保持 FP32；
CPU 请指定 `--device cpu --dtype fp32`。不支持 BF16 的 CUDA 设备可用 `--dtype fp16`（自动启用 GradScaler）。
`generate.py` 使用 FP32，不提供 `--dtype`；`--device cpu` 可用于 CPU 生成。

## 准备 TinyStories V2 GPT-4

```bash
# 全量数据池；只用前 2 万条训练故事拟合一次 BPE
uv run python prepare.py
```

默认配置：

- 官方仓库：[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)。
- 固定 revision：`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`，配置位于 `data.py`。
- 文件：`TinyStoriesV2-GPT4-train.txt` / `TinyStoriesV2-GPT4-valid.txt`，不是同仓库里的原版文件。
- 输出：`data/tinystories-v2/`；词表目标 8192（包含 EOS）。
- 全量 train 文本约 2.23 GB、valid 约 22.5 MB，需要联网并预留磁盘空间。

**完整数据池不等于每次训练都跑完整一遍数据**。训练按固定更新次数随机采样，
memmap 不把全量 token 装进 GPU；总 token 数以准备完成后的 `meta.json` 为准。

准备一个小子集排错时，另用目录，避免误当作全量数据：

```bash
uv run python prepare.py \
  --out-dir data/tinystories-v2-smoke \
  --max-train-stories 2000 --max-val-stories 200
```

直接读取 URL 失败或网络不稳定时，可先下载原始文件，再从本地准备。
下面会下载全量原始文件；如目标文件已存在，请先核对内容，避免覆盖已有数据。
自定义 train/val **必须成对指定**，防止混用版本：

```bash
mkdir -p data/raw
BASE=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
curl -fL --retry 3 "$BASE/TinyStoriesV2-GPT4-train.txt" -o data/raw/TinyStoriesV2-GPT4-train.txt
curl -fL --retry 3 "$BASE/TinyStoriesV2-GPT4-valid.txt" -o data/raw/TinyStoriesV2-GPT4-valid.txt
uv run python prepare.py \
  --train-file data/raw/TinyStoriesV2-GPT4-train.txt \
  --val-file data/raw/TinyStoriesV2-GPT4-valid.txt
```

### 数据处理约定

- 输入接受 UTF-8 本地文件或 HTTP(S) URL，故事用 `<|endoftext|>` 分隔。
- 只用 train 拟合 tokenizer，最多 `--tokenizer-stories 20000` 条，且不超出所选训练子集。
- `--max-train-stories` / `--max-val-stories` 默认 0，表示全部；非零时取前 N 条完整故事，**不是随机抽样**。
- `--vocab-size` 范围 257–65536；小语料实际词表可能更小，训练自动读取实际大小。
- `--tokenizer path/to/tokenizer.json` 可复用本项目已准备的 tokenizer；此时不重新拟合，`--vocab-size` 不改变已有词表。拒绝启用了 truncation/padding 的 tokenizer，避免静默丢失文本或引入填充。
- 去除故事首尾空白，编码后追加真正的 EOS ID；按故事顺序拼接，无 padding，允许因果 attention 跨故事边界。
- 输出 `tokenizer.json`、`train.bin`、`val.bin`、`meta.json`。token 文件为小端 uint16，取 batch 后才转成 int64。
- manifest 记录来源、故事数/token 数、文件与文本 SHA-256；加载时核验准备后的工件是否改变，不自动检查上游全文摘要或做 train/val 去重。
- 文件末尾未带 EOS 的最后一段也会作为故事编码；自行截取原始文件时，需避免截断故事。
- 不覆盖已有输出目录。失败时不会写完整 manifest，请排查原因后使用新目录重试。
- URL 每次遍历重新打开，训练源通常读两遍；网络不稳定时建议先下载到本地。脚本不支持断点下载或自动重试。
- 两个版本有重叠，不应直接拼接；V2 应使用配套的 V2 valid。切换主线时重新在 V2 train 拟合 tokenizer，不自动复用旧版工件。

数据遵循上游标注的 `CDLA-Sharing-1.0`，详见[官方数据卡](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md)。

## 训练与生成

准备完整数据后，默认命令即可训练：

```bash
uv run python train.py --run-name tinystories-v2

# 目录名以训练打印的为准
uv run python generate.py \
  --ckpt "runs/<时间戳>-tinystories-v2/best.pt" \
  --prompt "Once upon a time" --tokens 200 --seed 1337
```

默认模型 8 层 / 8 头 / 512 维、context=256、dropout=0.1，词表为 8192 时约 **33.75M 参数**，
输入输出 embedding 默认不共享（`--tie-embeddings` 共享后约 **29.55M**）。
默认 BF16、micro-batch=16、累积 8 次，即 **32768 tokens/update**；
20000 次更新尝试约处理 655.36M tokens（约 19.4 tokens/参数），已超过全量 train 的约 536.6M tokens，
即默认配置会跨 epoch 重复采样。这是 RTX 4060 Laptop 8GB 的起步配置，不是最优长训结论。

`16 / 8` 这组比例是实测选出来的，不要在“显存还剩很多”的直觉下随手调大：
在 RTX 4060 Laptop（功耗受限）上固定 32768 tokens/update，轮转跑 60 次更新各 3 遍，
`16/8` 为 26.70s，`32/4` 为 27.17s（+1.8%），`64/2` 为 28.03s（+5.0%），峰值显存依次是
1659 / 2730 / 4874 MB；`128/1` 峰值达 9031 MB，**超过 8188 MiB 设备显存**，WSL 的 sysmem
回退让它慢了约 5 倍。所以这里的显存余量是给更长的 `--block-size` 和评估留的安全边际，
不是可兑换成吞吐的空间。换机器后应用同样的轮转方式重测（顺序跑会被功耗/温度漂移骗到，
同配置热态比冷态慢约 9%）。

训练只读取已准备的数据，不自动下载全量数据；缺失时会提示先运行 `prepare.py`。
其他数据可通过训练的 `--data` 显式选择；checkpoint 仅用于生成，不支持恢复训练。

### 小规模验证

使用上面另存的小子集，先验证 BPE 训练路径：

```bash
uv run python train.py \
  --data data/tinystories-v2-smoke --max-iters 120 \
  --eval-interval 60 --eval-iters 20 --run-name v2-smoke
```

无需准备 BPE 的 CPU 字符级快速测试：

```bash
uv run python train.py \
  --tokenizer char --device cpu --dtype fp32 \
  --block-size 32 --batch-size 4 --grad-accum-steps 1 \
  --n-layer 1 --n-head 2 --n-embd 32 \
  --max-iters 10 --eval-interval 10 --eval-iters 2 --eval-batch-size 2 \
  --run-name char-smoke
```

`--tokenizer char` 默认读取 `input.txt`，缺失时下载 Tiny Shakespeare；自定义文件缺失则报错。
字符数据按前 90% / 后 10% 拆分，BPE 使用准备好的独立 train/val；两部分都至少需要 `block_size + 1` 个 token。
切换 tokenizer 不会自动改变其他模型/训练参数。

生成直接读取 checkpoint 中保存的 tokenizer；BPE 遇到 EOS 停止。
prompt 必须非空，temperature 为有限正数；字符模型不接受词表外字符，byte-level BPE 支持未见 UTF-8 文本。
`--top-k 0` 关闭 top-k 过滤。

## 消融参数

| 参数 | 默认值 | 常用消融 |
| --- | --- | --- |
| `--tokenizer` | `bpe` | `bpe` / `char` |
| `--data` | BPE: `data/tinystories-v2`；char: `input.txt` | 准备好的数据目录 / 字符文本 |
| `prepare.py --vocab-size` | 8192 | 4096 / 8192 / 16384；需重新准备数据，不是 train 参数 |
| `--n-layer / --n-head / --n-embd` | 8 / 8 / 512 | 维度必须能被头数整除 |
| `--block-size` | 256 | 256 / 512 |
| `--dropout` | 0.1 | 0 / 0.1 / 0.2 |
| `--batch-size` | 16 | 每个 micro-step 的序列数：8 / 16 / 32 / 64；与 `--grad-accum-steps` 联动，改 `16→32` 时把 `8→4` 才能保持 tokens/update 与 LR 不变 |
| `--grad-accum-steps` | 8 | 与 batch 联动，保持 tokens/update 一致 |
| `--lr` | 1e-3 | 1e-4 / 3e-4 / 1e-3 / 2e-3 |
| `--betas` | 0.9 0.95 | AdamW 的两个 beta |
| `--weight-decay` | 0.1 | 0 / 0.01 / 0.1；按 `dim>=2` 分组，只衰减矩阵权重，LayerNorm 与所有 bias 恒为 0 |
| `--lr-schedule` | `cosine` | `constant` / `cosine` |
| `--warmup-ratio` / `--warmup-iters` | 0.02 / 未指定 | 二选一；默认 ratio × max-iters 向下取整 |
| `--min-lr` | 3e-5 | cosine 终点；constant 时忽略 |
| `--grad-clip` | 1.0 | 0 关闭 / 1.0 开启 |
| `--max-iters` | 20000 | 更新尝试次数，不是 micro-step 数；默认值由 10000 步外推（那次 run 结束时 val 仍在下降），尚无跑完 20000 步的记录 |
| `--eval-interval` | 250 | 每多少次更新尝试做一次验证 |
| `--eval-batch-size / --eval-iters` | 64 / 50 | 消融时固定验证 token 预算；默认约 82 万 token/次 |
| `--seed` | 1337 | 重复实验时更换随机种子 |
| `--attention` | `sdpa` | `manual` / `sdpa` |
| `--tie-embeddings` / `--no-tie-embeddings` | `--no-tie-embeddings` | 输入 embedding 与 `lm_head` 共享权重；GPT-2 的做法，省 `vocab_size × n_embd` 个参数（默认配置下 4.19M） |
| `--dtype` | `bf16` | `fp32` / `fp16` / `bf16` |
| `--compile` | 关闭 | 编译模型 forward/backward，不含 Python 循环和优化器 |

关闭裁剪：`--grad-clip 0`；关闭调度和 warmup：`--lr-schedule constant --warmup-iters 0`；
关闭累积：`--grad-accum-steps 1`。若想隔离“累积实现”的影响，关闭累积时同步放大 micro-batch，保持有效 batch 不变。

权重共享：`--tie-embeddings` 让 `wte` 与 `lm_head` 指向同一张量。`named_parameters()` 会按张量身份去重，
所以优化器不会对同一权重重复更新；`state_dict()` 仍同时保留 `wte.weight` 与 `lm_head.weight` 两个键，
`generate.py` 通过 checkpoint 里的 `tie_embeddings` 重建模型，旧的不带该字段的 checkpoint 仍按不共享加载。

权重衰减：AdamW 分成两组，`dim>=2` 的权重（embedding、各 Linear 与注意力的投影矩阵）用 `--weight-decay`，
`dim<2` 的参数（LayerNorm 的 weight/bias、所有 Linear bias，默认配置下 83 个张量、约 6.2 万参数）恒不衰减。
`benchmark.py` / `profiler.py` 复用 `train.build_optimizer`，三处不会漂移。

核心顺序：设 LR → zero_grad → 多个 micro-batch 的 `loss / accum_steps` 分别 backward →
FP16 unscale → 全局 norm 裁剪一次 → AdamW step 一次。无需保留多个 micro-batch 的计算图。

warmup 从 `peak_lr / warmup_iters` 升到 peak；之后 cosine 从 peak 降到 min。
若只有一个 decay update，则使用 peak。FP16 溢出跳步时不推进调度；`max-iters` 限制更新尝试次数，
因此发生跳步时成功更新较少，可能尚未走到 min LR。

一次只改变一个主要变量。只有同一数据分布、同一 tokenizer 的 loss/perplexity 才适合直接比较。
SDPA 后端由 PyTorch 自动选择，不保证使用 FlashAttention；不同精度/后端不保证逐位一致。

## 训练记录

```text
runs/<本地时间>-<run-name>/
  config.json    模型/训练配置、数据 manifest 与 SHA-256、代码版本、运行环境
  metrics.csv    每次更新尝试一行；评估点批量写入
  best.pt        验证 loss 最低的模型、配置、tokenizer
  summary.json   最佳 loss、更新计数、总时间、峰值显存
```

重名目录自动追加后缀，不覆盖旧记录；`run-name` 不能包含路径分隔符。

| CSV 列 | 含义 |
| --- | --- |
| `step` / `tokens_seen` | 更新尝试次数 / 已处理训练 token 数，含跳步时处理的 token，不含验证 |
| `optimizer_steps` / `skipped_update` | 累计成功更新次数 / 本次是否因 FP16 溢出跳过更新 |
| `train_loss_step` | 本次所有 micro-batch 的平均原始 loss，含 dropout |
| `lr` | 该次更新尝试实际使用的学习率，保留浮点精度，避免小 LR 被记为 0 |
| `val_loss` / `val_ppl` | 固定验证窗口的 token 平均 NLL / exp(NLL)，仅评估点记录 |
| `grad_norm` | unscale 后、clip 前的全局 L2 范数，仅评估点记录 |
| `wall_time_s` | 含验证、保存、编译冷启动的累计时间，仅评估点记录 |

- 普通 BF16/FP32 步不读取 CUDA loss 标量，评估点批量传回；中断会丢失最近未写入的一段指标。
- 验证使用独立固定 seed，关闭 dropout，最后一步也评估。等长 batch 的均值即 token 加权均值。
  随机窗口会重叠，不代表整份验证集的无重复遍历。
- 默认验证预算 64×50 = 819,200 token/次：实测单次 `val_loss` 的标准差约 **0.005**
  （旧的 16×50 = 204,800 token 约 **0.015**，单次评估 95% 区间达 ±0.028，比多数消融效应还大）。
  因此**单点对比不可信**，消融要么用这个默认预算，要么比较多个评估点的趋势；
  `best_val_loss` 是 80 个含噪评估点的最大值，本身偏乐观 0.01 量级。
- 显存峰值在训练循环前重置，含驻留权重及循环中的训练/验证，不含初始化瞬时峰值；CPU 为 null。
- checkpoint 仅使用一种格式：`model`、`config`、`tokenizer`、`iter`、`val_loss`。
  `best.pt` 不依赖原数据目录；加载使用 `weights_only=True`，compile 训练也保存标准权重名。
- **不支持续训**：未保存 optimizer/scaler/RNG 状态；也没有自动定期生成或 KV cache。
  当前生成每个 token 都重算截断上下文，以保持代码直观。

## 独立测速与算子分析

两者默认模型形状与 V2 训练一致：8192 词表、8 层 / 8 头 / 512 维、dropout=0.1、batch=16、context=256。
如果实际 tokenizer 词表小于 8192，请用 `--vocab-size` 匹配。

```bash
uv run python benchmark.py \
  --device cuda --dtype bf16 \
  --warmup 50 --steps 100 --repeats 5 --run-name sdpa-bf16

uv run python profiler.py \
  --device cuda --dtype bf16 \
  --warmup 10 --steps 5 --annotate-model --with-flops \
  --run-name profile-sdpa
```

- 使用固定设备端合成 batch，包含 forward/backward、AdamW 与 AMP scaler；不含采样、H2D、验证、日志和保存。
- **每一步是单 micro-batch + optimizer**，不包含训练器的梯度累积、clip 或 LR 调度。
  不能直接把这些单步耗时当作 `train.py` 的完整更新耗时。
- 上面这条还有一层定量后果：benchmark 每个 micro-batch 都做一次 `zero_grad` + AdamW，
  而 `train.py` 每个 update 只做一次，所以用它推断梯度累积的吞吐时会**系统性低估大 accum 的配置**。
  实测这个“多出来的”优化器开销 c ≈ 6.4–9.2 ms/次（33.75M 参数、FP32 权重），
  换算修正为 `update_time ≈ accum × (micro_step_time - c) + c`；
  减掉这一项后，benchmark 里 `batch 32` 比 `batch 16` 快 6.8% 的优势会反号——
  与端到端实测的 `32/4` 偏慢一致。
- benchmark 在测量窗口两端同步，报告吞吐中位数/范围及平均单步时间；输出 config、benchmark.csv、summary。
  显存峰值在 warmup 后重置，包含驻留的模型和优化器状态；warmup 时间含首次编译。
  若测量窗口存在 FP16 跳步，benchmark 会拒绝该次结果，提示增加 warmup 或更换精度；成功更新计数在计时区间外检查。
- profiler 输出 `operators.txt` 与 `trace.json`；使用 <https://ui.perfetto.dev> 打开 trace，
  查看 `train/forward`、`train/backward`、`train/optimizer` 及实际 GPU kernels。
  算子表本身就带 `CPU total` / `CUDA total` 列（可与 `Self` 列对照区分“自身耗时”与“含子节点耗时”）。
- `--annotate-model` 把每个 `Linear` / `LayerNorm` / `Embedding` 的 forward 包一层
  `record_function`，名字沿用 state_dict 路径，于是算子表里直接出现
  `blocks.0.attn.q_proj` 这类行，并追加一张把 `blocks.<N>.` 前缀合并后的结构汇总表。
  只包叶子模块：叶子自己的 kernel 时间就是该行的 self time，可以跨层相加；
  包容器模块会让父行变成累计值，既不能相加也会与子行重复。
  插桩只作用于 profiler 进程内的实例，`model.py` 保持干净，`train.py` / `benchmark.py` 不受影响
  （8 层模型实测每步开销 < 0.1%）。
  **注意它只覆盖 forward**：backward 的 kernel 由 autograd 引擎在标注区域外发起，
  所以结构汇总表的占比是“标注模块内部”的占比，不是整个 step 的占比；
  forward / backward / optimizer 的整体切分仍看 `train/*` 区域行。
- `--with-flops` 估算 `mm` / `bmm` / `addmm` 的 FLOPs，并在表格中自动多出一列 `Total GFLOPs`，
  同时把单步总量与每 token 的估算写进 `operators.txt`（默认配置实测约 722 GFLOP/step、
  176 MFLOP/token，与 `2 × 参数量 × token` 的解析式相差 7% 以内）。
  它只覆盖矩阵乘：SDPA 的注意力核心不是 matmul，因此 `--attention sdpa` 会低估，
  `--attention manual` 才会把注意力的 `bmm` 计入。
  该开关会自动打开 `record-shapes`（torch 的行为），但表格是否按形状分组仍只由 `--record-shapes` 决定。
- `--record-shapes`、`--profile-memory`、`--with-stack` 会增加采集开销；`--row-limit` 控制表格长度。
  内存事件只覆盖采集窗口，self memory 是净分配量，可能为负。
  `--with-stack` 还会在 trace 里生成 `nn.Module:` 层级，但这类事件不会进入 `key_averages()` 表格，
  所以“表格里的结构归因”要用 `--annotate-model`，“trace 里的嵌套视图”用 `--with-stack`。
- profiler 会扰动性能，不把它的耗时当作正式吞吐。稳定对照还需多次启动进程，并控制温度、功耗和后台负载。

## 绘图与测试

```bash
uv run python plot.py
uv run python plot.py runs/a runs/b --smooth 50 --out runs/comparison.png
uv run python -m unittest discover -s tests -v
```

绘图只读取含 `metrics.csv` 的训练目录，输出验证 loss 对训练 tokens / 累计时间的曲线，
同次训练的 train/val 曲线使用同一颜色。默认合并所有训练目录；不同数据或 tokenizer 的记录应手动筛选，避免错误比较。
单元测试不下载真实数据；CUDA 可用时额外测试 FP16 溢出跳步。

`data/`、`input.txt`、`runs/` 和虚拟环境均不提交到 Git；发布源码后需自行准备数据和训练权重。

## 许可

MIT，见 [`LICENSE`](LICENSE)。
