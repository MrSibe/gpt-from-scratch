# mrsibe-llm

从可读的字符级 GPT 出发，逐步学习 Transformer 架构、训练优化和推理优化。
当前实现是手写因果注意力、Pre-LayerNorm、绝对位置编码和 GELU FFN 的 decoder-only 模型。

- [学习与优化路线](docs/ROADMAP.md)：阶段目标、实验顺序、指标和验收标准。
- `model.py`：模型和自回归生成。
- `data.py`：字符编码、数据划分和 batch 采样。
- `train.py` / `generate.py`：训练和生成入口。
- `tests/test_smoke.py`：正确性与训练→保存→生成闭环测试。

## 环境

Python 3.12+，使用 uv 安装依赖：

```bash
uv sync
```

当前项目在 Linux/Windows 上配置了 PyTorch CUDA 13.0 包源；使用 GPU 需要匹配的驱动。
脚本默认在 CUDA 可用时使用 GPU，也可以通过 `--device cpu` 强制使用 CPU。

## 快速验证闭环

下面是小模型短跑，只验证流程，不代表训练质量。默认数据文件不存在时会下载 Tiny Shakespeare，需要联网。

```bash
uv run python train.py \
  --device cpu --block-size 32 --batch-size 4 \
  --n-layer 1 --n-head 2 --n-embd 32 \
  --max-iters 10 --eval-interval 10 --eval-iters 2 \
  --out-dir out/smoke

uv run python generate.py \
  --device cpu --ckpt out/smoke/ckpt.pt \
  --prompt "To be" --tokens 100 --seed 1337
```

使用 `--data /path/to/corpus.txt` 可以指定本地 UTF-8 文本。
当前按前 90% / 后 10% 划分训练集和验证集，每部分都需要至少 `block_size + 2` 个字符。
生成 prompt 必须非空，且所有字符都在训练词表内；temperature 应大于零。

完整默认配置的训练和生成：

```bash
uv run python train.py
uv run python generate.py --ckpt out/ckpt.pt --prompt "To be" --seed 1337
```

默认模型比短跑配置大得多；显存不足时，先减小 `--batch-size`，必要时减小上下文和模型规模。
更多参数见 `uv run python train.py --help` 和 `uv run python generate.py --help`。

## Checkpoint 约定

- `out/ckpt.pt` 保存验证 loss 最好的模型，重复使用同一个输出目录会覆盖该文件。
- `config` 保存为普通字典；生成端使用 `torch.load(..., weights_only=True)`。
- `iter` 表示已完成的参数更新次数；评估发生在更新后，最后一步也会评估。
- 当前文件用于推理，不包含 optimizer / RNG 等完整训练状态，**不支持断点续训**。
- 修复前将 `GPTConfig` 对象直接写入文件的旧 checkpoint 不兼容当前加载方式。建议重新训练；如需保留旧权重，应仅对自己信任的旧文件做离线格式迁移，不要通过关闭安全加载来打开来源不明的文件。

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

测试使用 CPU 和临时本地语料，不下载数据、不依赖 GPU，也不会覆盖 `out/` 中的模型。
覆盖前向/反向、因果性、小 batch 过拟合、保存加载一致性、导入无副作用和 CLI 生成闭环。
