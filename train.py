"""训练字符级 GPT，并把每次实验的记录写入独立目录。

默认产物目录为 runs/<时间戳>-<模型标签>/，包含：

  config.json   模型/训练配置、数据标识、代码版本和运行环境
  metrics.csv   每步一行的训练指标；评估步额外包含验证指标
  best.pt       验证 loss 最低的模型权重（供 generate.py 使用）
  summary.json  训练结束时的汇总（best loss、步数、总耗时、峰值显存）

目录已存在时会自动追加 -2、-3 等后缀，不会覆盖旧实验。
当前不支持断点续训：best.pt 只保存推理所需的权重和词表。
"""

import argparse
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from data import DATA_PATH, CharDataset
from model import GPT, GPTConfig
from runlog import (
    RunLogger,
    environment_info,
    file_sha256,
    git_info,
)

ROOT_DIR = Path(__file__).resolve().parent

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="训练字符级 GPT")
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--runs-dir", default="runs", help="实验目录的父目录")
    p.add_argument(
        "--out-dir",
        default=None,
        help="显式指定实验目录，优先于 --runs-dir 和 --run-name",
    )
    p.add_argument("--run-name", default=None, help="实验目录的标签后缀")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--seed", type=int, default=1337, help="训练用随机种子")
    p.add_argument("--block-size", type=positive_int, default=256)
    p.add_argument("--batch-size", type=positive_int, default=64)
    p.add_argument("--max-iters", type=positive_int, default=5000)
    p.add_argument("--eval-interval", type=positive_int, default=250)
    p.add_argument("--eval-iters", type=positive_int, default=200)
    p.add_argument("--n-layer", type=positive_int, default=6)
    p.add_argument("--n-head", type=positive_int, default=6)
    p.add_argument("--n-embd", type=positive_int, default=384)
    return p.parse_args(argv)


def default_run_dir(args):
    # 目录名使用本地时间，方便对照日常记录。
    stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S")
    name = args.run_name or f"gpt{args.n_layer}x{args.n_embd}"
    return Path(args.runs_dir) / f"{stamp}-{name}"


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device):
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / 2**20


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, eval_iters, eval_seed):
    """用固定的一组评估 batch 计算平均 loss。

    每次调用都用 eval_seed 重新初始化生成器，因此：
    同一实验的不同评估点使用完全相同的样本；评估也不消耗训练随机数流，
    改变 --eval-interval 不会影响训练数据顺序。
    """
    generator = torch.Generator().manual_seed(eval_seed)
    was_training = model.training
    model.eval()
    out = {}
    try:
        for split in ("train", "val"):
            losses = []
            for _ in range(eval_iters):
                x, y = dataset.get_batch(split, batch_size, generator=generator)
                _, loss = model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return out


def save_checkpoint(path, model, cfg, dataset, step, val_loss):
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "stoi": dataset.stoi,
            "itos": dataset.itos,
            "iter": step,
            "val_loss": val_loss,
        },
        path,
    )


def build_config(args, cfg, dataset, device, n_params, eval_seed, run_dir):
    data_path = Path(args.data)
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "parameters": n_params,
        "model": asdict(cfg),
        "training": {
            "batch_size": args.batch_size,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": [BETA1, BETA2],
            "seed": args.seed,
            "eval_seed": eval_seed,
            "tokens_per_step": args.batch_size * args.block_size,
            "grad_clip": None,
        },
        "data": {
            "path": str(data_path.resolve()),
            "bytes": data_path.stat().st_size,
            "sha256": file_sha256(data_path),
            "val_ratio": dataset.val_ratio,
            "vocab_size": dataset.vocab_size,
            "train_tokens": len(dataset.train_data),
            "val_tokens": len(dataset.val_data),
        },
        "environment": environment_info(device),
        "code": git_info(ROOT_DIR),
    }


def main(argv=None):
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    dataset = CharDataset(path=args.data, block_size=args.block_size, device=device)
    # 当前采样器需要每个 split 至少有 block_size + 2 个 token。
    if min(len(dataset.train_data), len(dataset.val_data)) < args.block_size + 2:
        raise ValueError("训练集和验证集都需要至少 block_size + 2 个 token")

    cfg = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # 训练和评估使用不同的生成器：两者互不干扰，且都独立于全局随机数流。
    eval_seed = args.seed + 1
    train_rng = torch.Generator().manual_seed(args.seed)
    tokens_per_step = args.batch_size * args.block_size

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    logger = RunLogger(Path(args.out_dir) if args.out_dir else default_run_dir(args))
    config = build_config(
        args, cfg, dataset, device, n_params, eval_seed, logger.run_dir
    )
    logger.write_json("config.json", config)
    print(f"实验目录: {logger.run_dir}")
    print(
        f"设备: {device} | 词表: {dataset.vocab_size} | "
        f"训练 token: {len(dataset.train_data)} | 验证 token: {len(dataset.val_data)}"
    )
    print(f"参数量: {n_params / 1e6:.2f}M | 每步 token: {tokens_per_step}")

    best_val = float("inf")
    best_step = 0
    overall_peak = 0.0
    t0 = time.perf_counter()

    with logger:
        for step in range(1, args.max_iters + 1):
            # step 表示已完成的参数更新次数；最后一次更新后也会评估。
            reset_peak_memory(device)
            step_start = time.perf_counter()

            x, y = dataset.get_batch("train", args.batch_size, generator=train_rng)
            _, train_loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            # 尚未启用梯度裁剪，这里记录的是裁剪前的梯度范数。
            grad_norm = torch.nn.utils.get_total_norm(model.parameters())
            optimizer.step()
            synchronize(device)
            step_time = time.perf_counter() - step_start

            row = {
                "step": step,
                "tokens_seen": step * tokens_per_step,
                "train_loss_step": train_loss.item(),
                "lr": LEARNING_RATE,
                "grad_norm": float(grad_norm),
                "step_time_s": step_time,
                "train_tokens_per_sec": tokens_per_step / step_time,
                "peak_memory_mb": peak_memory_mb(device),
            }

            if step % args.eval_interval == 0 or step == args.max_iters:
                eval_start = time.perf_counter()
                losses = estimate_loss(
                    model, dataset, args.batch_size, args.eval_iters, eval_seed
                )
                synchronize(device)
                row["eval_time_s"] = time.perf_counter() - eval_start
                row["train_loss_eval"] = losses["train"]
                row["val_loss"] = losses["val"]

                if losses["val"] < best_val:
                    best_val = losses["val"]
                    best_step = step
                    save_checkpoint(
                        logger.run_dir / "best.pt",
                        model,
                        cfg,
                        dataset,
                        step,
                        best_val,
                    )
                    saved = " 已保存 best.pt"
                else:
                    saved = ""

                print(
                    f"step {step:5d} | train {losses['train']:.4f} "
                    f"| val {losses['val']:.4f} | grad {float(grad_norm):.3f} "
                    f"| {time.perf_counter() - t0:.1f}s{saved}"
                )

            if row["peak_memory_mb"] is not None:
                overall_peak = max(overall_peak, row["peak_memory_mb"])
            row["wall_time_s"] = time.perf_counter() - t0
            logger.log(**row)

        total_time = time.perf_counter() - t0
        summary = {
            "steps": args.max_iters,
            "tokens_seen": args.max_iters * tokens_per_step,
            "best_val_loss": best_val,
            "best_step": best_step,
            "wall_time_s": total_time,
            "peak_memory_mb": overall_peak or None,
            "metrics": str(logger.metrics_path),
        }
        logger.write_json("summary.json", summary)

    print(f"训练完成 | best val loss = {best_val:.4f} @ step {best_step}")
    print(f"总耗时 {total_time:.1f}s | 记录: {logger.metrics_path}")
    print(
        f"生成示例: uv run python generate.py --ckpt {logger.run_dir / 'best.pt'} "
        '--prompt "To be" --seed 1337'
    )
    print(f"绘图示例: uv run python plot.py {logger.run_dir}")


if __name__ == "__main__":
    main()
