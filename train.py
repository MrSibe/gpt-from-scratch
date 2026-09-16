"""训练字符级 GPT，并把每次实验的记录写入独立目录。

默认产物目录为 runs/<时间戳>-<模型标签>/，包含：

  config.json   模型/训练配置、数据标识、代码版本和运行环境
  metrics.csv   每步一行的训练指标；评估步额外包含验证指标
  best.pt       验证 loss 最低的模型权重（供 generate.py 使用）
  last.pt       最近一个评估点的完整训练状态，可用于续训
  summary.json  训练结束时的汇总（best loss、步数、总耗时、峰值显存）

目录已存在时会自动追加 -2、-3 等后缀，不会覆盖旧实验。

续训：`--resume <run>/last.pt` 会回到该实验目录追加 metrics.csv，并恢复
权重、优化器状态、随机数状态和累计训练时间。模型结构、batch size、seed
和数据以 checkpoint 为准（命令行给出不同值时只提示，不生效）。
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
    p.add_argument(
        "--resume",
        default=None,
        help="从某个实验目录的 last.pt 继续训练；会复用该目录",
    )
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--seed", type=int, default=1337, help="训练用随机种子")
    p.add_argument("--block-size", type=positive_int, default=256)
    p.add_argument("--batch-size", type=positive_int, default=64)
    p.add_argument(
        "--max-iters",
        type=positive_int,
        default=5000,
        help="参数更新总次数；续训时是包括已完成步数在内的目标值",
    )
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


def save_best(path, model, cfg, dataset, step, val_loss):
    """推理用的轻量 checkpoint：只含权重、模型配置和词表。"""
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


def save_last(
    path,
    model,
    optimizer,
    cfg,
    dataset,
    data_path,
    step,
    val_loss,
    best_val,
    best_step,
    train_rng,
    wall_time_s,
    device,
    batch_size,
    seed,
):
    """续训用的完整状态：权重、优化器、随机数和进度。

    随机数状态在评估之后保存。评估使用独立生成器，所以此处保存的全局
    状态正好等于最后一个训练步结束时的状态。
    """
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "training": {
                "batch_size": batch_size,
                "seed": seed,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "betas": [BETA1, BETA2],
            },
            "stoi": dataset.stoi,
            "itos": dataset.itos,
            "data": {
                "path": str(Path(data_path).resolve()),
                "sha256": file_sha256(data_path),
            },
            "iter": step,
            "val_loss": val_loss,
            "best_val": best_val,
            "best_step": best_step,
            "wall_time_s": wall_time_s,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                "train_generator": train_rng.get_state(),
            },
        },
        path,
    )


def build_config(args, cfg, dataset, device, n_params, eval_seed, run_dir, batch_size):
    data_path = Path(args.data)
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "parameters": n_params,
        "model": asdict(cfg),
        "training": {
            "batch_size": batch_size,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": [BETA1, BETA2],
            "seed": args.seed,
            "eval_seed": eval_seed,
            "tokens_per_step": batch_size * cfg.block_size,
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

    resume_path = Path(args.resume).resolve() if args.resume else None
    checkpoint = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise SystemExit(f"找不到 checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        for key in ("model", "optimizer", "config", "training", "rng", "iter"):
            if key not in checkpoint:
                raise SystemExit(f"{resume_path} 缺少 {key}，不是可续训的 last.pt")

    # 结构、batch size 和 seed 以 checkpoint 为准，避免续训时静默改变实验条件。
    if checkpoint is not None:
        cfg = GPTConfig(**checkpoint["config"])
        saved = checkpoint["training"]
        batch_size = int(saved["batch_size"])
        seed = int(saved["seed"])
        for label, given, kept in (
            ("block-size", args.block_size, cfg.block_size),
            ("batch-size", args.batch_size, batch_size),
            ("seed", args.seed, seed),
            ("n-layer", args.n_layer, cfg.n_layer),
            ("n-head", args.n_head, cfg.n_head),
            ("n-embd", args.n_embd, cfg.n_embd),
        ):
            if given != kept:
                print(f"注意: --{label}={given} 被忽略，使用 checkpoint 的 {kept}")
        start_step = int(checkpoint["iter"])
        best_val = float(
            checkpoint.get("best_val", checkpoint.get("val_loss", float("inf")))
        )
        best_step = int(checkpoint.get("best_step", start_step))
        time_offset = float(checkpoint.get("wall_time_s", 0.0))
        if args.max_iters <= start_step:
            raise SystemExit(
                f"--max-iters={args.max_iters} 不大于已完成的 {start_step} 步，无需续训"
            )
        run_dir = resume_path.parent
        if args.out_dir and Path(args.out_dir).resolve() != run_dir:
            raise SystemExit(
                f"--out-dir 与 checkpoint 所在目录不一致: "
                f"{Path(args.out_dir).resolve()} != {run_dir}"
            )
    else:
        batch_size = args.batch_size
        seed = args.seed
        cfg = GPTConfig(
            vocab_size=0,  # 由数据集确定，先占位
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
        )
        start_step = 0
        best_val = float("inf")
        best_step = 0
        time_offset = 0.0
        run_dir = None

    torch.manual_seed(seed)

    dataset = CharDataset(path=args.data, block_size=cfg.block_size, device=device)
    # 当前采样器需要每个 split 至少有 block_size + 2 个 token。
    if min(len(dataset.train_data), len(dataset.val_data)) < cfg.block_size + 2:
        raise ValueError("训练集和验证集都需要至少 block_size + 2 个 token")
    if cfg.vocab_size not in (0, dataset.vocab_size):
        raise SystemExit(
            f"checkpoint 词表大小 {cfg.vocab_size} 与数据 {dataset.vocab_size} 不一致"
        )
    cfg.vocab_size = dataset.vocab_size

    if checkpoint is not None:
        saved_data = checkpoint.get("data") or {}
        if saved_data.get("sha256") != file_sha256(args.data):
            raise SystemExit(
                "数据文件与 checkpoint 不一致（SHA-256 不同），拒绝续训；"
                "请使用原来的 --data 或改为新的实验"
            )
        if checkpoint["stoi"] != dataset.stoi:
            raise SystemExit("checkpoint 词表与当前数据不一致，拒绝续训")

    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    eval_seed = seed + 1
    train_rng = torch.Generator().manual_seed(seed)
    tokens_per_step = batch_size * cfg.block_size

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_rng.set_state(checkpoint["rng"]["train_generator"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        cuda_states = checkpoint["rng"].get("cuda")
        if device == "cuda":
            if cuda_states and len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_states)
            else:
                print("注意: CUDA 随机数状态与当前设备不匹配，未恢复")
        logger = RunLogger(run_dir, append=True)
        logger.write_json(
            "resume.json",
            {
                "resumed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "from": str(resume_path),
                "start_step": start_step,
                "target_steps": args.max_iters,
                "device": device,
            },
        )
    else:
        logger = RunLogger(
            Path(args.out_dir) if args.out_dir else default_run_dir(args)
        )
        config = build_config(
            args, cfg, dataset, device, n_params, eval_seed, logger.run_dir, batch_size
        )
        logger.write_json("config.json", config)

    print(f"实验目录: {logger.run_dir}")
    print(
        f"设备: {device} | 词表: {dataset.vocab_size} | "
        f"训练 token: {len(dataset.train_data)} | 验证 token: {len(dataset.val_data)}"
    )
    print(
        f"参数量: {n_params / 1e6:.2f}M | 每步 token: {tokens_per_step}"
        + (f" | 从 step {start_step} 继续" if start_step else "")
    )

    t0 = time.perf_counter()
    overall_peak = 0.0

    with logger:
        for step in range(start_step + 1, args.max_iters + 1):
            # step 表示已完成的参数更新次数；最后一次更新后也会评估。
            reset_peak_memory(device)
            step_start = time.perf_counter()

            x, y = dataset.get_batch("train", batch_size, generator=train_rng)
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
                    model, dataset, batch_size, args.eval_iters, eval_seed
                )
                synchronize(device)
                row["eval_time_s"] = time.perf_counter() - eval_start
                row["train_loss_eval"] = losses["train"]
                row["val_loss"] = losses["val"]

                notes = []
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    best_step = step
                    save_best(
                        logger.run_dir / "best.pt",
                        model,
                        cfg,
                        dataset,
                        step,
                        best_val,
                    )
                    notes.append("已保存 best.pt")

                # 先固定本行的累计时间，再存 last.pt。否则续训时恢复的累计时间
                # 会小于上一行，wall_time_s 在续训节点上回退。
                row["wall_time_s"] = time_offset + (time.perf_counter() - t0)

                # last.pt 保存评估之后的状态；评估不消耗训练随机数流，
                # 因此此时保存的随机数状态等于最后一个训练步结束时的状态。
                save_last(
                    logger.run_dir / "last.pt",
                    model,
                    optimizer,
                    cfg,
                    dataset,
                    args.data,
                    step,
                    losses["val"],
                    best_val,
                    best_step,
                    train_rng,
                    row["wall_time_s"],
                    device,
                    batch_size,
                    seed,
                )
                notes.append("已保存 last.pt")

                print(
                    f"step {step:5d} | train {losses['train']:.4f} "
                    f"| val {losses['val']:.4f} | grad {float(grad_norm):.3f} "
                    f"| {row['wall_time_s']:.1f}s | {' '.join(notes)}"
                )

            if row["peak_memory_mb"] is not None:
                overall_peak = max(overall_peak, row["peak_memory_mb"])
            row.setdefault("wall_time_s", time_offset + (time.perf_counter() - t0))
            logger.log(**row)

        total_time = time_offset + (time.perf_counter() - t0)
        summary = {
            "steps": args.max_iters,
            "tokens_seen": args.max_iters * tokens_per_step,
            "best_val_loss": best_val,
            "best_step": best_step,
            "wall_time_s": total_time,
            "session_time_s": time.perf_counter() - t0,
            "resumed_from_step": start_step or None,
            "peak_memory_mb": overall_peak or None,
            "metrics": str(logger.metrics_path),
        }
        logger.write_json("summary.json", summary)

    print(f"训练完成 | best val loss = {best_val:.4f} @ step {best_step}")
    print(f"累计耗时 {total_time:.1f}s | 记录: {logger.metrics_path}")
    print(
        f"生成示例: uv run python generate.py --ckpt {logger.run_dir / 'best.pt'} "
        '--prompt "To be" --seed 1337'
    )
    if args.max_iters < 5000:
        print(
            f"续训示例: uv run python train.py --resume "
            f"{logger.run_dir / 'last.pt'} --max-iters {args.max_iters * 2}"
        )
    print(f"绘图示例: uv run python plot.py {logger.run_dir}")


if __name__ == "__main__":
    main()
