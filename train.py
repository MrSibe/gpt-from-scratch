"""短训练实验：只保存 config.json、metrics.csv、best.pt、summary.json。

训练 loss 暂存在设备上，到评估点批量写入，避免每步 .item() 同步 CUDA。
性能对照请用 benchmark.py；这里不测单步耗时或吞吐，不支持续训。
"""

import argparse
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from data import DATA_PATH, CharDataset
from log import RunLogger, environment_info, file_sha256, git_info
from model import GPT, GPTConfig

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="训练字符级 GPT（不支持续训）")
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--run-name", default="gpt")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=positive_int, default=256)
    p.add_argument("--batch-size", type=positive_int, default=64)
    p.add_argument("--max-iters", type=positive_int, default=5000)
    p.add_argument("--eval-interval", type=positive_int, default=250)
    p.add_argument("--eval-iters", type=positive_int, default=200)
    p.add_argument("--n-layer", type=positive_int, default=6)
    p.add_argument("--n-head", type=positive_int, default=6)
    p.add_argument("--n-embd", type=positive_int, default=384)
    p.add_argument("--attention", choices=("manual", "sdpa"), default="manual")
    p.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args(argv)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA 不可用")
    if args.dtype != "fp32" and args.device != "cuda":
        p.error("当前 fp16/bf16 实验仅支持 CUDA；CPU 请使用 fp32")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        p.error("当前设备不支持 BF16")
    if args.n_embd % args.n_head:
        p.error("n-embd 必须能整除 n-head")
    if not args.run_name or any(c in args.run_name for c in "/\\"):
        p.error("run-name 必须是非空标签，不能包含路径分隔符")
    return args


def autocast(device, dtype):
    if dtype == "fp32":
        return nullcontext()
    return torch.autocast(
        device_type=device, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    )


@torch.no_grad()
def estimate_val_loss(model, dataset, batch_size, eval_iters, eval_seed, dtype="fp32"):
    """固定验证样本，不推进训练采样或 dropout 的随机数流。"""
    generator = torch.Generator().manual_seed(eval_seed)
    was_training = model.training
    model.eval()
    try:
        losses = []
        for _ in range(eval_iters):
            x, y = dataset.get_batch("val", batch_size, generator=generator)
            with autocast(dataset.device, dtype):
                _, loss = model(x, y)
            losses.append(loss.detach())
        return torch.stack(losses).mean().item()
    finally:
        model.train(was_training)


def save_checkpoint(path, model, dataset, step, val_loss):
    # 始终保存原模型，避免 torch.compile 的 _orig_mod. 权重名前缀。
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model.cfg),
            "stoi": dataset.stoi,
            "itos": dataset.itos,
            "iter": step,
            "val_loss": val_loss,
        },
        path,
    )


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    torch.manual_seed(args.seed)
    dataset = CharDataset(path=args.data, block_size=args.block_size, device=device)
    if min(len(dataset.train_data), len(dataset.val_data)) < args.block_size + 2:
        raise ValueError("训练集和验证集都需要至少 block_size + 2 个 token")
    cfg = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        attention=args.attention,
    )
    raw_model = GPT(cfg).to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
    train_rng = torch.Generator().manual_seed(args.seed)
    tokens_per_step = args.batch_size * cfg.block_size
    run_dir = (
        Path("runs") / f"{datetime.now(UTC).astimezone():%Y%m%d-%H%M%S}-{args.run_name}"
    )

    with RunLogger(run_dir) as logger:
        logger.write_json(
            "config.json",
            {
                "created_at": datetime.now(UTC).astimezone().isoformat(),
                "model": asdict(cfg),
                "parameters": sum(p.numel() for p in raw_model.parameters()),
                "training": {
                    **vars(args),
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "betas": BETAS,
                    "eval_seed": args.seed + 1,
                    "tokens_per_step": tokens_per_step,
                    "grad_clip": None,
                },
                "data": {
                    "path": str(Path(args.data).resolve()),
                    "sha256": file_sha256(args.data),
                    "val_ratio": dataset.val_ratio,
                    "vocab_size": dataset.vocab_size,
                    "train_tokens": len(dataset.train_data),
                    "val_tokens": len(dataset.val_data),
                },
                "environment": environment_info(device),
                "code": git_info(Path(__file__).resolve().parent),
            },
        )
        print(f"实验目录: {logger.run_dir}")
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        best_val, best_step = float("inf"), 0
        pending_losses = []

        for step in range(1, args.max_iters + 1):
            evaluate = step % args.eval_interval == 0 or step == args.max_iters
            x, y = dataset.get_batch("train", args.batch_size, generator=train_rng)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, args.dtype):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            if evaluate:
                # get_total_norm 不会自动取 .grad；AMP 下先解除梯度缩放。
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.get_total_norm(
                    [p.grad for p in raw_model.parameters() if p.grad is not None],
                    norm_type=2.0,
                )
            scaler.step(optimizer)
            scaler.update()
            pending_losses.append(loss.detach())
            if not evaluate:
                continue

            # 一次传回这一段的 loss；普通训练步不读取 CUDA 标量。
            losses = torch.stack(pending_losses).cpu().tolist()
            grad_value = float(grad_norm)
            val_loss = estimate_val_loss(
                model,
                dataset,
                args.batch_size,
                args.eval_iters,
                args.seed + 1,
                args.dtype,
            )
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                save_checkpoint(
                    logger.run_dir / "best.pt", raw_model, dataset, step, val_loss
                )
            elapsed = time.perf_counter() - start
            first_step = step - len(losses) + 1
            for logged_step, train_loss in enumerate(losses, first_step):
                logger.log(
                    step=logged_step,
                    tokens_seen=logged_step * tokens_per_step,
                    train_loss_step=train_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    val_loss=val_loss if logged_step == step else None,
                    grad_norm=grad_value if logged_step == step else None,
                    wall_time_s=elapsed if logged_step == step else None,
                )
            pending_losses.clear()
            print(
                f"step {step:5d} | train {losses[-1]:.4f} | val {val_loss:.4f} "
                f"| grad {grad_value:.3f} | {elapsed:.1f}s"
            )

        if device == "cuda":
            torch.cuda.synchronize()
        logger.write_json(
            "summary.json",
            {
                "steps": args.max_iters,
                "tokens_seen": args.max_iters * tokens_per_step,
                "best_val_loss": best_val,
                "best_step": best_step,
                "wall_time_s": time.perf_counter() - start,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20
                if device == "cuda"
                else None,
            },
        )
        print(
            f"训练完成 | best val {best_val:.4f} @ step {best_step} | {logger.run_dir}"
        )


if __name__ == "__main__":
    main()
