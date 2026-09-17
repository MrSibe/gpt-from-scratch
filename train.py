"""可读的 GPT 训练循环：梯度累积、AMP、裁剪、warmup/cosine 和固定验证。

step 是一次累积窗口（一次更新尝试），optimizer_steps 是实际成功更新次数。
只保存最佳模型，不支持续训；普通 BF16/FP32 步不读取 CUDA loss 标量。
"""

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from data import BPE_DATA_PATH, DATA_PATH, BPEDataset, CharDataset
from log import RunLogger, environment_info, git_info
from model import GPT, GPTConfig

# 训练超参数的默认值就写在这里：CLI 默认值和 benchmark.py / profiler.py 复现默认优化器
# 配置时读的都是这几个常量，避免同一数值在多处各写一份后悄悄漂移。
# 模型结构默认值仍归 model.GPTConfig。
LR = 1e-3
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
MAX_ITERS = 20000
WARMUP_RATIO = 0.02
MIN_LR = 3e-5
GRAD_CLIP = 1.0


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="训练字符级 / BPE GPT（不支持续训）")
    p.add_argument("--tokenizer", choices=("char", "bpe"), default="bpe")
    p.add_argument("--data", help=f"bpe 默认 {BPE_DATA_PATH}；char 默认 {DATA_PATH}")
    p.add_argument("--run-name", default="gpt")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=positive_int, default=GPTConfig.block_size)
    p.add_argument(
        "--batch-size",
        type=positive_int,
        default=16,
        help="每个 micro-step 的序列数",
    )
    p.add_argument("--grad-accum-steps", type=positive_int, default=8)
    p.add_argument(
        "--max-iters",
        type=positive_int,
        default=MAX_ITERS,
        help="更新尝试次数，不是 micro-step 数",
    )
    p.add_argument("--eval-interval", type=positive_int, default=250)
    p.add_argument("--eval-iters", type=positive_int, default=50)
    p.add_argument(
        "--eval-batch-size",
        type=positive_int,
        default=16,
        help="独立于训练 batch，消融时保持固定",
    )
    p.add_argument("--n-layer", type=positive_int, default=GPTConfig.n_layer)
    p.add_argument("--n-head", type=positive_int, default=GPTConfig.n_head)
    p.add_argument("--n-embd", type=positive_int, default=GPTConfig.n_embd)
    p.add_argument("--dropout", type=float, default=GPTConfig.dropout)
    p.add_argument(
        "--tie-embeddings",
        action=argparse.BooleanOptionalAction,
        default=GPTConfig.tie_embeddings,
        help="输入 embedding 与输出投影共享权重；默认不共享",
    )
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--betas", type=float, nargs=2, default=BETAS)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument(
        "--grad-clip", type=float, default=GRAD_CLIP, help="全局梯度范数上限；0=不裁剪"
    )
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    warmup = p.add_mutually_exclusive_group()
    warmup.add_argument("--warmup-iters", type=int, default=None)
    warmup.add_argument(
        "--warmup-ratio",
        type=float,
        default=WARMUP_RATIO,
        help="乘 max-iters 后向下取整；默认 2%%，也可用 warmup-iters 指定绝对步数",
    )
    p.add_argument(
        "--min-lr", type=float, default=MIN_LR, help="cosine 的最后一次计划更新所用 LR"
    )
    p.add_argument(
        "--attention", choices=("manual", "sdpa"), default=GPTConfig.attention
    )
    p.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args(argv)
    if args.data is None:
        args.data = BPE_DATA_PATH if args.tokenizer == "bpe" else DATA_PATH
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA 不可用")
    if args.dtype != "fp32" and args.device != "cuda":
        p.error("当前 fp16/bf16 实验仅支持 CUDA；CPU 请使用 fp32")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        p.error("当前设备不支持 BF16")
    if args.n_embd % args.n_head:
        p.error("n-embd 必须能整除 n-head")
    values = [
        args.dropout,
        args.lr,
        args.min_lr,
        args.weight_decay,
        args.grad_clip,
        args.warmup_ratio,
        *args.betas,
    ]
    if not all(math.isfinite(v) for v in values):
        p.error("浮点超参数必须有限")
    if not 0 <= args.dropout < 1 or not all(0 <= b < 1 for b in args.betas):
        p.error("dropout 和 betas 必须在 [0, 1)")
    if args.lr <= 0 or min(args.min_lr, args.weight_decay, args.grad_clip) < 0:
        p.error("lr 必须为正，min-lr / weight-decay / grad-clip 必须非负")
    if args.lr_schedule == "cosine" and args.min_lr > args.lr:
        p.error("min-lr 不能大于 lr")
    if not 0 <= args.warmup_ratio < 1:
        p.error("warmup-ratio 必须在 [0, 1)")
    if args.warmup_iters is None:
        args.warmup_iters = int(args.max_iters * args.warmup_ratio)
    else:
        args.warmup_ratio = None  # 显式使用绝对步数，配置中不再保留未生效的默认比例。
    if not 0 <= args.warmup_iters < args.max_iters:
        p.error("warmup-iters 必须非负且小于 max-iters")
    if not args.run_name or any(c in args.run_name for c in "/\\"):
        p.error("run-name 必须是非空标签，不能包含路径分隔符")
    return args


def get_lr(step, args):
    """step 从 1 开始，按成功 optimizer update 调度，FP16 跳步不推进。"""
    if step <= args.warmup_iters:
        return args.lr * step / args.warmup_iters
    if args.lr_schedule == "constant":
        return args.lr
    # warmup 后从 peak 到 min；若只有一个 decay update，则使用 peak。
    progress = (step - args.warmup_iters - 1) / max(
        1, args.max_iters - args.warmup_iters - 1
    )
    progress = min(1.0, max(0.0, progress))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def autocast(device, dtype):
    if dtype == "fp32":
        return nullcontext()
    return torch.autocast(
        device_type=device, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    )


def train_step(
    model,
    optimizer,
    scaler,
    batches,
    accum_steps=1,
    dtype="fp32",
    grad_clip=0.0,
    measure_grad=False,
):
    """一次更新尝试：zero → 累积平均梯度 → unscale → clip → step。

    batches 是恰好产生 accum_steps 个等大小 (x, y) 的迭代器。
    返回原始平均 loss、裁剪前 norm（可为 None）、是否真正更新了权重。
    """
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for _ in range(accum_steps):
        x, y = next(batches)
        with autocast(x.device.type, dtype):
            _, loss = model(x, y)
        # 除以累积次数，使累积梯度对应整个有效 batch 的平均 loss。
        scaler.scale(loss / accum_steps).backward()
        total_loss = total_loss + loss.detach() / accum_steps

    norm = None
    if grad_clip > 0 or measure_grad:
        scaler.unscale_(optimizer)  # FP16 梯度先恢复真实尺度，只调用一次。
        parameters = list(model.parameters())
        if grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        else:
            norm = torch.nn.utils.get_total_norm(
                [p.grad for p in parameters if p.grad is not None]
            )
    old_scale = scaler.get_scale() if scaler.is_enabled() else None
    scaler.step(optimizer)
    scaler.update()
    # 仅 FP16 需要读取 scale；溢出时 scale 下降，optimizer.step 被跳过。
    updated = old_scale is None or scaler.get_scale() >= old_scale
    return total_loss, norm, updated


@torch.no_grad()
def estimate_val_loss(model, dataset, batch_size, eval_iters, eval_seed, dtype="fp32"):
    """固定验证窗口，不推进训练采样或 dropout 的随机数流。"""
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
        # 每个 batch 有同样多的有效 token，所以 batch 均值也是 token 加权均值。
        return torch.stack(losses).mean().item()
    finally:
        model.train(was_training)


def save_checkpoint(path, model, dataset, step, val_loss):
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model.cfg),
            "tokenizer": dataset.tokenizer.state(),
            "iter": step,
            "val_loss": val_loss,
        },
        path,
    )


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    torch.manual_seed(args.seed)
    dataset_cls = CharDataset if args.tokenizer == "char" else BPEDataset
    dataset = dataset_cls(path=args.data, block_size=args.block_size, device=device)
    cfg = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        attention=args.attention,
        tie_embeddings=args.tie_embeddings,
    )
    raw_model = GPT(cfg).to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
    train_rng = torch.Generator().manual_seed(args.seed)
    tokens_per_step = args.batch_size * cfg.block_size * args.grad_accum_steps
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
                    "eval_seed": args.seed + 1,
                    "tokens_per_step": tokens_per_step,
                },
                "data": dataset.metadata(),
                "environment": environment_info(device),
                "code": git_info(Path(__file__).resolve().parent),
            },
        )
        print(
            f"实验目录: {logger.run_dir} | {tokens_per_step} tokens/update", flush=True
        )
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        best_val, best_step, optimizer_steps = float("inf"), 0, 0
        pending = []

        for step in range(1, args.max_iters + 1):
            evaluate = step % args.eval_interval == 0 or step == args.max_iters
            lr = get_lr(optimizer_steps + 1, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batches = (
                dataset.get_batch("train", args.batch_size, generator=train_rng)
                for _ in range(args.grad_accum_steps)
            )
            loss, grad_norm, updated = train_step(
                model,
                optimizer,
                scaler,
                batches,
                accum_steps=args.grad_accum_steps,
                dtype=args.dtype,
                grad_clip=args.grad_clip,
                measure_grad=evaluate,
            )
            optimizer_steps += int(updated)
            pending.append((loss, lr, optimizer_steps, not updated))
            if not evaluate:
                continue

            losses = torch.stack([row[0] for row in pending]).cpu().tolist()
            grad_value = float(grad_norm)
            val_loss = estimate_val_loss(
                model,
                dataset,
                args.eval_batch_size,
                args.eval_iters,
                args.seed + 1,
                args.dtype,
            )
            if not all(math.isfinite(v) for v in [*losses, val_loss]):
                raise RuntimeError(
                    "出现非有限 loss，停止训练；请降低 LR / 检查精度和数据"
                )
            val_ppl = math.exp(val_loss) if val_loss < 700 else float("inf")
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                # 始终保存原模型，避免 compile 的 _orig_mod. 权重名前缀。
                save_checkpoint(
                    logger.run_dir / "best.pt", raw_model, dataset, step, val_loss
                )
            elapsed = time.perf_counter() - start
            first_step = step - len(pending) + 1
            for logged_step, (train_loss, record) in enumerate(
                zip(losses, pending), start=first_step
            ):
                _, used_lr, updates, skipped = record
                is_eval_step = logged_step == step
                logger.log(
                    step=logged_step,
                    optimizer_steps=updates,
                    skipped_update=skipped,
                    tokens_seen=logged_step * tokens_per_step,
                    train_loss_step=train_loss,
                    lr=used_lr,
                    val_loss=val_loss if is_eval_step else None,
                    val_ppl=val_ppl if is_eval_step else None,
                    grad_norm=grad_value if is_eval_step else None,
                    wall_time_s=elapsed if is_eval_step else None,
                )
            pending.clear()
            print(
                f"step {step:5d} | train {losses[-1]:.4f} | val {val_loss:.4f} "
                f"| ppl {val_ppl:.2f} | lr {lr:.2g} | grad {grad_value:.3f} | {elapsed:.1f}s",
                flush=True,
            )

        if device == "cuda":
            torch.cuda.synchronize()
        logger.write_json(
            "summary.json",
            {
                "steps": args.max_iters,
                "optimizer_steps": optimizer_steps,
                "skipped_updates": args.max_iters - optimizer_steps,
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
