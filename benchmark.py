"""独立训练计算测速：设备上固定合成 batch，warmup 后按窗口测量。

包含 forward/backward/AdamW/AMP scaler；不包含数据采样、H2D、评估、日志或保存。
只从 train.py 取默认优化器超参数，不复用它的训练循环和观测逻辑。
"""

import argparse
import csv
import json
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from log import environment_info, git_info, unique_dir
from model import GPT, GPTConfig
from train import BETAS, LR, WEIGHT_DECAY, build_optimizer


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="固定合成 batch 的训练计算测速（不是端到端吞吐）"
    )
    p.add_argument("--run-name", default="benchmark")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=positive_int, default=GPTConfig.block_size)
    p.add_argument("--batch-size", type=positive_int, default=16)
    p.add_argument("--vocab-size", type=positive_int, default=GPTConfig.vocab_size)
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
    p.add_argument(
        "--attention", choices=("manual", "sdpa"), default=GPTConfig.attention
    )
    p.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--warmup", type=positive_int, default=50)
    p.add_argument("--steps", type=positive_int, default=100, help="每个测量窗口的步数")
    p.add_argument("--repeats", type=positive_int, default=5, help="连续测量窗口数")
    args = p.parse_args(argv)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA 不可用")
    if args.dtype != "fp32" and args.device != "cuda":
        p.error("当前 fp16/bf16 实验仅支持 CUDA；CPU 请使用 fp32")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        p.error("当前设备不支持 BF16")
    if not 0 <= args.dropout < 1:
        p.error("dropout 必须在 [0, 1)")
    if args.n_embd % args.n_head:
        p.error("n-embd 必须能整除 n-head")
    if not args.run_name or any(c in args.run_name for c in "/\\"):
        p.error("run-name 必须是非空标签，不能包含路径分隔符")
    return args


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def optimizer_steps(optimizer):
    """AdamW 为每个有梯度的参数计数；取任一已有 state 的参数即可。

    优化器现在分成 decay / no-decay 两组，不再假定组 0 的第一个参数有 state。
    """
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if state:
                return int(state["step"])
    return 0


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    torch.manual_seed(args.seed)
    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        attention=args.attention,
        dropout=args.dropout,
        tie_embeddings=args.tie_embeddings,
    )
    raw_model = GPT(cfg).to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizer = build_optimizer(raw_model, LR, BETAS, WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
    x = torch.randint(cfg.vocab_size, (args.batch_size, cfg.block_size), device=device)
    y = torch.randint(cfg.vocab_size, x.shape, device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        precision = (
            nullcontext()
            if args.dtype == "fp32"
            else torch.autocast(
                device_type=device,
                dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype],
            )
        )
        with precision:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.warmup):
        step()
    synchronize(device)
    warmup_time = time.perf_counter() - start
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    windows = []
    for index in range(args.repeats):
        updates_before = optimizer_steps(optimizer)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            loss = step()
        synchronize(device)
        elapsed = time.perf_counter() - start
        # 在计时外读取计数；FP16 跳步会省掉 AdamW，不能混作完整更新的吞吐。
        updates = optimizer_steps(optimizer) - updates_before
        if updates != args.steps:
            raise RuntimeError(
                f"测速窗口只完成 {updates}/{args.steps} 次 optimizer 更新；"
                "可能发生 FP16 溢出，请增加 warmup 或更换精度后重测"
            )
        windows.append(
            {
                "window": index + 1,
                "steps": args.steps,
                "optimizer_steps": updates,
                "elapsed_s": elapsed,
                "step_time_s": elapsed / args.steps,
                "tokens_per_sec": args.batch_size
                * cfg.block_size
                * args.steps
                / elapsed,
            }
        )
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    # 检查位于计时区间之外；非有限 loss 的结果不作为有效 benchmark 保存。
    if not torch.isfinite(loss).item() or not all(
        torch.isfinite(p).all().item() for p in raw_model.parameters()
    ):
        raise RuntimeError("benchmark 出现非有限 loss 或权重，结果无效")

    run_dir = unique_dir(
        Path("runs") / f"{datetime.now(UTC).astimezone():%Y%m%d-%H%M%S}-{args.run_name}"
    )
    run_dir.mkdir(parents=True)
    config = {
        "benchmark": vars(args),
        "model": asdict(cfg),
        "optimizer": {
            "name": "AdamW",
            "lr": LR,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "decay_scope": "dim>=2 only; LayerNorm and bias use weight_decay=0",
        },
        "parameters": sum(p.numel() for p in raw_model.parameters()),
        "scope": "fixed device-resident synthetic batch: forward + backward + optimizer + scaler",
        "attention_backend": "automatic (not profiled)"
        if args.attention == "sdpa"
        else "manual",
        "environment": environment_info(device),
        "code": git_info(Path(__file__).resolve().parent),
    }
    rates = [w["tokens_per_sec"] for w in windows]
    summary = {
        "warmup_time_s": warmup_time,
        "warmup_includes_compile": args.compile,
        "tokens_per_sec_median": statistics.median(rates),
        "tokens_per_sec_min": min(rates),
        "tokens_per_sec_max": max(rates),
        "step_time_s_median": statistics.median(w["step_time_s"] for w in windows),
        "peak_memory_mb": peak,
        "final_loss": loss.item(),
    }
    for name, payload in (("config.json", config), ("summary.json", summary)):
        (run_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    with (run_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    print(
        f"训练计算吞吐: {statistics.median(rates):.1f} tokens/s "
        f"[{min(rates):.1f}, {max(rates):.1f}] | 记录: {run_dir}"
    )


if __name__ == "__main__":
    main()
