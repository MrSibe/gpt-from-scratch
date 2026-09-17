"""训练计算的 PyTorch Profiler：算子统计与 Chrome/Perfetto trace。

固定设备上的合成 batch，不含数据采样、H2D、评估或保存权重。
只从 train.py 取默认优化器超参数，不复用它的训练循环和观测逻辑。
Profiler 会扰动性能；吞吐对照请用 benchmark.py。
"""

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from log import environment_info, git_info, unique_dir
from model import GPT, GPTConfig
from train import BETAS, LR, WEIGHT_DECAY


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="训练计算算子分析（不是吞吐测速）")
    p.add_argument("--run-name", default="profiler")
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
    p.add_argument(
        "--warmup", type=positive_int, default=10, help="采集前的训练预热步数"
    )
    p.add_argument("--steps", type=positive_int, default=5, help="实际采集的训练步数")
    p.add_argument(
        "--row-limit", type=positive_int, default=30, help="每张算子表的行数"
    )
    p.add_argument("--record-shapes", action="store_true", help="记录并按输入形状分组")
    p.add_argument(
        "--profile-memory", action="store_true", help="记录内存分配/释放事件"
    )
    p.add_argument("--with-stack", action="store_true", help="记录算子的 Python 调用栈")
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
    raw_model = GPT(cfg).to(device).train()
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
    x = torch.randint(cfg.vocab_size, (args.batch_size, cfg.block_size), device=device)
    y = torch.randint(cfg.vocab_size, x.shape, device=device)

    def step(annotate=False):
        # 外部 warmup 不插桩；采集时用范围标记串起 CPU 调度与 GPU kernel。
        def region(name):
            return record_function(name) if annotate else nullcontext()

        with region("train/zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        precision = (
            nullcontext()
            if args.dtype == "fp32"
            else torch.autocast(
                device_type=device,
                dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype],
            )
        )
        with region("train/forward"), precision:
            _, loss = model(x, y)
        with region("train/backward"):
            scaler.scale(loss).backward()
        with region("train/optimizer"):
            scaler.step(optimizer)
            scaler.update()
        return loss.detach()

    run_dir = unique_dir(
        Path("runs") / f"{datetime.now(UTC).astimezone():%Y%m%d-%H%M%S}-{args.run_name}"
    )
    run_dir.mkdir(parents=True)
    config = {
        "profiler": vars(args),
        "model": asdict(cfg),
        "optimizer": {
            "name": "AdamW",
            "lr": LR,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
        },
        "parameters": sum(p.numel() for p in raw_model.parameters()),
        "scope": "fixed device-resident synthetic batch: forward + backward + optimizer + scaler",
        "profiler_warmup_steps": 1,
        "environment": environment_info(device),
        "code": git_info(Path(__file__).resolve().parent),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Profiler 目录: {run_dir}", flush=True)
    print(f"预热 {args.warmup} 步（含首次编译），随后采集 {args.steps} 步", flush=True)
    for _ in range(args.warmup):
        step()
    synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        # 再给 profiler 本身一步预热，只保留后续 steps 步。
        schedule=schedule(wait=0, warmup=1, active=args.steps, repeat=1),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        for _ in range(1 + args.steps):
            with record_function("train/step"):
                loss = step(annotate=True)
            # 仅在采集窗口结束前同步，不在每步读取 CUDA loss。
            if _ == args.steps:
                synchronize(device)
            prof.step()

    prof.export_chrome_trace(str(run_dir / "trace.json"))
    averages = prof.key_averages(group_by_input_shape=args.record_shapes)
    tables = []
    sorts = [("CPU self time", "self_cpu_time_total")]
    if device == "cuda":
        sorts.append(("CUDA self time", "self_device_time_total"))
    if args.profile_memory:
        sorts.append(("CPU self memory", "self_cpu_memory_usage"))
        if device == "cuda":
            sorts.append(("CUDA self memory", "self_device_memory_usage"))
    for title, sort_by in sorts:
        tables.append(
            f"\n=== {title} ===\n"
            + averages.table(
                sort_by=sort_by, row_limit=args.row_limit, max_name_column_width=80
            )
        )
    report = "\n".join(tables)
    (run_dir / "operators.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"出现非有限 loss；请检查 {run_dir} 中的 trace")
    print(f"最终合成 batch loss: {loss.item():.4f}（不代表真实训练质量）")
    print(f"用 https://ui.perfetto.dev 打开 {run_dir / 'trace.json'}")
    print("Profiler 有额外开销；性能对照请使用 benchmark.py")


if __name__ == "__main__":
    main()
