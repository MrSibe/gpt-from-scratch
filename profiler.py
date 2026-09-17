"""训练计算的 PyTorch Profiler：算子统计与 Chrome/Perfetto trace。

固定设备上的合成 batch，不含数据采样、H2D、评估或保存权重。
只从 train.py 取默认优化器超参数，不复用它的训练循环和观测逻辑。
Profiler 会扰动性能；吞吐对照请用 benchmark.py。
"""

import argparse
import functools
import json
import re
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from log import environment_info, git_info, unique_dir
from model import GPT, GPTConfig
from train import BETAS, LR, WEIGHT_DECAY

# --annotate-model 只包叶子模块：叶子自己的 kernel 时间就是这一行的 self time，
# 可以直接跨层相加。如果连容器模块（Block / CausalSelfAttention）一起包，
# 父行的 self time 会变成含子节点的累计值，既不能相加也会和子行重复计算。
ANNOTATED_TYPES = (nn.Linear, nn.LayerNorm, nn.Embedding)
BLOCK_PREFIX = re.compile(r"^blocks\.\d+\.")


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
    p.add_argument(
        "--with-stack",
        action="store_true",
        help="记录算子的 Python 调用栈；trace 里会同时出现 nn.Module 层级",
    )
    p.add_argument(
        "--annotate-model",
        action="store_true",
        help=(
            "把每个 Linear/LayerNorm/Embedding 的 forward 包一层 record_function，"
            "算子表按 GPT 结构（state_dict 路径）分组并追加结构汇总表；"
            "只改本进程的模块实例，不作为模型配置保存"
        ),
    )
    p.add_argument(
        "--with-flops",
        action="store_true",
        help=(
            "估算并统计 mm/bmm/addmm 的 FLOPs（自动开启 record-shapes）；"
            "SDPA 的注意力核心不计入"
        ),
    )
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


def annotate_model(model):
    """把叶子模块的 forward 包一层 record_function，名字沿用 state_dict 路径。

    只替换本进程内的实例属性：model.py 保持干净，train.py / benchmark.py 不受影响。
    record_function 在没有活跃 profiler 时几乎无开销，实测 8 层模型每步 < 0.1%。
    返回被标注的模块名，供结构汇总表区分标注行和真实算子行。
    """
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, ANNOTATED_TYPES):
            continue
        original = module.forward

        @functools.wraps(original)
        def forward(*args, _name=name, _original=original, **kwargs):
            with record_function(_name):
                return _original(*args, **kwargs)

        module.forward = forward
        names.append(name)
    return names


def module_summary(averages, annotated, key, title):
    """按结构汇总被标注叶子模块的 self time：blocks.<N>.attn.q_proj → attn.q_proj。

    只相加叶子模块，所以 self time 不重叠。但 record_function 只包住 forward：
    backward 的 kernel 由 autograd 引擎在区域外发起，不会被归入任何模块。
    因此这里的占比是“标注模块内部”的相对占比，不是整个 step 的占比；
    forward / backward / optimizer 的整体切分看上面的 train/forward 等区域行。
    """
    annotated = set(annotated)
    groups = {}
    for event in averages:
        if event.key not in annotated:
            continue
        group = BLOCK_PREFIX.sub("", event.key)
        groups[group] = groups.get(group, 0.0) + (getattr(event, key, 0.0) or 0.0)
    total = sum(groups.values())
    if total <= 0:
        return ""
    lines = [
        f"\n=== {title} ===",
        f"{'module group':<28}{'time':>12}{'share':>9}",
    ]
    lines += [
        f"{name:<28}{value / 1000:>10.3f}ms{value / total * 100:>8.1f}%"
        for name, value in sorted(groups.items(), key=lambda item: -item[1])
    ]
    lines.append(f"{'annotated total':<28}{total / 1000:>10.3f}ms{100.0:>8.1f}%")
    return "\n".join(lines)


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
    # 先标注再 compile，让 dynamo 直接追踪包好的 forward。
    annotated = annotate_model(raw_model) if args.annotate_model else []
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
    x = torch.randint(cfg.vocab_size, (args.batch_size, cfg.block_size), device=device)
    y = torch.randint(cfg.vocab_size, x.shape, device=device)

    def step():
        # 标记训练循环的语义边界：这些区域不属于任何 nn.Module，
        # 只能手写；模型结构相关的算子交给 --annotate-model。
        with record_function("train/zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        precision = (
            nullcontext()
            if args.dtype == "fp32"
            else torch.autocast(
                device_type=device,
                dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype],
            )
        )
        with record_function("train/forward"), precision:
            _, loss = model(x, y)
        with record_function("train/backward"):
            scaler.scale(loss).backward()
        with record_function("train/optimizer"):
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
        "profiled_steps": args.steps,
        "pre_warmup_steps": args.warmup,
        "annotated_modules": len(annotated),
        # --with-flops 会强制打开 record_shapes，但表格是否按形状分组只由
        # --record-shapes 决定，避免逐 op 行数被形状组合放大。
        "record_shapes_for_table": args.record_shapes,
        "environment": environment_info(device),
        "code": git_info(Path(__file__).resolve().parent),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Profiler 目录: {run_dir}", flush=True)
    print(f"预热 {args.warmup} 步（含首次编译），随后采集 {args.steps} 步", flush=True)
    if args.annotate_model:
        print(f"已按结构标注 {len(annotated)} 个叶子模块", flush=True)
    for _ in range(args.warmup):
        step()
    synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        # 外部 warmup 已经跑过热路径，这里直接采集全部 steps 步。
        # 不再使用 schedule(warmup=1)：那会让循环多跑一步并与 --warmup 混淆。
        record_shapes=args.record_shapes or args.with_flops,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_flops=args.with_flops,
    ) as prof:
        for index in range(args.steps):
            with record_function("train/step"):
                loss = step()
            # 仅在采集窗口结束前同步，不在每步读取 CUDA loss。
            if index == args.steps - 1:
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
    if args.annotate_model:
        key = "self_device_time_total" if device == "cuda" else "self_cpu_time_total"
        title = (
            f"Module self {'CUDA' if device == 'cuda' else 'CPU'} time"
            f"（{args.steps} 步窗口；只含标注模块 forward 内的 kernel）"
        )
        summary = module_summary(averages, annotated, key, title)
        if summary:
            report += summary
    if args.with_flops:
        # key_averages() 覆盖整个采集窗口，所以先除以步数还原单步。
        # 实测与解析式 2*N*tokens + 3x 的估算相差约 7% 以内，不重复计数。
        window_flops = sum(event.flops or 0 for event in averages)
        step_flops = window_flops / args.steps
        tokens = args.batch_size * cfg.block_size
        if window_flops <= 0:
            report += "\n=== FLOPs（估算）===\n没有找到 mm / bmm / addmm，无法估算\n"
        else:
            scale, unit = (1e9, "GFLOP") if step_flops >= 1e9 else (1e6, "MFLOP")
            report += (
                f"\n=== FLOPs（估算，{args.steps} 步平均）===\n"
                f"每步 {step_flops / scale:.1f} {unit}，每 token "
                f"{step_flops / tokens / 1e6:.2f} MFLOP\n"
                "只覆盖 mm / bmm / addmm；SDPA 的注意力核心不计入，"
                "所以 sdpa 路径会低估（manual 路径的 bmm 会计入）\n"
            )
    (run_dir / "operators.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"出现非有限 loss；请检查 {run_dir} 中的 trace")
    print(f"最终合成 batch loss: {loss.item():.4f}（不代表真实训练质量）")
    print(f"用 https://ui.perfetto.dev 打开 {run_dir / 'trace.json'}")
    print("Profiler 有额外开销；性能对照请使用 benchmark.py")


if __name__ == "__main__":
    main()
