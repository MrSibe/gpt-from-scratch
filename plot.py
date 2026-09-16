"""对比多个实验的 metrics.csv，输出一张对比图。

用法：
  uv run python plot.py                       # 对比 runs/ 下的全部实验
  uv run python plot.py runs/a runs/b         # 只对比指定目录
  uv run python plot.py --smooth 50 --out runs/latest.png

坐标轴标签使用英文，以避免不同系统缺少中文字体时出现方块字。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PANELS = ("val_loss_vs_tokens", "val_loss_vs_time")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="绘制实验对比图")
    p.add_argument(
        "runs",
        nargs="*",
        help="实验目录或 runs 父目录；省略时使用 --runs-dir",
    )
    p.add_argument("--runs-dir", default="runs", help="未指定目录时的搜索路径")
    p.add_argument("--out", default=None, help="输出图片路径，默认写入输出目录")
    p.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="逐步指标的滑动平均窗口，1 表示不平滑",
    )
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args(argv)


def find_run_dirs(paths):
    """把参数展开为包含 metrics.csv 的实验目录列表。"""
    found = []
    for path in paths:
        path = Path(path)
        if (path / "metrics.csv").is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(p.parent for p in path.glob("*/metrics.csv")))
        else:
            print(f"跳过 {path}: 未找到 metrics.csv", file=sys.stderr)
    return list(dict.fromkeys(found))


def load_run(run_dir):
    """读取一个实验的指标，返回 {列名: [值或 None]}。"""
    columns = {}
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, text in row.items():
                columns.setdefault(key, []).append(parse_value(text))
    return columns


def parse_value(text):
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def series(columns, x_key, y_key):
    """按 x 递增顺序取出 (x, y)，并跳过任一维为空的点。"""
    xs = columns.get(x_key, [])
    ys = columns.get(y_key, [])
    points = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    points.sort(key=lambda point: point[0])
    if not points:
        return [], []
    xs, ys = zip(*points)
    return list(xs), list(ys)


def smooth(values, window):
    if window <= 1 or len(values) <= window:
        return values
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def label_for(run_dir):
    """用目录名和参数量生成图例标签。"""
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return run_dir.name
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return run_dir.name
    params = config.get("parameters")
    if isinstance(params, (int, float)):
        return f"{run_dir.name} ({params / 1e6:.1f}M)"
    return run_dir.name


def time_axis(values):
    """按量级选择秒/分/时，返回缩放后的数值和单位。"""
    largest = max(values) if values else 0
    if largest >= 7200:
        return [value / 3600 for value in values], "h"
    if largest >= 300:
        return [value / 60 for value in values], "min"
    return values, "s"


def plot_runs(runs, smooth_window, out_path, dpi):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes = dict(zip(PANELS, axes.ravel()))
    loaded = {run_dir: load_run(run_dir) for run_dir in runs}
    # 所有实验共用时间单位，不能把一个实验的分钟与另一个的秒画在同一轴。
    all_times = [
        t
        for columns in loaded.values()
        for t in columns.get("wall_time_s", [])
        if t is not None
    ]
    _, unit = time_axis(all_times)
    divisor = {"s": 1, "min": 60, "h": 3600}[unit]

    for run_dir, columns in loaded.items():
        label = label_for(run_dir)

        # 训练 loss 逐步波动较大，用细线并做平滑；验证 loss 保留原始点。
        xs, ys = series(columns, "tokens_seen", "train_loss_step")
        if xs:
            axes["val_loss_vs_tokens"].plot(
                xs, smooth(ys, smooth_window), linewidth=0.7, alpha=0.4
            )
        xs, ys = series(columns, "tokens_seen", "val_loss")
        if xs:
            axes["val_loss_vs_tokens"].plot(
                xs, ys, marker="o", markersize=3, label=label
            )

        xs, ys = series(columns, "wall_time_s", "val_loss")
        if xs:
            axes["val_loss_vs_time"].plot(
                [x / divisor for x in xs], ys, marker="o", markersize=3, label=label
            )

    axes["val_loss_vs_tokens"].set(
        xlabel="training tokens seen", ylabel="loss (val solid, train faint)"
    )
    axes["val_loss_vs_tokens"].set_title("Val loss vs tokens")
    axes["val_loss_vs_time"].set(
        xlabel=f"cumulative wall time ({unit}, incl. eval/save)", ylabel="val loss"
    )
    axes["val_loss_vs_time"].set_title("Val loss vs wall time")
    for axis in axes.values():
        axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)

    figure.suptitle("mrsibe-llm experiment comparison")
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return out_path


def main(argv=None):
    args = parse_args(argv)
    paths = args.runs or [args.runs_dir]
    runs = find_run_dirs(paths)
    if not runs:
        print(f"没有找到任何实验记录（搜索路径: {paths}）", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else runs[0].parent / "comparison.png"
    plot_runs(runs, args.smooth, out_path, args.dpi)
    print(f"已绘制 {len(runs)} 个实验: {out_path}")
    for run_dir in runs:
        print(f"  - {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
