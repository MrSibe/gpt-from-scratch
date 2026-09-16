"""实验记录工具：运行目录、配置快照和指标 CSV。

只依赖标准库和 torch，供 train.py 和绘图/分析脚本复用。
"""

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

# metrics.csv 的列定义。一步一行；只在评估步出现的指标留空。
# peak_memory_mb 只在该步使用了 CUDA 时才有值；它是该步（含该步内发生的评估）
# 的峰值 allocated 显存，CPU 运行时留空。
METRIC_FIELDS = (
    "step",
    "tokens_seen",
    "train_loss_step",
    "train_loss_eval",
    "val_loss",
    "lr",
    "grad_norm",
    "step_time_s",
    "train_tokens_per_sec",
    "peak_memory_mb",
    "eval_time_s",
    "wall_time_s",
)


def file_sha256(path, chunk_size=1 << 20):
    """计算文件摘要，用于标识实验使用的数据版本。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(repo_dir):
    """返回提交号和是否有未提交改动；缺少 git 时返回 None 值。"""
    empty = {"git_commit": None, "git_dirty": None}

    def run(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_dir), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return empty
    status = run("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def environment_info(device):
    """记录软件版本和实际使用的设备。"""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = ".".join(
            str(value) for value in torch.cuda.get_device_capability(0)
        )
        info["cuda"] = torch.version.cuda
    return info


def unique_dir(base):
    """返回一个尚不存在的目录路径，避免覆盖已有实验记录。"""
    base = Path(base)
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = base.with_name(f"{base.name}-{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法为 {base} 找到可用的目录名")


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return round(value, 6)
    return value


class RunLogger:
    """管理一次实验的目录：写入 config.json 并逐行追加 metrics.csv。

    metrics.csv 每写入一行就 flush，训练中断也能保留已有记录。
    append=True 时复用已有目录继续追加（续训场景），会先校验表头一致。
    """

    def __init__(self, run_dir, append=False):
        if append:
            self.run_dir = Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_path = self.run_dir / "metrics.csv"
            empty = (
                not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
            )
            if not empty:
                self._check_header(self.metrics_path)
            self._handle = self.metrics_path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._handle, fieldnames=METRIC_FIELDS, restval=""
            )
            if empty:
                self._writer.writeheader()
        else:
            self.run_dir = unique_dir(run_dir)
            self.run_dir.mkdir(parents=True)
            self.metrics_path = self.run_dir / "metrics.csv"
            self._handle = self.metrics_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._handle, fieldnames=METRIC_FIELDS, restval=""
            )
            self._writer.writeheader()
        self._handle.flush()

    @staticmethod
    def _check_header(path):
        with path.open(newline="", encoding="utf-8") as handle:
            header = handle.readline().strip()
        if header.split(",") != list(METRIC_FIELDS):
            raise RuntimeError(
                f"{path} 的表头与当前指标定义不一致，无法追加写入；请改用新的实验目录"
            )

    def write_json(self, name, payload):
        path = self.run_dir / name
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def log(self, **row):
        unknown = set(row) - set(METRIC_FIELDS)
        if unknown:
            raise ValueError(f"未定义的指标字段: {sorted(unknown)}")
        self._writer.writerow({key: _format_value(value) for key, value in row.items()})
        self._handle.flush()

    def close(self):
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False
