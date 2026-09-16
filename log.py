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

# 训练指标一步一行，在评估点批量写入。
# val_loss / grad_norm / wall_time_s 只在评估步有值；性能指标由 benchmark.py 记录。
METRIC_FIELDS = (
    "step",
    "optimizer_steps",
    "skipped_update",
    "tokens_seen",
    "train_loss_step",
    "val_loss",
    "val_ppl",
    "lr",
    "grad_norm",
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
    return value


class RunLogger:
    """为每次实验创建独立目录，不覆盖或追加旧实验。"""

    def __init__(self, run_dir):
        self.run_dir = unique_dir(run_dir)
        self.run_dir.mkdir(parents=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self._handle = self.metrics_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=METRIC_FIELDS, restval=""
        )
        self._writer.writeheader()
        self._handle.flush()

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
