"""字符级小数据 / 预分词 BPE memmap，使用相同的随机窗口采样接口。"""

import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from log import file_sha256
from tokenizer import BPETokenizer, CharTokenizer

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"
BPE_DATA_PATH = "data/tinystories-v2"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{TINYSTORIES_REVISION}"


def sample_batch(data, block_size, batch_size, device, generator=None):
    # 共 len(data)-block_size 个合法起点，包含最后一个窗口。
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    positions = ix[:, None] + torch.arange(block_size + 1)
    if isinstance(data, np.ndarray):
        # memmap 只读取当前 batch；磁盘 uint16 -> embedding 所需的 int64。
        batch = torch.from_numpy(data[positions.numpy()].astype(np.int64))
    else:
        batch = data[positions]
    return batch[:, :-1].to(device), batch[:, 1:].to(device)


class CharDataset:
    def __init__(self, path=DATA_PATH, block_size=256, val_ratio=0.1, device="cpu"):
        path = Path(path)
        if not path.exists():
            if path.resolve() != Path(DATA_PATH).resolve():
                raise FileNotFoundError(f"数据文件不存在: {path}")
            print(f"下载 Tiny Shakespeare 到 {path} ...")
            urllib.request.urlretrieve(DATA_URL, path)
        text = path.read_text(encoding="utf-8")
        self.tokenizer = CharTokenizer(sorted(set(text)))
        self.vocab_size = self.tokenizer.vocab_size
        self.path, self.val_ratio = str(path), val_ratio
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(len(data) * (1 - val_ratio))
        self.train_data, self.val_data = data[:n], data[n:]
        self.block_size, self.device = block_size, device
        if min(len(self.train_data), len(self.val_data)) <= block_size:
            raise ValueError("训练集和验证集都需要至少 block_size + 1 个 token")

    def get_batch(self, split, batch_size, generator=None):
        if split not in ("train", "val"):
            raise ValueError(f"未知 split: {split}")
        data = self.train_data if split == "train" else self.val_data
        return sample_batch(data, self.block_size, batch_size, self.device, generator)

    def metadata(self):
        return {
            "kind": "char",
            "path": str(Path(self.path).resolve()),
            "sha256": file_sha256(self.path),
            "val_ratio": self.val_ratio,
            "vocab_size": self.vocab_size,
            "train_tokens": len(self.train_data),
            "val_tokens": len(self.val_data),
        }


class BPEDataset:
    def __init__(self, path=BPE_DATA_PATH, block_size=256, device="cpu"):
        self.path = Path(path)
        manifest_path = self.path / "meta.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"缺少 {manifest_path}；请先运行 prepare.py --out-dir {self.path}，"
                "或用 --data 指定已准备的数据目录"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest["format_version"] != 1 or self.manifest["dtype"] != "<u2":
            raise ValueError("不支持的 BPE 数据格式，请使用 prepare.py")
        tokenizer_path = self.path / "tokenizer.json"
        if file_sha256(tokenizer_path) != self.manifest["tokenizer_sha256"]:
            raise ValueError("tokenizer.json 与数据 manifest 不匹配")
        self.tokenizer = BPETokenizer(Tokenizer.from_file(str(tokenizer_path)))
        self.vocab_size = self.tokenizer.vocab_size
        if (self.vocab_size, self.tokenizer.eos_id) != (
            self.manifest["vocab_size"],
            self.manifest["eos_id"],
        ):
            raise ValueError("tokenizer 词表/EOS 与数据不匹配")
        for split in ("train", "val"):
            binary = self.path / f"{split}.bin"
            info = self.manifest["splits"][split]
            if (
                info["tokens"] <= block_size
                or binary.stat().st_size != info["tokens"] * 2
            ):
                raise ValueError(f"{split} 长度不匹配或不足 block_size + 1")
            if file_sha256(binary) != info["sha256"]:
                raise ValueError(f"{split}.bin 与数据 manifest 不匹配")
        self.train_data = np.memmap(self.path / "train.bin", dtype="<u2", mode="r")
        self.val_data = np.memmap(self.path / "val.bin", dtype="<u2", mode="r")
        self.block_size, self.device = block_size, device

    def get_batch(self, split, batch_size, generator=None):
        if split not in ("train", "val"):
            raise ValueError(f"未知 split: {split}")
        data = self.train_data if split == "train" else self.val_data
        return sample_batch(data, self.block_size, batch_size, self.device, generator)

    def metadata(self):
        return {
            "kind": "bpe",
            "path": str(self.path.resolve()),
            "sha256": file_sha256(self.path / "meta.json"),
            "vocab_size": self.vocab_size,
            "manifest": self.manifest,
            "train_tokens": len(self.train_data),
            "val_tokens": len(self.val_data),
        }
