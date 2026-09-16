import os
import urllib.request

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"


class CharDataset:
    """字符级数据集：编码/解码 + 随机采样 batch"""

    def __init__(self, path=DATA_PATH, block_size=256, val_ratio=0.1, device="cpu"):
        if not os.path.exists(path):
            print(f"下载数据集到 {path} ...")
            urllib.request.urlretrieve(DATA_URL, path)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.path = path
        self.val_ratio = val_ratio

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * (1 - val_ratio))
        self.train_data = data[:n]
        self.val_data = data[n:]

        self.block_size = block_size
        self.device = device

    def get_batch(self, split: str, batch_size: int, generator=None):
        """返回 (x, y)，y 是 x 右移一位的目标。

        传入 generator 时不消耗全局随机数流，训练与评估因此互不影响。
        """
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(
            len(data) - self.block_size - 1, (batch_size,), generator=generator
        )
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)
