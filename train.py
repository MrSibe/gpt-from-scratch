import os
import time

import torch

from data import CharDataset
from model import GPT, GPTConfig

# 超参数
block_size = 256
batch_size = 64
max_iters = 5000
eval_interval = 250
eval_iters = 200
learning_rate = 3e-4
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
out_dir = "out"
seed = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed(seed)

# 数据
dataset = CharDataset(block_size=block_size, device=device)
print(
    f"词表大小: {dataset.vocab_size}, 训练 token: {len(dataset.train_data)}, 验证 token: {len(dataset.val_data)}"
)

# 模型
cfg = GPTConfig(
    vocab_size=dataset.vocab_size,
    block_size=block_size,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
)
model = GPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {n_params / 1e6:.2f}M")

# 优化器：全部参数丢进去，固定 lr
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


os.makedirs(out_dir, exist_ok=True)
best_val = float("inf")
t0 = time.time()

for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        dt = time.time() - t0
        print(
            f"step {it:5d} | lr {learning_rate:.2e} | train {losses['train']:.4f} "
            f"| val {losses['val']:.4f} | {dt:.1f}s"
        )

        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "stoi": dataset.stoi,
                    "itos": dataset.itos,
                    "iter": it,
                    "val_loss": best_val,
                },
                os.path.join(out_dir, "ckpt.pt"),
            )
            print(f"  -> 保存 checkpoint (val {best_val:.4f})")

    x, y = dataset.get_batch("train", batch_size)
    _, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"训练完成，best val loss = {best_val:.4f}")
