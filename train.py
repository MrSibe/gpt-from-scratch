import argparse
import os
import time
from dataclasses import asdict

import torch

from data import DATA_PATH, CharDataset
from model import GPT, GPTConfig


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="训练字符级 GPT")
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--out-dir", default="out")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=positive_int, default=256)
    p.add_argument("--batch-size", type=positive_int, default=64)
    p.add_argument("--max-iters", type=positive_int, default=5000)
    p.add_argument("--eval-interval", type=positive_int, default=250)
    p.add_argument("--eval-iters", type=positive_int, default=200)
    p.add_argument("--n-layer", type=positive_int, default=6)
    p.add_argument("--n-head", type=positive_int, default=6)
    p.add_argument("--n-embd", type=positive_int, default=384)
    return p.parse_args(argv)


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, eval_iters):
    was_training = model.training
    model.eval()
    out = {}
    try:
        for split in ("train", "val"):
            losses = []
            for _ in range(eval_iters):
                x, y = dataset.get_batch(split, batch_size)
                _, loss = model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return out


def main(argv=None):
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    dataset = CharDataset(path=args.data, block_size=args.block_size, device=device)
    # 当前采样器需要每个 split 至少有 block_size + 2 个 token。
    if min(len(dataset.train_data), len(dataset.val_data)) < args.block_size + 2:
        raise ValueError("训练集和验证集都需要至少 block_size + 2 个 token")
    print(
        f"词表大小: {dataset.vocab_size}, 训练 token: {len(dataset.train_data)}, "
        f"验证 token: {len(dataset.val_data)}"
    )

    cfg = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f}M")

    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    # step 表示已完成的参数更新次数；最后一次更新后也会评估。
    for step in range(1, args.max_iters + 1):
        x, y = dataset.get_batch("train", args.batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_iters:
            losses = estimate_loss(model, dataset, args.batch_size, args.eval_iters)
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | lr {learning_rate:.2e} "
                f"| train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| elapsed {elapsed:.1f}s"
            )

            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "stoi": dataset.stoi,
                        "itos": dataset.itos,
                        "iter": step,
                        "val_loss": best_val,
                    },
                    os.path.join(args.out_dir, "ckpt.pt"),
                )
                print(f"  -> 保存 checkpoint (val {best_val:.4f})")

    print(f"训练完成，best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
