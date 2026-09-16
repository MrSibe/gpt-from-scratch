import argparse

import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="out/ckpt.pt")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 新 checkpoint 只包含张量和基础 Python 类型，无需反序列化配置类。
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stoi, itos = ckpt["stoi"], ckpt["itos"]
    idx = torch.tensor(
        [[stoi[c] for c in args.prompt]], dtype=torch.long, device=device
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
    out = model.generate(
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
    )

    print("".join(itos[int(i)] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
