import argparse
import math

import torch

from model import GPT, GPTConfig
from tokenizer import tokenizer_from_state


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="训练产物中的 best.pt")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50, help="0=不限制")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = p.parse_args(argv)
    if not args.prompt or args.tokens < 0 or args.top_k < 0:
        p.error("prompt 不能为空，tokens / top-k 必须非负")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        p.error("temperature 必须是有限正数")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA 不可用")

    # 先在 CPU 加载，避免 checkpoint 和模型各占一份 GPU 权重。
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    tokenizer = tokenizer_from_state(ckpt["tokenizer"])
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError("checkpoint 的 tokenizer 和模型词表不匹配")
    try:
        prompt_ids = tokenizer.encode(args.prompt)
    except KeyError as error:
        p.error(f"prompt 包含字符词表外的字符: {error}")
    if not prompt_ids:
        p.error("prompt 编码后不能为空")
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model.to(device).eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    out = model.generate(
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
        eos_id=tokenizer.eos_id,
    )

    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
