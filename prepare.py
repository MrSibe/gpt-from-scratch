"""准备 TinyStories V2 GPT-4：训练 byte-level BPE，写入 uint16 token 文件。

默认流式读取固定版本的官方 V2 train/valid；可成对指定本地文本或 HTTP(S) URL。
故事以 <|endoftext|> 分隔，仅用 train 拟合 tokenizer，不自动重新拆分。
"""

import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from data import BPE_DATA_PATH, TINYSTORIES_URL
from log import file_sha256
from tokenizer import EOS, BPETokenizer


def stories(source, limit=0):
    """最多保留一个故事的文本；limit=0 表示读完。远端源每次遍历重新打开。"""
    with (
        io.TextIOWrapper(urllib.request.urlopen(source, timeout=60), encoding="utf-8")
        if str(source).startswith(("https://", "http://"))
        else Path(source).open(encoding="utf-8")
    ) as handle:
        parts, count = [], 0
        for line in handle:
            chunks = line.split(EOS)
            for index, chunk in enumerate(chunks):
                parts.append(chunk)
                if index == len(chunks) - 1:
                    continue
                text = "".join(parts).strip()
                parts = []
                if text:
                    yield text
                    count += 1
                    if limit and count >= limit:
                        return
        text = "".join(parts).strip()
        if text:
            yield text


def train_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return BPETokenizer(tokenizer)


def write_split(source, path, tokenizer, limit):
    count, tokens = 0, 0
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for text in stories(source, limit):
            ids = tokenizer.encode(text) + [tokenizer.eos_id]
            np.asarray(ids, dtype="<u2").tofile(handle)
            # 摘要标识实际使用的规范化故事序列，而不是没读完的远端文件。
            digest.update((text + EOS).encode("utf-8"))
            count += 1
            tokens += len(ids)
            if count % 10000 == 0:
                print(f"{path.name}: {count} stories, {tokens} tokens", flush=True)
    if not count:
        raise ValueError(f"没有故事: {source}")
    return {
        "source": str(source),
        "stories": count,
        "tokens": tokens,
        "text_sha256": digest.hexdigest(),
        "sha256": file_sha256(path),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-file", help="本地文本或 URL；默认官方 TinyStoriesV2-GPT4-train.txt"
    )
    p.add_argument(
        "--val-file", help="本地文本或 URL；默认官方 TinyStoriesV2-GPT4-valid.txt"
    )
    p.add_argument(
        "--out-dir",
        default=Path(BPE_DATA_PATH),
        type=Path,
        help=f"新目录，不覆盖已有数据（默认 {BPE_DATA_PATH}）",
    )
    p.add_argument(
        "--vocab-size", type=int, default=8192, help="包含 EOS；实际词表可能更小"
    )
    p.add_argument(
        "--tokenizer-stories", type=int, default=20000, help="拟合 BPE 的训练故事数"
    )
    p.add_argument("--tokenizer", type=Path, help="复用已有 tokenizer.json，不重新拟合")
    p.add_argument(
        "--max-train-stories", type=int, default=0, help="取前 N 个完整故事，0=全部"
    )
    p.add_argument(
        "--max-val-stories", type=int, default=0, help="取前 N 个完整故事，0=全部"
    )
    args = p.parse_args(argv)
    if (args.train_file is None) != (args.val_file is None):
        p.error(
            "自定义数据时必须同时指定 --train-file 和 --val-file，避免混用不同版本的 split"
        )
    if args.train_file is None:
        args.train_file = f"{TINYSTORIES_URL}/TinyStoriesV2-GPT4-train.txt"
        args.val_file = f"{TINYSTORIES_URL}/TinyStoriesV2-GPT4-valid.txt"
    elif not args.train_file or not args.val_file:
        p.error("train-file 和 val-file 不能为空")
    if not 257 <= args.vocab_size <= 65536:
        p.error("vocab-size 必须在 [257, 65536]，包含 256 个字节和 EOS")
    if (
        args.tokenizer_stories < 1
        or min(args.max_train_stories, args.max_val_stories) < 0
    ):
        p.error("tokenizer-stories 必须为正，max-*-stories 必须非负")
    if args.train_file == args.val_file:
        p.error("train-file 和 val-file 不能相同")
    if (
        not args.train_file.startswith(("http://", "https://"))
        and not args.val_file.startswith(("http://", "https://"))
        and Path(args.train_file).resolve() == Path(args.val_file).resolve()
    ):
        p.error("train-file 和 val-file 不能指向同一文件")
    return args


def main(argv=None):
    args = parse_args(argv)
    # 失败时不写 meta.json；训练端不会接受未完成的目录。
    args.out_dir.mkdir(parents=True, exist_ok=False)
    fit_limit = min(
        args.tokenizer_stories, args.max_train_stories or args.tokenizer_stories
    )
    if args.tokenizer:
        tokenizer = BPETokenizer(Tokenizer.from_file(str(args.tokenizer)))
    else:
        print(f"用前 {fit_limit} 个训练故事拟合 BPE ...", flush=True)
        tokenizer = train_tokenizer(
            stories(args.train_file, fit_limit), args.vocab_size
        )
    if not 257 <= tokenizer.vocab_size <= 65536:
        raise ValueError("tokenizer 词表不能用 uint16 表示")
    if not set(pre_tokenizers.ByteLevel.alphabet()).issubset(
        tokenizer.tokenizer.get_vocab()
    ):
        raise ValueError("tokenizer 缺少完整 byte alphabet")
    tokenizer_path = args.out_dir / "tokenizer.json"
    tokenizer.tokenizer.save(str(tokenizer_path))
    splits = {}
    for split, source, limit in [
        ("train", args.train_file, args.max_train_stories),
        ("val", args.val_file, args.max_val_stories),
    ]:
        splits[split] = write_split(
            source, args.out_dir / f"{split}.bin", tokenizer, limit
        )
    metadata = {
        "format_version": 1,
        "dtype": "<u2",
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "tokenizers_version": tokenizers.__version__,
        "preparation": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "splits": splits,
        "packing": "concatenated stories with EOS; causal attention may cross story boundaries",
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"完成: {args.out_dir} | vocab={tokenizer.vocab_size} | {splits}")


if __name__ == "__main__":
    main()
