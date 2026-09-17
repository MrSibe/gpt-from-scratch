"""无需 pytest：uv run python -m unittest discover -s tests -v"""

import copy
import csv
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import benchmark
import generate
import prepare
import profiler
import train
from data import (
    BPE_DATA_PATH,
    DATA_PATH,
    TINYSTORIES_URL,
    BPEDataset,
    CharDataset,
    sample_batch,
)
from model import GPT, GPTConfig
from tokenizer import EOS, CharTokenizer, tokenizer_from_state


def tiny_model(**kwargs):
    return GPT(
        GPTConfig(
            vocab_size=300,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            **kwargs,
        )
    )


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train_file = self.root / "train.txt"
        self.val_file = self.root / "val.txt"
        self.train_file.write_text(
            (f"A little dog played in the park.\nIt was happy!{EOS}\n" * 30),
            encoding="utf-8",
        )
        self.val_file.write_text(
            (f"A kitten was sleeping. 你好，世界🌍!{EOS}\n" * 10), encoding="utf-8"
        )

    def prepare_data(self, name="bpe", extra=()):
        out = self.root / name
        with redirect_stdout(io.StringIO()):
            prepare.main(
                [
                    "--train-file",
                    str(self.train_file),
                    "--val-file",
                    str(self.val_file),
                    "--out-dir",
                    str(out),
                    "--vocab-size",
                    "300",
                    "--tokenizer-stories",
                    "2",
                    *extra,
                ]
            )
        return out

    def parse(self, extra=()):
        return train.parse_args(["--device", "cpu", "--dtype", "fp32", *extra])

    def test_cli_help_exits_successfully(self):
        for entry in (
            prepare.parse_args,
            train.parse_args,
            benchmark.parse_args,
            profiler.parse_args,
            generate.main,
        ):
            output = io.StringIO()
            with self.subTest(entry=entry.__module__), redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    entry(["--help"])
                self.assertEqual(raised.exception.code, 0)
                self.assertIn("usage:", output.getvalue())

    def test_default_v2_recipe_and_explicit_char_path(self):
        args = self.parse()
        self.assertEqual((args.tokenizer, args.data), ("bpe", BPE_DATA_PATH))
        self.assertEqual((args.n_layer, args.n_head, args.n_embd), (8, 8, 512))
        self.assertEqual(args.dropout, 0.1)
        self.assertEqual(
            args.batch_size * args.block_size * args.grad_accum_steps, 32768
        )
        self.assertEqual((args.max_iters, args.warmup_iters), (20000, 400))
        self.assertEqual((args.lr_schedule, args.grad_clip), ("cosine", 1.0))
        self.assertEqual(train.get_lr(args.max_iters, args), args.min_lr)
        self.assertEqual(self.parse(["--tokenizer", "char"]).data, DATA_PATH)
        self.assertEqual(self.parse(["--data", "custom-bpe"]).data, "custom-bpe")
        cfg = GPTConfig()
        for module in (benchmark, profiler):
            other = module.parse_args(["--device", "cpu", "--dtype", "fp32"])
            for name in (
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "dropout",
                "tie_embeddings",
            ):
                self.assertEqual(getattr(args, name), getattr(other, name))
                self.assertEqual(getattr(args, name), getattr(cfg, name))
            self.assertEqual(other.vocab_size, cfg.vocab_size)
            self.assertEqual(other.batch_size, args.batch_size)

    def test_prepare_defaults_are_pinned_v2_and_full(self):
        args = prepare.parse_args([])
        self.assertEqual(
            args.train_file, f"{TINYSTORIES_URL}/TinyStoriesV2-GPT4-train.txt"
        )
        self.assertEqual(
            args.val_file, f"{TINYSTORIES_URL}/TinyStoriesV2-GPT4-valid.txt"
        )
        self.assertEqual(args.out_dir, Path(BPE_DATA_PATH))
        self.assertEqual((args.max_train_stories, args.max_val_stories), (0, 0))
        self.assertEqual((args.vocab_size, args.tokenizer_stories), (8192, 20000))

    def test_default_sources_prepare_both_v2_splits(self):
        args = prepare.parse_args([])
        sources = {
            args.train_file: self.train_file.read_bytes(),
            args.val_file: self.val_file.read_bytes(),
        }
        out = self.root / "v2-smoke"
        with (
            patch(
                "prepare.urllib.request.urlopen",
                side_effect=lambda url, **kw: io.BytesIO(sources[url]),
            ) as download,
            redirect_stdout(io.StringIO()),
        ):
            prepare.main(
                [
                    "--out-dir",
                    str(out),
                    "--vocab-size",
                    "300",
                    "--tokenizer-stories",
                    "2",
                    "--max-train-stories",
                    "3",
                    "--max-val-stories",
                    "2",
                ]
            )
        dataset = BPEDataset(out, block_size=8)
        splits = dataset.manifest["splits"]
        self.assertEqual(splits["train"]["source"], args.train_file)
        self.assertEqual(splits["val"]["source"], args.val_file)
        self.assertEqual((splits["train"]["stories"], splits["val"]["stories"]), (3, 2))
        self.assertEqual(
            [call.args[0] for call in download.call_args_list],
            [args.train_file, args.train_file, args.val_file],
        )

    def test_reject_unpaired_or_empty_source_override(self):
        invalid = [
            ["--train-file", "custom.txt"],
            ["--val-file", "custom.txt"],
            ["--train-file", "", "--val-file", "custom.txt"],
            ["--train-file", "custom.txt", "--val-file", ""],
        ]
        for args in invalid:
            with (
                self.subTest(args=args),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                prepare.parse_args(args)

    def test_missing_bpe_data_has_prepare_hint(self):
        with self.assertRaisesRegex(FileNotFoundError, "prepare.py"):
            BPEDataset(self.root / "not-prepared")

    def test_story_boundaries_and_limits(self):
        self.train_file.write_text(f" a\nb {EOS}c{EOS}\n{EOS}tail", encoding="utf-8")
        self.assertEqual(list(prepare.stories(self.train_file)), ["a\nb", "c", "tail"])
        self.assertEqual(list(prepare.stories(self.train_file, 2)), ["a\nb", "c"])
        with patch(
            "prepare.urllib.request.urlopen",
            return_value=io.BytesIO(f"a{EOS}b".encode()),
        ):
            self.assertEqual(
                list(prepare.stories("https://example.org/stories.txt")), ["a", "b"]
            )

    def test_tokenizer_state_roundtrip(self):
        cases = [
            (CharTokenizer("ab"), "abba"),
            (prepare.train_tokenizer(["hello world"], 300), " 你好🌍\n\tCafé — hello!"),
        ]
        for tokenizer, text in cases:
            restored = tokenizer_from_state(tokenizer.state())
            self.assertEqual(restored.decode(restored.encode(text)), text)
            self.assertEqual(restored.encode(text), tokenizer.encode(text))
            if restored.eos_id is not None:
                self.assertEqual(restored.encode(EOS), [restored.eos_id])
                self.assertEqual(restored.decode([restored.eos_id]), "")

    def test_prepare_splits_eos_limits_and_train_only_fit(self):
        out = self.prepare_data(
            extra=("--max-train-stories", "3", "--max-val-stories", "2")
        )
        dataset = BPEDataset(out, block_size=8)
        expected = prepare.train_tokenizer(prepare.stories(self.train_file, 2), 300)
        self.assertEqual(
            dataset.tokenizer.tokenizer.get_vocab(), expected.tokenizer.get_vocab()
        )
        self.assertIsInstance(dataset.train_data, np.memmap)
        self.assertEqual(dataset.train_data.dtype, np.dtype("<u2"))
        self.assertEqual(int((dataset.train_data == dataset.tokenizer.eos_id).sum()), 3)
        self.assertEqual(int((dataset.val_data == dataset.tokenizer.eos_id).sum()), 2)
        expected_ids = []
        for text in prepare.stories(self.train_file, 3):
            expected_ids.extend(expected.encode(text) + [expected.eos_id])
        np.testing.assert_array_equal(dataset.train_data, expected_ids)
        x, y = dataset.get_batch("train", 2, torch.Generator().manual_seed(9))
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(x.shape, (2, 8))
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        with self.assertRaises(FileExistsError):
            self.prepare_data()

    def test_reuse_rejects_tokenizer_truncation_and_padding(self):
        for mode in ("truncation", "padding"):
            with self.subTest(mode=mode):
                tokenizer = prepare.train_tokenizer(
                    prepare.stories(self.train_file, 2), 300
                )
                if mode == "truncation":
                    tokenizer.tokenizer.enable_truncation(max_length=2)
                else:
                    tokenizer.tokenizer.enable_padding(
                        length=100, pad_id=tokenizer.eos_id, pad_token=EOS
                    )
                path = self.root / f"{mode}.json"
                tokenizer.tokenizer.save(str(path))
                with self.assertRaisesRegex(ValueError, "truncation/padding"):
                    self.prepare_data(mode, ("--tokenizer", str(path)))
                self.assertFalse((self.root / mode / "meta.json").exists())

    def test_reuse_tokenizer(self):
        first = self.prepare_data()
        second = self.prepare_data(
            "bpe2", ("--tokenizer", str(first / "tokenizer.json"))
        )
        self.assertEqual(
            (first / "tokenizer.json").read_bytes(),
            (second / "tokenizer.json").read_bytes(),
        )
        self.assertEqual(
            (first / "train.bin").read_bytes(), (second / "train.bin").read_bytes()
        )

    def test_reject_corrupted_data_or_tokenizer(self):
        out = self.prepare_data()
        with (out / "train.bin").open("r+b") as handle:
            handle.write(b"\xff\xff")
        with self.assertRaisesRegex(ValueError, "manifest"):
            BPEDataset(out, block_size=8)
        out = self.prepare_data("second")
        with (out / "tokenizer.json").open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "manifest"):
            BPEDataset(out, block_size=8)

    def test_reject_missing_custom_char_path_without_download(self):
        with patch("data.urllib.request.urlretrieve") as download:
            with self.assertRaises(FileNotFoundError):
                CharDataset(self.root / "missing.txt")
            download.assert_not_called()

    def test_last_sampling_window_and_shift(self):
        for data in (torch.arange(9), np.arange(9, dtype=np.uint16)):
            x, y = sample_batch(data, 8, 2, "cpu")
            torch.testing.assert_close(x[0], torch.arange(8))
            torch.testing.assert_close(y[0], torch.arange(1, 9))
        x, _ = sample_batch(
            torch.arange(10), 8, 100, "cpu", torch.Generator().manual_seed(0)
        )
        self.assertEqual(set(x[:, 0].tolist()), {0, 1})

    def test_constant_and_cosine_schedule(self):
        args = self.parse(
            [
                "--lr-schedule",
                "cosine",
                "--max-iters",
                "6",
                "--warmup-iters",
                "2",
                "--lr",
                "0.001",
                "--min-lr",
                "0.0001",
            ]
        )
        rates = [train.get_lr(i, args) for i in range(1, 7)]
        self.assertAlmostEqual(rates[0], 0.0005)
        self.assertAlmostEqual(rates[1], 0.001)
        self.assertAlmostEqual(rates[2], 0.001)
        self.assertAlmostEqual(rates[-1], 0.0001)
        self.assertTrue(all(a >= b for a, b in pairwise(rates[2:])))
        args.lr_schedule = "constant"
        self.assertEqual(train.get_lr(100, args), args.lr)
        self.assertEqual(train.get_lr(1, args), args.lr / 2)
        args = self.parse(["--max-iters", "100", "--warmup-ratio", ".02"])
        self.assertEqual(args.warmup_iters, 2)
        args = self.parse(["--max-iters", "1", "--lr-schedule", "cosine"])
        self.assertEqual(train.get_lr(1, args), args.lr)
        args = self.parse(["--max-iters", "2", "--lr-schedule", "cosine"])
        self.assertEqual(train.get_lr(1, args), args.lr)
        self.assertEqual(train.get_lr(2, args), args.min_lr)

    def test_explicit_warmup_iters_drops_unused_ratio(self):
        args = self.parse(["--warmup-iters", "0"])
        self.assertEqual(args.warmup_iters, 0)
        self.assertIsNone(args.warmup_ratio)
        args = self.parse(["--warmup-ratio", "0.1", "--max-iters", "100"])
        self.assertEqual(args.warmup_iters, 10)
        self.assertEqual(args.warmup_ratio, 0.1)

    def test_cli_validation(self):
        invalid = [
            ["--grad-accum-steps", "0"],
            ["--grad-clip", "-1"],
            ["--dropout", "1"],
            ["--lr", "nan"],
            ["--betas", ".9", "1"],
            ["--warmup-ratio", "1"],
            ["--warmup-iters", "5000", "--max-iters", "3000"],
            ["--lr-schedule", "cosine", "--min-lr", "1"],
            ["--warmup-iters", "1", "--warmup-ratio", ".1"],
        ]
        for args in invalid:
            with (
                self.subTest(args=args),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.parse(args)

    def test_accumulation_matches_large_batch_and_clips_once(self):
        for clip in (0.0, 0.01):
            with self.subTest(clip=clip):
                large = tiny_model(attention="manual").double()
                accumulated = copy.deepcopy(large)
                x = torch.randint(300, (4, 8))
                y = torch.randint(300, x.shape)
                results = []
                for model, batches, accumulation in [
                    (large, [(x, y)], 1),
                    (accumulated, [(x[:2], y[:2]), (x[2:], y[2:])], 2),
                ]:
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=0.001,
                        betas=(0.9, 0.95),
                        weight_decay=0.1,
                    )
                    scaler = torch.amp.GradScaler("cuda", enabled=False)
                    results.append(
                        train.train_step(
                            model,
                            optimizer,
                            scaler,
                            iter(batches),
                            accumulation,
                            grad_clip=clip,
                            measure_grad=True,
                        )
                    )
                    self.assertTrue(
                        all(state["step"] == 1 for state in optimizer.state.values())
                    )
                    norm = torch.nn.utils.get_total_norm(
                        [p.grad for p in model.parameters()]
                    )
                    if clip:
                        self.assertLessEqual(norm.item(), clip + 1e-8)
                        self.assertGreater(results[-1][1].item(), clip)
                torch.testing.assert_close(
                    results[0][0], results[1][0], rtol=1e-10, atol=1e-10
                )
                torch.testing.assert_close(
                    results[0][1], results[1][1], rtol=1e-10, atol=1e-10
                )
                for p1, p2 in zip(large.parameters(), accumulated.parameters()):
                    torch.testing.assert_close(p1, p2, rtol=1e-8, atol=1e-9)

    def test_manual_and_sdpa_agree_and_are_causal(self):
        manual = tiny_model(attention="manual").double().eval()
        sdpa = tiny_model(attention="sdpa").double().eval()
        sdpa.load_state_dict(manual.state_dict())
        x = torch.randint(300, (2, 8))
        y = torch.randint(300, x.shape)
        logits1, loss1 = manual(x, y)
        logits2, loss2 = sdpa(x, y)
        torch.testing.assert_close(logits1, logits2, atol=1e-9, rtol=1e-9)
        loss1.backward()
        loss2.backward()
        for p1, p2 in zip(manual.parameters(), sdpa.parameters()):
            torch.testing.assert_close(p1.grad, p2.grad, atol=1e-9, rtol=1e-9)
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 300
        for model in (manual, sdpa):
            original, _ = model(x, y)
            modified, _ = model(changed, y)
            torch.testing.assert_close(
                original[:, :4], modified[:, :4], atol=1e-9, rtol=1e-9
            )

    def test_weight_tying_shares_one_tensor_end_to_end(self):
        untied = tiny_model()
        tied = tiny_model(tie_embeddings=True)
        vocab, embd = tied.cfg.vocab_size, tied.cfg.n_embd
        self.assertFalse(untied.cfg.tie_embeddings)
        self.assertIsNot(untied.lm_head.weight, untied.wte.weight)
        self.assertIs(tied.lm_head.weight, tied.wte.weight)

        # named_parameters() 按张量身份去重，所以优化器不会重复更新共享权重。
        tied_params = list(tied.parameters())
        self.assertEqual(len(tied_params), len({id(p) for p in tied_params}))
        self.assertEqual(
            len(list(tied.named_parameters())),
            len(list(untied.named_parameters())) - 1,
        )
        self.assertEqual(
            sum(p.numel() for p in untied.parameters())
            - sum(p.numel() for p in tied.parameters()),
            vocab * embd,
        )

        # 反向只累积到共享的那一份权重上，两条路径的梯度合并在同一个 .grad。
        x = torch.randint(vocab, (2, tied.cfg.block_size))
        y = torch.randint(vocab, x.shape)
        _, loss = tied(x, y)
        loss.backward()
        self.assertIsNotNone(tied.wte.weight.grad)
        self.assertIs(tied.lm_head.weight.grad, tied.wte.weight.grad)

        # state_dict 两个键都在；保存/加载后共享关系与数值都不变。
        state = tied.state_dict()
        self.assertIn("wte.weight", state)
        self.assertIn("lm_head.weight", state)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        reloaded = tiny_model(tie_embeddings=True)
        reloaded.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertIs(reloaded.lm_head.weight, reloaded.wte.weight)
        torch.testing.assert_close(reloaded.wte.weight, tied.wte.weight)

        # checkpoint 里存的是 asdict(cfg)，generate 用 GPTConfig(**config) 重建，
        # 所以 tying 开关必须能原样往返，否则重建出来的模型不带共享。
        self.assertIs(GPTConfig(**asdict(tied.cfg)).tie_embeddings, True)
        self.assertIs(GPTConfig(**asdict(GPTConfig())).tie_embeddings, False)

    def test_eval_preserves_rng_and_training_mode(self):
        dataset = CharDataset(self.train_file, block_size=8)
        model = GPT(
            GPTConfig(
                vocab_size=dataset.vocab_size,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=16,
                dropout=0.2,
            )
        )
        state = torch.get_rng_state().clone()
        first = train.estimate_val_loss(model, dataset, 2, 3, 123)
        second = train.estimate_val_loss(model, dataset, 2, 3, 123)
        self.assertEqual(first, second)
        self.assertTrue(model.training)
        torch.testing.assert_close(state, torch.get_rng_state())
        model.eval()
        train.estimate_val_loss(model, dataset, 2, 1, 123)
        self.assertFalse(model.training)

    def test_generation_stops_at_eos(self):
        model = tiny_model().eval()
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.bias.zero_()
            model.lm_head.bias[0] = 100
        output = model.generate(
            torch.ones((2, 1), dtype=torch.long), 10, top_k=1, eos_id=0
        )
        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(output[:, -1].eq(0).all())
        with self.assertRaises(ValueError):
            model.generate(torch.ones((1, 1), dtype=torch.long), 1, temperature=0)

    def test_bpe_training_logging_checkpoint_and_generation(self):
        data = self.prepare_data()
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with redirect_stdout(io.StringIO()):
                train.main(
                    [
                        "--tokenizer",
                        "bpe",
                        "--data",
                        str(data),
                        "--device",
                        "cpu",
                        "--dtype",
                        "fp32",
                        "--n-layer",
                        "1",
                        "--n-head",
                        "2",
                        "--n-embd",
                        "16",
                        "--block-size",
                        "8",
                        "--batch-size",
                        "2",
                        "--grad-accum-steps",
                        "2",
                        "--max-iters",
                        "4",
                        "--eval-interval",
                        "2",
                        "--eval-iters",
                        "2",
                        "--eval-batch-size",
                        "2",
                        "--lr",
                        ".001",
                        "--lr-schedule",
                        "cosine",
                        "--warmup-iters",
                        "1",
                        "--min-lr",
                        ".0001",
                        "--grad-clip",
                        "1",
                        "--dropout",
                        "0",
                    ]
                )
            run = next(Path("runs").iterdir())
            with (run / "metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(r["tokens_seen"]) for r in rows], [32, 64, 96, 128])
            self.assertEqual(
                [float(r["lr"]) for r in rows], [0.001, 0.001, 0.00055, 0.0001]
            )
            self.assertEqual([int(r["optimizer_steps"]) for r in rows], [1, 2, 3, 4])
            self.assertTrue(all(r["skipped_update"] == "0" for r in rows))
            for r in rows:
                if r["val_loss"]:
                    self.assertAlmostEqual(
                        float(r["val_ppl"]), math.exp(float(r["val_loss"])), delta=0.001
                    )
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["tokens_seen"], 128)
            self.assertEqual(summary["optimizer_steps"], 4)
            checkpoint = torch.load(run / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["tokenizer"]["kind"], "bpe")
            (data / "tokenizer.json").unlink()  # 生成不依赖原数据目录。
            output = io.StringIO()
            with redirect_stdout(output):
                generate.main(
                    [
                        "--ckpt",
                        str(run / "best.pt"),
                        "--device",
                        "cpu",
                        "--prompt",
                        "你好",
                        "--tokens",
                        "2",
                        "--top-k",
                        "1",
                    ]
                )
            self.assertTrue(output.getvalue().startswith("你好"))
        finally:
            os.chdir(previous)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fp16_overflow_skips_update_and_does_not_advance_schedule(self):
        model = tiny_model().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", init_scale=128)
        x = torch.randint(300, (2, 8), device="cuda")
        y = torch.randint(300, x.shape, device="cuda")
        before = [p.detach().clone() for p in model.parameters()]
        hook = next(model.parameters()).register_hook(
            lambda g: torch.full_like(g, float("inf"))
        )
        args = self.parse(["--max-iters", "10", "--warmup-iters", "2"])
        updates = 0
        lr = train.get_lr(updates + 1, args)
        _, norm, updated = train.train_step(
            model, optimizer, scaler, iter([(x, y)] * 2), 2, "fp16", 1.0
        )
        updates += int(updated)
        hook.remove()
        self.assertFalse(updated)
        self.assertFalse(torch.isfinite(norm))
        self.assertEqual(scaler.get_scale(), 64)
        for p, old in zip(model.parameters(), before):
            torch.testing.assert_close(p, old, rtol=0, atol=0)
        self.assertEqual(train.get_lr(updates + 1, args), lr)
        _, norm, updated = train.train_step(
            model, optimizer, scaler, iter([(x, y)] * 2), 2, "fp16", 1.0
        )
        self.assertTrue(updated)
        self.assertTrue(torch.isfinite(norm))


if __name__ == "__main__":
    unittest.main()
