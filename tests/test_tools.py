"""日志、绘图、生成与测速的回归测试；不下载数据。"""

import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import benchmark
import generate
import plot
from log import RunLogger
from model import GPT, GPTConfig
from tokenizer import CharTokenizer


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root)

    def checkpoint(self):
        cfg = GPTConfig(
            vocab_size=2, block_size=8, n_layer=1, n_head=2, n_embd=8, dropout=0
        )
        path = self.root / "best.pt"
        torch.save(
            {
                "model": GPT(cfg).state_dict(),
                "config": asdict(cfg),
                "tokenizer": CharTokenizer("ab").state(),
            },
            path,
        )
        return path

    def benchmark_args(self):
        return [
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--n-layer",
            "1",
            "--n-head",
            "2",
            "--n-embd",
            "8",
            "--vocab-size",
            "16",
            "--batch-size",
            "2",
            "--block-size",
            "8",
            "--warmup",
            "1",
            "--steps",
            "2",
            "--repeats",
            "2",
        ]

    def test_csv_preserves_small_lr_and_float_precision(self):
        with RunLogger(self.root / "log") as logger:
            logger.log(
                step=1, lr=1e-9, train_loss_step=1.123456789012345, skipped_update=False
            )
            path = logger.metrics_path
        with path.open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(float(row["lr"]), 1e-9)
        self.assertEqual(float(row["train_loss_step"]), 1.123456789012345)
        self.assertEqual(row["skipped_update"], "0")
        self.assertEqual(row["val_loss"], "")

    def test_smooth_also_handles_short_curves(self):
        self.assertEqual(plot.smooth([1, 3], 2), [1, 2])
        self.assertEqual(plot.smooth([1, 3], 10), [1, 2])
        self.assertEqual(plot.smooth([1, 3, 5], 2), [1, 2, 4])
        self.assertEqual(plot.smooth([1, 3], 1), [1, 3])
        self.assertEqual(plot.smooth([], 10), [])

    def test_plot_matches_train_and_val_colors_across_panels(self):
        runs = [self.root / "a", self.root / "b"]
        for run in runs:
            with RunLogger(run) as logger:
                for step in (1, 2):
                    logger.log(
                        step=step,
                        tokens_seen=step * 16,
                        train_loss_step=2 / step,
                        val_loss=3 / step,
                        wall_time_s=step,
                    )
        output = self.root / "curves.png"
        with patch("plot.plt.close") as close:
            plot.plot_runs(runs, 10, output, 40)
            figure = close.call_args.args[0]
        try:
            self.assertTrue(output.is_file())
            first = [line.get_color() for line in figure.axes[0].lines]
            second = [line.get_color() for line in figure.axes[1].lines]
            self.assertEqual(first, ["C0", "C0", "C1", "C1"])
            self.assertEqual(second, ["C0", "C1"])
        finally:
            plot.plt.close(figure)

    def test_generate_rejects_invalid_temperature_before_loading(self):
        for temperature in ("0", "-1", "nan", "inf"):
            with (
                self.subTest(temperature=temperature),
                patch("generate.torch.load") as load,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    generate.main(
                        [
                            "--ckpt",
                            "missing.pt",
                            "--temperature",
                            temperature,
                            "--device",
                            "cpu",
                        ]
                    )
                self.assertEqual(raised.exception.code, 2)
                load.assert_not_called()

    def test_generate_rejects_unavailable_cuda_before_loading(self):
        with (
            patch("generate.torch.cuda.is_available", return_value=False),
            patch("generate.torch.load") as load,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                generate.main(["--ckpt", "missing.pt", "--device", "cuda"])
            self.assertEqual(raised.exception.code, 2)
            load.assert_not_called()

    def test_generate_unknown_char_has_cli_error(self):
        checkpoint = self.checkpoint()
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(SystemExit) as raised:
            generate.main(
                ["--ckpt", str(checkpoint), "--prompt", "z", "--device", "cpu"]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("词表外", output.getvalue())

    def test_generate_loads_checkpoint_on_cpu(self):
        checkpoint = self.checkpoint()
        load = torch.load
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            output = io.StringIO()
            with (
                self.subTest(device=device),
                patch("generate.torch.load", wraps=load) as mocked,
                redirect_stdout(output),
            ):
                generate.main(
                    [
                        "--ckpt",
                        str(checkpoint),
                        "--prompt",
                        "ab",
                        "--tokens",
                        "0",
                        "--device",
                        device,
                    ]
                )
                mocked.assert_called_once_with(
                    str(checkpoint), map_location="cpu", weights_only=True
                )
            self.assertEqual(output.getvalue().strip(), "ab")

    def test_benchmark_counts_successful_updates(self):
        with redirect_stdout(io.StringIO()):
            benchmark.main(self.benchmark_args())
        path = next(Path("runs").glob("*/benchmark.csv"))
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(int(row["optimizer_steps"]) == 2 for row in rows))

    def test_benchmark_rejects_skipped_update_in_timed_window(self):
        # CPU 模拟 scaler：预热成功，测量的第一次 step 跳过，之后恢复。
        class SkippingScaler:
            def __init__(self):
                self.calls = 0

            def scale(self, loss):
                return loss

            def step(self, optimizer):
                self.calls += 1
                if self.calls != 2:
                    optimizer.step()

            def update(self):
                pass

        with (
            patch("benchmark.torch.amp.GradScaler", return_value=SkippingScaler()),
            self.assertRaisesRegex(RuntimeError, "1/2"),
        ):
            benchmark.main(self.benchmark_args())
        self.assertFalse(Path("runs").exists())


if __name__ == "__main__":
    unittest.main()
