"""无需下载数据或使用 GPU：python -m unittest discover -s tests -v"""

import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from data import CharDataset
from model import GPT, GPTConfig
from runlog import METRIC_FIELDS, RunLogger

ROOT = Path(__file__).resolve().parents[1]

SMALL_TRAIN_ARGS = (
    "--device",
    "cpu",
    "--block-size",
    8,
    "--batch-size",
    2,
    "--eval-iters",
    1,
    "--n-layer",
    1,
    "--n-head",
    2,
    "--n-embd",
    16,
)


def run_python(*args, cwd):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"{args} 失败:\n{result.stdout}{result.stderr}")
    return result.stdout


def run_python_failing(*args, cwd):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError(f"{args} 本应失败，但正常退出了:\n{result.stdout}")
    return result.stdout + result.stderr


def read_metrics(run_dir):
    with (run_dir / "metrics.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


class ModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(1337)
        self.cfg = GPTConfig(
            vocab_size=8,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0,
        )
        self.model = GPT(self.cfg)
        self.x = torch.randint(self.cfg.vocab_size, (2, 8))
        self.y = torch.randint(self.cfg.vocab_size, (2, 8))

    def test_forward_backward(self):
        logits, loss = self.model(self.x, self.y)
        self.assertEqual(logits.shape, (2, 8, self.cfg.vocab_size))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_causal_prefix(self):
        self.model.eval()
        changed = self.x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % self.cfg.vocab_size
        with torch.no_grad():
            before, _ = self.model(self.x, self.y)
            after, _ = self.model(changed, self.y)
        torch.testing.assert_close(before[:, :4], after[:, :4])

    def test_overfit_fixed_batch(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.02)
        initial_loss = self.model(self.x, self.y)[1].item()
        for _ in range(80):
            _, loss = self.model(self.x, self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_loss = self.model(self.x, self.y)[1].item()
        self.assertLess(final_loss, initial_loss * 0.2)

    def test_checkpoint_round_trip(self):
        self.model.eval()
        buffer = io.BytesIO()
        torch.save(
            {"model": self.model.state_dict(), "config": asdict(self.cfg)},
            buffer,
        )
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = GPT(GPTConfig(**checkpoint["config"]))
        restored.load_state_dict(checkpoint["model"])
        restored.eval()
        with torch.no_grad():
            expected, _ = self.model(self.x, self.y)
            actual, _ = restored(self.x, self.y)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


class DatasetBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "corpus.txt"
        self.path.write_text("abcd\n" * 50, encoding="utf-8")
        self.dataset = CharDataset(
            path=self.path, block_size=8, val_ratio=0.2, device="cpu"
        )

    def test_generator_is_reproducible(self):
        first = self.dataset.get_batch(
            "train", 4, generator=torch.Generator().manual_seed(7)
        )
        second = self.dataset.get_batch(
            "train", 4, generator=torch.Generator().manual_seed(7)
        )
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right)

    def test_eval_generator_does_not_advance_global_rng(self):
        """评估使用独立生成器，不能影响之后的训练采样。"""
        torch.manual_seed(123)
        expected = torch.randint(1000, (3,))
        torch.manual_seed(123)
        for _ in range(2):
            self.dataset.get_batch("val", 2, generator=torch.Generator().manual_seed(0))
        actual = torch.randint(1000, (3,))
        torch.testing.assert_close(expected, actual)


class RunLoggerTests(unittest.TestCase):
    def test_avoids_overwriting_and_rejects_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "run"
            first = RunLogger(base)
            self.assertEqual(first.run_dir, base)
            first.log(step=1, train_loss_step=1.5, val_loss=None)
            first.close()

            second = RunLogger(base)
            self.addCleanup(second.close)
            self.assertEqual(second.run_dir, base.with_name("run-2"))
            with self.assertRaises(ValueError):
                second.log(step=1, unknown_metric=0)

            with first.metrics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), list(METRIC_FIELDS))
            self.assertEqual(rows[0]["step"], "1")
            self.assertEqual(rows[0]["val_loss"], "")

    def test_append_mode_keeps_single_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            first = RunLogger(run_dir)
            first.log(step=1, train_loss_step=1.0)
            first.close()

            resumed = RunLogger(run_dir, append=True)
            self.addCleanup(resumed.close)
            self.assertEqual(resumed.run_dir, run_dir)
            resumed.log(step=2, train_loss_step=0.5)

            with resumed.metrics_path.open(newline="") as handle:
                text = handle.read()
            self.assertEqual(text.count("step,tokens_seen"), 1)
            self.assertEqual(
                [row["step"] for row in csv.DictReader(io.StringIO(text))], ["1", "2"]
            )

    def test_append_mode_rejects_unknown_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "metrics.csv").write_text(
                "other,columns\n1,2\n", encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                RunLogger(run_dir, append=True)


class CLISmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.data = self.tmp_path / "corpus.txt"
        self.data.write_text("abcd\n" * 100, encoding="utf-8")

    def train(self, out_dir, *extra):
        return run_python(
            ROOT / "train.py",
            "--data",
            self.data,
            "--out-dir",
            out_dir,
            *SMALL_TRAIN_ARGS,
            *extra,
            cwd=self.tmp_path,
        )

    def test_import_train_has_no_side_effects(self):
        output = run_python("-c", "import train", cwd=self.tmp_path)
        self.assertEqual(output, "")
        self.assertEqual(list(self.tmp_path.iterdir()), [self.data])

    def test_train_save_generate(self):
        out = self.tmp_path / "out"
        # eval_interval 大于训练步数，验证最后一次更新后仍会评估和保存。
        self.train(out, "--max-iters", 2, "--eval-interval", 10)

        checkpoint = torch.load(out / "best.pt", weights_only=True)
        self.assertIsInstance(checkpoint["config"], dict)
        self.assertEqual(checkpoint["iter"], 2)
        self.assertEqual(checkpoint["config"]["vocab_size"], 5)
        self.assertTrue(torch.isfinite(torch.tensor(checkpoint["val_loss"])))
        self.assertEqual(
            {index: char for char, index in checkpoint["stoi"].items()},
            checkpoint["itos"],
        )

        generated = run_python(
            ROOT / "generate.py",
            "--ckpt",
            out / "best.pt",
            "--device",
            "cpu",
            "--prompt",
            "ab",
            "--tokens",
            12,
            "--seed",
            1337,
            cwd=self.tmp_path,
        )
        # 去掉 print 自带的一个换行，保留模型可能生成的换行。
        text = generated[:-1]
        self.assertTrue(text.startswith("ab"))
        self.assertEqual(len(text), 14)
        self.assertLessEqual(set(text), set(checkpoint["stoi"]))

    def test_run_records(self):
        out = self.tmp_path / "out"
        self.train(out, "--max-iters", 3, "--eval-interval", 2)

        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(
            config["parameters"],
            sum(
                parameter.numel()
                for parameter in GPT(GPTConfig(**config["model"])).parameters()
            ),
        )
        self.assertEqual(config["training"]["seed"], 1337)
        self.assertEqual(
            config["data"]["sha256"],
            hashlib.sha256(self.data.read_bytes()).hexdigest(),
        )
        self.assertEqual(config["environment"]["device"], "cpu")
        self.assertEqual(
            config["environment"]["cuda_available"], torch.cuda.is_available()
        )

        with (out / "metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["step"] for row in rows], ["1", "2", "3"])
        # 评估只发生在 step 2 和 step 3。
        self.assertEqual(rows[0]["val_loss"], "")
        self.assertNotEqual(rows[1]["val_loss"], "")
        self.assertNotEqual(rows[2]["val_loss"], "")
        self.assertAlmostEqual(float(rows[0]["tokens_seen"]), 16)
        self.assertTrue(all(float(row["grad_norm"]) >= 0 for row in rows))
        self.assertTrue(all(float(row["train_tokens_per_sec"]) > 0 for row in rows))

        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["steps"], 3)
        # metrics.csv 只保留 6 位小数，这里按同样的精度比较。
        self.assertAlmostEqual(
            summary["best_val_loss"], float(rows[2]["val_loss"]), places=6
        )

    def test_eval_interval_does_not_change_training_stream(self):
        """评估不消耗训练随机数流：改变评估频率不影响逐步训练指标。"""
        frequent = self.tmp_path / "frequent"
        sparse = self.tmp_path / "sparse"
        self.train(frequent, "--max-iters", 3, "--eval-interval", 1)
        self.train(sparse, "--max-iters", 3, "--eval-interval", 100)

        first = [row["train_loss_step"] for row in read_metrics(frequent)]
        second = [row["train_loss_step"] for row in read_metrics(sparse)]
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            self.assertAlmostEqual(float(left), float(right), places=9)

    def test_last_checkpoint_has_full_training_state(self):
        out = self.tmp_path / "out"
        self.train(out, "--max-iters", 2, "--eval-interval", 2)

        last = torch.load(out / "last.pt", weights_only=True)
        for key in ("model", "optimizer", "rng", "iter", "wall_time_s", "best_val"):
            self.assertIn(key, last)
        self.assertEqual(last["iter"], 2)
        self.assertEqual(last["training"]["batch_size"], 2)
        self.assertEqual(last["training"]["seed"], 1337)
        self.assertEqual(
            len(last["optimizer"]["state"]),
            len(list(GPT(GPTConfig(**last["config"])).named_parameters())),
        )
        self.assertIn("torch", last["rng"])
        self.assertIn("train_generator", last["rng"])
        self.assertGreater(last["wall_time_s"], 0)

        # best.pt 保持轻量，不包含优化器和随机数状态。
        best = torch.load(out / "best.pt", weights_only=True)
        self.assertNotIn("optimizer", best)
        self.assertNotIn("rng", best)

    def test_resume_reproduces_continuous_training(self):
        """先跑 2 步再续训 2 步，应与一次跑 4 步逐步一致。"""
        full = self.tmp_path / "full"
        partial = self.tmp_path / "partial"
        self.train(full, "--max-iters", 4, "--eval-interval", 2)
        self.train(partial, "--max-iters", 2, "--eval-interval", 2)
        self.train(
            partial,
            "--max-iters",
            4,
            "--eval-interval",
            2,
            "--resume",
            partial / "last.pt",
        )

        continuous = read_metrics(full)
        resumed = read_metrics(partial)
        # 续训是追加写入，不会重复表头或丢行。
        self.assertEqual([row["step"] for row in resumed], ["1", "2", "3", "4"])
        for column in ("train_loss_step", "val_loss", "grad_norm"):
            for expected, actual in zip(continuous, resumed):
                if expected[column] == "" or actual[column] == "":
                    self.assertEqual(expected[column], actual[column])
                    continue
                self.assertAlmostEqual(
                    float(expected[column]),
                    float(actual[column]),
                    places=9,
                    msg=f"{column} @ step {expected['step']}",
                )
        # 累计时间应当继续增长，而不是从 0 重新开始，也不能在续训节点回退。
        times = [float(record["wall_time_s"]) for record in resumed]
        self.assertEqual(times, sorted(times), "续训后 wall_time_s 必须单调不减")
        self.assertGreater(times[0], 0)
        resume_record = json.loads(
            (partial / "resume.json").read_text(encoding="utf-8")
        )
        self.assertEqual(resume_record["start_step"], 2)
        self.assertEqual(resume_record["target_steps"], 4)

    def test_resume_rejects_changed_data(self):
        out = self.tmp_path / "out"
        self.train(out, "--max-iters", 2, "--eval-interval", 2)
        # 字符集不变（词表一致），只有内容不同，应被数据摘要校验拦住。
        other = self.tmp_path / "other.txt"
        other.write_text(
            self.data.read_text(encoding="utf-8") + "abcd\n" * 10, encoding="utf-8"
        )
        message = run_python_failing(
            ROOT / "train.py",
            "--data",
            other,
            "--out-dir",
            out,
            *SMALL_TRAIN_ARGS,
            "--max-iters",
            4,
            "--eval-interval",
            2,
            "--resume",
            out / "last.pt",
            cwd=self.tmp_path,
        )
        self.assertIn("拒绝续训", message)

    def test_resume_requires_larger_max_iters(self):
        out = self.tmp_path / "out"
        self.train(out, "--max-iters", 2, "--eval-interval", 2)
        message = run_python_failing(
            ROOT / "train.py",
            "--data",
            self.data,
            "--out-dir",
            out,
            *SMALL_TRAIN_ARGS,
            "--max-iters",
            2,
            "--eval-interval",
            2,
            "--resume",
            out / "last.pt",
            cwd=self.tmp_path,
        )
        self.assertIn("无需续训", message)


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "需要安装 matplotlib")
class PlotScriptTests(unittest.TestCase):
    @staticmethod
    def write_run(run_dir, params, steps):
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps({"parameters": params}), encoding="utf-8"
        )
        with (run_dir / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, restval="")
            writer.writeheader()
            for step in range(1, steps + 1):
                writer.writerow(
                    {
                        "step": step,
                        "tokens_seen": step * 16,
                        "train_loss_step": 4.0 / step,
                        "val_loss": 4.1 / step if step % 2 == 0 else "",
                        "train_tokens_per_sec": 1000.0 + step,
                        "wall_time_s": step * 0.5,
                    }
                )

    def test_plot_multiple_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs_dir = tmp / "runs"
            self.write_run(runs_dir / "a", 1_000_000, 4)
            self.write_run(runs_dir / "b", 2_000_000, 6)
            out = tmp / "plots" / "comparison.png"
            output = run_python(
                ROOT / "plot.py",
                runs_dir,
                "--smooth",
                2,
                "--out",
                out,
                cwd=tmp,
            )
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)
            self.assertIn("已绘制 2 个实验", output)

    def test_plot_without_runs_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(ROOT / "plot.py"), str(Path(tmp) / "empty")],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("没有找到任何实验记录", result.stderr)


if __name__ == "__main__":
    unittest.main()
