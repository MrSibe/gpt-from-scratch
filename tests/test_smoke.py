"""无需下载数据或使用 GPU：python -m unittest discover -s tests -v"""

import io
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]


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


class CLISmokeTests(unittest.TestCase):
    def run_python(self, *args, cwd):
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run(
            [sys.executable, *map(str, args)],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_import_train_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.run_python("-c", "import train", cwd=tmp)
            self.assertEqual(output, "")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_train_save_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "corpus.txt"
            data.write_text("abcd\n" * 100, encoding="utf-8")
            out = tmp / "out"
            # eval_interval 大于训练步数，验证最后一次更新后仍会保存。
            self.run_python(
                ROOT / "train.py",
                "--data",
                data,
                "--out-dir",
                out,
                "--device",
                "cpu",
                "--block-size",
                8,
                "--batch-size",
                2,
                "--max-iters",
                2,
                "--eval-interval",
                10,
                "--eval-iters",
                1,
                "--n-layer",
                1,
                "--n-head",
                2,
                "--n-embd",
                16,
                cwd=tmp,
            )
            checkpoint = torch.load(out / "ckpt.pt", weights_only=True)
            self.assertIsInstance(checkpoint["config"], dict)
            self.assertEqual(checkpoint["iter"], 2)
            self.assertEqual(checkpoint["config"]["vocab_size"], 5)
            self.assertTrue(torch.isfinite(torch.tensor(checkpoint["val_loss"])))
            self.assertEqual(
                {index: char for char, index in checkpoint["stoi"].items()},
                checkpoint["itos"],
            )
            generated = self.run_python(
                ROOT / "generate.py",
                "--ckpt",
                out / "ckpt.pt",
                "--device",
                "cpu",
                "--prompt",
                "ab",
                "--tokens",
                12,
                "--seed",
                1337,
                cwd=tmp,
            )
            # 去掉 print 自带的一个换行，保留模型可能生成的换行。
            text = generated[:-1]
            self.assertTrue(text.startswith("ab"))
            self.assertEqual(len(text), 14)
            self.assertLessEqual(set(text), set(checkpoint["stoi"]))


if __name__ == "__main__":
    unittest.main()
