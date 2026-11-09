"""Single-pass 2-D training and safe latest-weight retention, using tiny synthetic arrays."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import train_2d_once as t


def fixture(root):
    for subset, protected in (("VT-BUF", 60), ("VT-DD", 50), ("VT-GU", 40)):
        folder = root / "VT" / subset
        folder.mkdir(parents=True)
        groups = np.repeat([10, 20, 30, 40, 50, 60], 2)
        index = np.zeros((12, 13), dtype=np.float64)
        index[:, 0] = np.repeat(np.arange(6), 2)
        index[:, 1] = np.tile([0, 1], 6)
        index[:, 11] = groups
        index[:, 5] = .72
        index[:, 9] = .5
        if subset == "VT-DD":
            index[:, 3] = .72
            index[:, 6] = .5
            index[:, 9] = 2.
        x = np.linspace(0, 2*np.pi, 401)
        data = np.empty((12, 3, 401), dtype=np.float64)
        data[:, 0] = .05*np.sin(x)
        data[:, 1] = -.5*np.sin(x) + np.arange(12)[:, None]*.001
        data[:, 2] = 2 + .2*np.cos(x)
        np.save(folder / "index.npy", index)
        np.save(folder / "data.npy", data)
        np.savetxt(folder / "traindataindex.txt", np.flatnonzero(groups != protected), fmt="%d")
    return root


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class OnePassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def test_global_running_id_split_protects_every_official_test_family(self):
        dataset = t.JointSuperFoil(fixture(self.base / "source"))
        self.assertEqual(set(dataset.protected_test_ids), {40, 50, 60})
        train, val = dataset.split(.34, 42)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(val), 6)
        self.assertFalse(set(dataset.groups[train]) & set(dataset.groups[val]))
        self.assertEqual(set(dataset.groups[np.r_[train, val]]), {10, 20, 30})
        for row in (0, 11, 12, 23, 24, 35):
            self.assertEqual(dataset[row]["row"], row)

    def test_one_epoch_keeps_last_partial_batch_and_only_latest_weight(self):
        source = fixture(self.base / "source")
        before = hashes(source)
        output = self.base / "weights"
        output.mkdir()
        (output / "older.pth").write_bytes(b"obsolete diagnostic weight")
        (output / "notes.txt").write_text("keep this non-weight file", encoding="utf-8")
        checkpoint = t.run(source, output, device="cpu", batch_size=5,
                           validation_fraction=.34, checkpoint_every=100)
        _, _, payload = t.m.load_checkpoint(checkpoint, "airfoil")
        info = payload["training"]
        self.assertEqual(info["epochs_completed"], 1)
        self.assertEqual(info["training_steps"], 3)  # 12 records: batches 5,5,2.
        self.assertEqual(info["samples_seen"], 12)
        self.assertTrue(info["every_selected_training_record_seen_exactly_once"])
        self.assertEqual(info["learning_rate"], 1e-5)
        self.assertEqual(info["checkpoint_selection"], "latest_optimizer_step_not_best_validation")
        self.assertEqual([p.name for p in output.iterdir() if p.suffix in t.WEIGHT_SUFFIXES], ["latest.pt"])
        self.assertEqual((output / "notes.txt").read_text(encoding="utf-8"), "keep this non-weight file")
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
        self.assertEqual(hashes(source), before)

    def test_invalid_new_checkpoint_preserves_previous_weight_and_source(self):
        output = self.base / "weights"
        output.mkdir()
        (output / "latest.pt").write_bytes(b"prior successfully published weight")
        (output / "old.pth").write_bytes(b"previous generation")
        model = t.m.AirfoilPredictor(t.m.ModelSpec())
        with torch.no_grad():
            next(model.parameters()).flatten()[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite checkpoint"):
            t.save_latest(output, model, t.m.ModelSpec(), {"training_steps": 1}, self.base / "source")
        self.assertEqual((output / "latest.pt").read_bytes(), b"prior successfully published weight")
        self.assertTrue((output / "old.pth").is_file())
        with self.assertRaisesRegex(ValueError, "input data"):
            t.save_latest(self.base / "source" / "weights", model, t.m.ModelSpec(),
                          {"training_steps": 1}, self.base / "source")


if __name__ == "__main__":
    unittest.main()
