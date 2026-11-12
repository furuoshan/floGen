"""Synthetic regression tests for fixed-test SuperWing residual training."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import main as m
import train_3d_train_test as t


torch.set_num_threads(2)


class TinyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = m.FeatureNorm(2)
        self.input_normalizer = m.FeatureNorm(5, channel_dim=1)
        self.output_normalizer = m.FeatureNorm(3, channel_dim=1)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, inputs, conditions):
        delta = self.bias[None, :, None, None].expand(len(inputs), 3, 128, 256)
        return self.output_normalizer.inverse_delta(delta)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.airfoil = nn.Linear(1, 1).requires_grad_(False)
        self.lift = nn.Linear(1, 1).requires_grad_(False)
        self.residual = TinyResidual()


class ArrayData:
    def __init__(self, targets, priors=None):
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.priors = torch.zeros(len(targets), 2, 128, 256) if priors is None else torch.as_tensor(priors)
        self.geometry = torch.zeros(len(targets), 3, 128, 256)
        self.conditions = torch.tensor([[.75, 2.], [.85, 10.]])[:len(targets)]

    def iter_batches(self, rows, batch_size, device, shuffle=False, seed=42):
        rows = np.array(rows, copy=True)
        if shuffle: np.random.default_rng(seed).shuffle(rows)
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            yield {"row": torch.from_numpy(selected), "geometry": self.geometry[selected].to(device),
                   "prior": self.priors[selected].to(device), "conditions": self.conditions[selected].to(device),
                   "target": self.targets[selected].to(device)}


def prepared_fixture(root):
    cache = root / "cache"
    (cache / "split").mkdir(parents=True)
    geometry, _ = m.synthetic_geometry()
    arrays = {"geometry": np.stack((geometry, geometry)).astype(np.float32),
              "prior": np.zeros((10, 2, 128, 256), dtype=np.float32),
              "conditions": np.tile([.75, 2.], (10, 1)).astype(np.float32),
              "geometry_indices": np.r_[np.zeros(9, dtype=np.int64), 1],
              "groups": np.r_[np.full(9, 11, dtype=np.int64), 22],
              "sld": np.full((10, 128), .5), "reynolds": np.full(10, 2e7)}
    for name, values in arrays.items(): np.save(cache / (name + ".npy"), values)
    target_path = root / "targets.npy"
    np.save(target_path, np.zeros((10, 3, 128, 256), dtype=np.float32))
    for name, rows in (("train_rows", np.arange(9)), ("test_rows", np.array([9])),
                       ("excluded_rows", np.empty(0, dtype=np.int64))):
        np.save(cache / "split" / (name + ".npy"), rows)
    manifest = {"schema": "superwing_vlm_training_cache_v1", "status": "complete", "sample_count": 10,
                "geometry_count": 2, "sld_source": "pytornado_vlm", "sld_uses_cfd": False,
                "airfoil_checkpoint_sha256": "test", "targets_path": str(target_path),
                "source_files": {"targets": {"path": str(target_path), "sha256": t.digest_file(target_path),
                                             "size": target_path.stat().st_size}},
                "cache_file_hashes": {p.relative_to(cache).as_posix(): {"sha256": t.digest_file(p), "size": p.stat().st_size}
                                      for p in cache.rglob("*.npy")}}
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache


class WingTrainingTests(unittest.TestCase):
    def test_defaults_are_200_epochs_and_ten_epoch_snapshots(self):
        args = t.parser().parse_args(["--cache-root", "cache", "--airfoil-checkpoint", "foil.pt", "--output-dir", "out"])
        self.assertEqual(args.epochs, 200)
        self.assertEqual(args.checkpoint_every, 10)
        self.assertFalse(args.resume)

    def test_prepared_cache_partitions_whole_geometries_and_keeps_sources_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = prepared_fixture(root)
            before = {str(p): t.digest_file(p) for p in root.rglob("*") if p.is_file()}
            with patch.object(t, "ROOT", root):
                data = t.PreparedWingData(cache)
                try:
                    self.assertEqual(len(data.train), 9 * len(data.test))
                    self.assertFalse(data.targets.flags.writeable)
                    batch = next(data.iter_batches(data.test, 2, "cpu"))
                    self.assertEqual(tuple(batch["geometry"].shape), (1, 3, 128, 256))
                    self.assertEqual(batch["row"].tolist(), [9])
                finally: data.close()
                with self.assertRaisesRegex(ValueError, "outside the authorized project"):
                    t.project_path(root.parent / "outside.npy")
            self.assertEqual(before, {str(p): t.digest_file(p) for p in root.rglob("*") if p.is_file()})

    def test_cache_rejects_shape_leakage_and_cfd_sld(self):
        for case in ("leakage", "cfd"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve(); cache = prepared_fixture(root)
                if case == "leakage":
                    np.save(cache / "groups.npy", np.full(10, 11, dtype=np.int64))
                else:
                    path = cache / "manifest.json"
                    manifest = json.loads(path.read_text()); manifest["sld_uses_cfd"] = True
                    path.write_text(json.dumps(manifest))
                with patch.object(t, "ROOT", root), self.assertRaises(ValueError):
                    t.PreparedWingData(cache)

    def test_content_checksums_reject_changed_prior_targets_manifest_and_missing_hash(self):
        for case in ("prior", "target", "manifest", "missing_hash"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve(); cache = prepared_fixture(root)
                with patch.object(t, "ROOT", root):
                    data = t.PreparedWingData(cache)
                    try:
                        t.verify_source_files(data)
                        if case in ("prior", "target"):
                            path = cache / "prior.npy" if case == "prior" else root / "targets.npy"
                            values = np.load(path, mmap_mode="r+")
                            values[0, 0, 0, 0] = 1.
                            values.flush(); values._mmap.close()
                        elif case == "manifest":
                            path = cache / "manifest.json"
                            path.write_bytes(path.read_bytes() + b"\n")
                        else:
                            del data.manifest["cache_file_hashes"]["prior.npy"]
                        with self.assertRaises(ValueError):
                            t.verify_source_files(data)
                    finally:
                        data.close()

    def test_cached_forward_equals_full_physics_model_and_preserves_frozen_airfoil(self):
        m.seed_everything(5)
        model = m.PhysicsEmbeddedWingPredictor(m.ModelSpec()).eval()
        centers, vertices = m.synthetic_geometry()
        features = m.geometry_features(centers, vertices)
        batch = {name: torch.as_tensor(value.copy())[None] for name, value in features.items()}
        batch.update(parameters=torch.zeros(1, 38), conditions=torch.tensor([[.78, 4.]]),
                     reynolds=torch.tensor([2e7]), sld=torch.full((1, 128), .5))
        before = t.state_digest(model.airfoil)
        with torch.inference_mode():
            full, prior, _ = model(batch, return_details=True)
            cached = t.cached_forward(model, dict(batch, prior=prior))
        torch.testing.assert_close(full, cached, rtol=0, atol=0)
        self.assertEqual(t.state_digest(model.airfoil), before)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.airfoil.parameters()))
        del model; gc.collect()

    def test_train_only_normalization_and_heldout_metrics_never_change_buffers(self):
        model = TinyModel()
        targets = np.empty((2, 3, 128, 256), dtype=np.float32)
        contour = np.linspace(-1, 1, 256, dtype=np.float32)
        targets[0] = contour
        targets[1] = 1000 + contour
        data = ArrayData(targets)
        t.fit_train_normalizers(model, data, np.array([0]), 2, "cpu")
        np.testing.assert_allclose(model.residual.output_normalizer.minimum, [-1, -1, -1])
        np.testing.assert_allclose(model.residual.output_normalizer.scale, [2, 2, 2])
        before = t.normalizer_digest(model)
        metric = t.evaluate_epoch(model, data, np.array([1]), 2, "cpu")
        self.assertEqual(t.normalizer_digest(model), before)
        self.assertFalse(metric["test_used_for_parameter_updates"])
        self.assertEqual(metric["test_samples"], 1)
        self.assertGreater(metric["cp"]["nmae_range_percent"]["mean"], 10000)
        self.assertAlmostEqual(metric["cf_tau"]["physical_mae"] * 150,
                               metric["cf_tau"]["stored_mae"])
        self.assertAlmostEqual(metric["cf_z"]["physical_mae"] * 300,
                               metric["cf_z"]["stored_mae"])

    def test_zero_denominators_are_undefined_not_hidden_by_epsilon(self):
        model = TinyModel()
        data = ArrayData(np.zeros((1, 3, 128, 256), dtype=np.float32))
        metric = t.evaluate_epoch(model, data, np.array([0]), 1, "cpu")
        for name, _, _ in t.CHANNELS:
            self.assertIsNone(metric[name]["nmae_range_percent"]["mean"])
            self.assertIsNone(metric[name]["relative_l1_percent"]["mean"])
            self.assertEqual(metric[name]["nmae_range_percent"]["undefined_source_rows"], [0])
        json.dumps(metric, allow_nan=False)

    def test_snapshot_survives_latest_replacement_and_contains_honest_vlm_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model = TinyModel()
            optimizer = torch.optim.Adam(model.residual.parameters())
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            info = {"epochs_completed": 10, "training_steps": 1, "sld_source": "pytornado_vlm",
                    "sld_network_used": False, "sld_training_steps": 0}
            with patch.object(t, "ROOT", root):
                t.save_epoch(root, model, m.ModelSpec(), optimizer, scaler, info, [], 10)
                original = t.digest_file(root / "epoch_010.pt")
                with torch.no_grad(): model.residual.bias.add_(1)
                info = dict(info, epochs_completed=11)
                t.save_epoch(root, model, m.ModelSpec(), optimizer, scaler, info, [], 10)
            self.assertEqual(t.digest_file(root / "epoch_010.pt"), original)
            self.assertNotEqual(t.digest_file(root / "latest.pt"), original)
            payload = torch.load(root / "epoch_010.pt", weights_only=True)
            self.assertEqual(payload["training"]["sld_training_steps"], 0)
            self.assertFalse(payload["training"]["sld_network_used"])
            self.assertEqual(payload["format"], m.CHECKPOINT_FORMAT)
            self.assertTrue(any(name.startswith("airfoil.") for name in payload["state_dict"]))


if __name__ == "__main__":
    unittest.main()
