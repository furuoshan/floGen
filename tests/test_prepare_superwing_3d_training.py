"""Preparation orchestration on synthetic files; no production data or CFD runs."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_main import m
import prepare_superwing_3d_training as p


def native_fixture(root, counts=None):
    counts = [2] * 10 if counts is None else counts
    root.mkdir()
    configs = np.zeros((len(counts), 55), dtype=np.float64)
    configs[:, 0] = np.arange(len(counts)) * 17 + 101
    configs[:, 1:39] = np.arange(38)[None, :] / 100
    configs[:, 39:47] = .75 + np.arange(8)[None, :] * .001
    configs[:, 47:55] = 2 + np.arange(8)[None, :] * .1
    centers, vertices = m.synthetic_geometry()
    np.save(root / "geom0.npy", np.repeat(centers[None], len(counts), axis=0))
    np.save(root / "origingeom.npy", np.repeat(vertices[None], len(counts), axis=0))
    indices = []
    records = []
    for g, count in enumerate(counts):
        record = {"running_id": int(configs[g, 0])}
        for k in range(count):
            indices.append([g, k, configs[g, 47 + k], configs[g, 39 + k], 1.6, 2., .4, .02, .1, .4, .02, .1])
            record[f"{k}_running_info"] = {"working_conditions": {"re": 2e7, "temp": 300}}
        records.append(record)
    np.save(root / "index.npy", np.asarray(indices, dtype=np.float64))
    # These values are deliberately unrelated to the VLM mock output. SLD
    # preparation has no CFD-field argument and must never integrate them.
    np.save(root / "data.npy", np.full((sum(counts), 3, 128, 256), 7., dtype=np.float64))
    np.savetxt(root / "configs.dat", configs, header="native synthetic configuration", comments="")
    (root / "running_info.jsonl").write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    return root


def airfoil_fixture(path):
    spec = m.ModelSpec()
    # Interface fixture only: no claim that this randomly initialized 2-D model
    # has predictive accuracy. Production always loads the real 60-epoch model.
    torch.save({"format": m.CHECKPOINT_FORMAT, "stage": "airfoil", "spec": asdict(spec),
                "state_dict": m.AirfoilPredictor(spec).state_dict(),
                "training": {"training_steps": 1},
                "conventions": {"wing_multipliers": [1, 150, 300], "normalization": m.NORMALIZATION}}, path)
    return path


def geometry_only_vlm(centers, vertices, mach, aoa_deg, chord_panels, span_panels):
    assert centers.shape == (3, 128, 256)
    assert vertices.shape == (3, 129, 257)
    values = np.full((1, 128), aoa_deg * .1, dtype=np.float64)
    info = {"solver": "pytornado", "version": "test-only", "backend": "test-only",
            "method": "synthetic orchestration fixture", "uses_cfd": False,
            "chord_panels": chord_panels, "span_panels_per_half": span_panels,
            "total_panels": 2 * chord_panels * span_panels, "compressibility_correction": "none",
            "camber_surface": "test-only", "symmetry": "test-only", "coordinate_mapping": "test-only",
            "output_quantity": "dimensionless_sectional_cl_before_sweep_transform",
            "output_shape": [1, 128], "interpolation": "test-only", "limitations": "mock only"}
    return values, info


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(2)

    def test_exact_split_whole_test_geometries_and_fixed_training_remainder(self):
        groups = np.repeat(np.arange(12), [1] + [2] * 11)
        split, info = p.build_split(groups)
        self.assertEqual(len(split["train_rows"]), 18)
        self.assertEqual(len(split["test_rows"]), 2)
        self.assertEqual(len(split["excluded_rows"]), 3)
        self.assertEqual(info["geometry_overlap_count"], 0)
        self.assertFalse(np.intersect1d(groups[split["train_rows"]], groups[split["test_rows"]]).size)
        self.assertFalse(np.intersect1d(split["excluded_rows"], split["test_rows"]).size)
        combined = np.sort(np.concatenate(list(split.values())))
        np.testing.assert_array_equal(combined, np.arange(len(groups)))
        again, same_info = p.build_split(groups)
        for name in split:
            np.testing.assert_array_equal(split[name], again[name])
        self.assertEqual(info, same_info)

    def test_unachievable_whole_geometry_ratio_is_not_silently_relaxed(self):
        with self.assertRaisesRegex(ValueError, "Cannot make exact"):
            p.build_split(np.repeat(np.arange(2), 10))

    def test_project_boundary_rejects_parent_and_existing_symlink_escape(self):
        project = self.root / "project"
        project.mkdir()
        with self.assertRaisesRegex(ValueError, "inside project"):
            p.inside(self.root / "outside.npy", project, "input")

    def test_preparation_resumes_vlm_and_prior_without_touching_source(self):
        source = native_fixture(self.root / "source")
        weight = airfoil_fixture(self.root / "airfoil.pt")
        cache = self.root / "cache"
        before = {name: p.sha256_file(source / name) for name in p.SOURCE_NAMES}
        original = p.sha256_file(weight)
        with patch.object(p, "estimate_spanwise_lift", side_effect=geometry_only_vlm) as vlm, \
             patch.object(p.m, "sectional_lift", side_effect=AssertionError("Used CFD for SLD")), \
             contextlib.redirect_stdout(io.StringIO()):
            first = p.prepare(source, cache, weight, project_root=self.root, max_cases=3,
                              prior_batch_size=2, section_batch_size=256)
            self.assertEqual(first["status"], "preparing")
            self.assertEqual(first["cases_completed"], 3)
            self.assertEqual(vlm.call_count, 3)
            first_prior = np.load(cache / "prior.npy", mmap_mode="r")[:3].copy()
            second = p.prepare(source, cache, weight, project_root=self.root,
                               prior_batch_size=2, section_batch_size=256)
            self.assertEqual(second["status"], "complete")
            self.assertEqual(second["cases_completed"], 20)
            self.assertEqual(vlm.call_count, 20)
            np.testing.assert_array_equal(np.load(cache / "prior.npy")[:3], first_prior)
            third = p.prepare(source, cache, weight, project_root=self.root)
            self.assertEqual(third["status"], "complete")
            self.assertEqual(vlm.call_count, 20)
        self.assertEqual(before, {name: p.sha256_file(source / name) for name in p.SOURCE_NAMES})
        self.assertEqual(original, p.sha256_file(weight))
        self.assertEqual(second["airfoil_checkpoint_sha256"], original)
        self.assertFalse(second["sld_uses_cfd"])
        self.assertEqual(second["targets_path"], str(source / "data.npy"))
        self.assertEqual(np.load(cache / "geometry.npy").shape, (10, 3, 128, 256))
        self.assertEqual(np.load(cache / "prior.npy").shape, (20, 2, 128, 256))
        self.assertEqual(np.load(cache / "sld.npy").dtype, np.float64)
        np.testing.assert_array_equal(np.load(cache / "reynolds.npy"), np.full(20, 2e7))
        self.assertEqual(second["split"]["train_samples"], 18)
        self.assertEqual(second["split"]["test_samples"], 2)
        self.assertIn("prior.npy", second["cache_file_hashes"])
        self.assertEqual(second["cache_file_hashes"]["split/test_rows.npy"]["sha256"],
                         p.sha256_file(cache / "split/test_rows.npy"))

    def test_changed_checkpoint_rejects_resume_without_damaging_complete_manifest(self):
        source = native_fixture(self.root / "source")
        weight = airfoil_fixture(self.root / "airfoil.pt")
        cache = self.root / "cache"
        with patch.object(p, "estimate_spanwise_lift", side_effect=geometry_only_vlm), \
             contextlib.redirect_stdout(io.StringIO()):
            p.prepare(source, cache, weight, project_root=self.root)
        before = (cache / "manifest.json").read_bytes()
        replacement = airfoil_fixture(self.root / "replacement.pt")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                p.prepare(source, cache, replacement, project_root=self.root)
        self.assertEqual(before, (cache / "manifest.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
