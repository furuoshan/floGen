"""External geometry/conditions/SLD contract; synthetic forward passes only.

No model is trained here. The export fixture declares checkpoint metadata solely
to exercise the file interface and does not establish aerodynamic accuracy.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from test_main import m, wing_fixture
import foilpara as f


def foilpara_fixture(root, index_columns=6):
    """Select exactly one native geometry/case and save its 41-column layout."""
    native = wing_fixture(root.parent / (root.name + "_native"))
    for name in ("foiloc", "foilgeo", "foilsld"):
        (root / name).mkdir(parents=True)
    index = np.load(native / "index.npy", allow_pickle=False)[:1, :index_columns]
    np.save(root / "foiloc" / "index.npy", index)
    np.save(root / "foiloc" / "reynolds.npy", np.full(len(index), 2e7))
    configs = np.loadtxt(native / "configs.dat", skiprows=1, ndmin=2)
    single = np.r_[configs[0, :39], index[0, 3], index[0, 2]][None, :]
    np.savetxt(root / "foilgeo" / "configs.dat", single,
               header="original_id geometry_parameters_1_to_38 mach aoa_deg", comments="")
    for name in ("geom0.npy", "origingeom.npy"):
        np.save(root / "foilgeo" / name, np.load(native / name, allow_pickle=False)[:1])
    sld = np.linspace(.35, .65, 128)[None, :]
    np.save(root / "foilsld" / "sectional_cl.npy", sld)
    np.save(root / "foilsld" / "index.npy", index[:, :6])
    return root


def source_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ForbiddenSLD(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("External foilsld must bypass the learned SLD model")


class RecordingFoil(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, geometry, conditions):
        self.seen.append(conditions.detach().clone())
        # Make Cp depend on effective CL so the full physical transformation can
        # be checked, rather than merely checking a dictionary key is present.
        output = torch.zeros(len(geometry), 2, 401, device=geometry.device)
        output[:, 0] = conditions[:, 1, None]
        return output


class FoilParaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def assert_invalid(self, root):
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            dataset = f.FoilParaData(root)
            dataset[0]

    def test_native_geometry_and_external_sld_preserve_row_order(self):
        root = foilpara_fixture(self.root / "input")
        before = source_hashes(root)
        dataset = f.FoilParaData(root)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(np.loadtxt(root / "foilgeo" / "configs.dat", skiprows=1, ndmin=2).shape, (1, 41))
        expected = np.load(root / "foilsld" / "sectional_cl.npy")
        index = np.load(root / "foiloc" / "index.npy")
        for row in range(len(dataset)):
            sample = dataset[row]
            self.assertEqual(tuple(sample["geometry"].shape), (3, 128, 256))
            np.testing.assert_allclose(sample["sld"].numpy(), expected[row], rtol=1e-6)
            np.testing.assert_allclose(sample["conditions"].numpy(), index[row, [3, 2]], rtol=1e-6)
            self.assertEqual(float(sample["reynolds"]), 2e7)
            self.assertNotIn("target", sample)
        self.assertEqual(before, source_hashes(root))

    def test_foilpara_rejects_twelve_column_conditions(self):
        root = foilpara_fixture(self.root / "input", index_columns=12)
        self.assert_invalid(root)

    def test_single_case_rejects_old_55_column_configs(self):
        root = foilpara_fixture(self.root / "input")
        path = root / "foilgeo" / "configs.dat"
        single = np.loadtxt(path, skiprows=1, ndmin=2)[0]
        legacy = np.r_[single[:39], np.full(8, single[39]), np.full(8, single[40])]
        np.savetxt(path, legacy[None], header="legacy 55-column layout", comments="")
        self.assert_invalid(root)

    def test_single_case_cfd_preserves_twelve_columns_with_41_column_config(self):
        root = foilpara_fixture(self.root / "input")
        native = self.root / "input_native"
        (native / "configs.dat").write_bytes((root / "foilgeo" / "configs.dat").read_bytes())
        for name in ("index.npy", "geom0.npy", "origingeom.npy", "data.npy"):
            path = native / name
            np.save(path, np.load(path)[:1])
        dataset = m.SuperWingData(native, target="field", require_reynolds=True, reynolds=2e7)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.configs.shape, (1, 41))
        self.assertEqual(dataset.index.shape, (1, 12))
        np.testing.assert_allclose(dataset[0]["conditions"], [.78, 2.], rtol=1e-6)
        self.assertEqual(tuple(dataset[0]["target"].shape), (3, 128, 256))

    def test_single_case_rejects_second_geometry_or_condition(self):
        for case in ("second_geometry", "second_condition", "nonzero_geometry_id", "nonzero_condition_id"):
            with self.subTest(case=case):
                root = foilpara_fixture(self.root / case)
                if case == "second_geometry":
                    path = root / "foilgeo" / "configs.dat"
                    configs = np.loadtxt(path, skiprows=1, ndmin=2)
                    second = configs.copy(); second[0, 0] += 1
                    np.savetxt(path, np.concatenate((configs, second)), header="two geometries", comments="")
                    for name in ("geom0.npy", "origingeom.npy"):
                        path = root / "foilgeo" / name
                        values = np.load(path)
                        np.save(path, np.concatenate((values, values)))
                elif case == "second_condition":
                    index = np.load(root / "foiloc" / "index.npy")
                    second = index.copy(); second[0, 1] = 1
                    index = np.concatenate((index, second))
                    np.save(root / "foiloc" / "index.npy", index)
                    np.save(root / "foilsld" / "index.npy", index)
                    np.save(root / "foiloc" / "reynolds.npy", np.full(2, 2e7))
                    path = root / "foilsld" / "sectional_cl.npy"
                    values = np.load(path)
                    np.save(path, np.concatenate((values, values)))
                else:
                    index = np.load(root / "foiloc" / "index.npy")
                    index[0, 0 if case == "nonzero_geometry_id" else 1] = 1
                    np.save(root / "foiloc" / "index.npy", index)
                    np.save(root / "foilsld" / "index.npy", index)
                self.assert_invalid(root)

    def test_missing_required_files_are_rejected_without_database_fallback(self):
        names = (
            "foiloc/index.npy", "foilgeo/configs.dat", "foilgeo/geom0.npy",
            "foilgeo/origingeom.npy", "foilsld/sectional_cl.npy", "foilsld/index.npy",
        )
        for number, name in enumerate(names):
            with self.subTest(file=name):
                root = foilpara_fixture(self.root / str(number))
                (root / name).unlink()
                self.assert_invalid(root)

    def test_sld_binding_rejects_changed_case_identity_or_operating_conditions(self):
        for case in ("different_condition", "different_mach"):
            with self.subTest(case=case):
                root = foilpara_fixture(self.root / case)
                path = root / "foilsld" / "index.npy"
                index = np.load(path)
                if case == "different_condition":
                    index[0, 1] = 1
                else:
                    index[0, 3] += .01
                np.save(path, index)
                self.assert_invalid(root)

    def test_external_sld_rejects_wrong_shape_and_nonfinite_values(self):
        for case in ("short_span", "missing_sample", "nan", "inf"):
            with self.subTest(case=case):
                root = foilpara_fixture(self.root / case)
                path = root / "foilsld" / "sectional_cl.npy"
                sld = np.load(path)
                if case == "short_span":
                    sld = sld[:, :127]
                elif case == "missing_sample":
                    sld = sld[:0]
                else:
                    sld[0, 0] = np.nan if case == "nan" else np.inf
                np.save(path, sld)
                self.assert_invalid(root)

    def test_reynolds_is_required_and_must_be_positive_finite_per_sample(self):
        for case in ("missing", "short", "zero", "negative", "nan"):
            with self.subTest(case=case):
                root = foilpara_fixture(self.root / case)
                path = root / "foiloc" / "reynolds.npy"
                if case == "missing":
                    path.unlink()
                else:
                    values = np.full(2 if case == "short" else 1, 2e7)
                    if case not in ("short",):
                        values[0] = {"zero": 0., "negative": -1., "nan": np.nan}[case]
                    np.save(path, values)
                self.assert_invalid(root)

    def test_external_sld_changes_effective_cl_and_bypasses_lift_network(self):
        root = foilpara_fixture(self.root / "input")
        dataset = f.FoilParaData(root)
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
        foil = RecordingFoil()
        model = m.PhysicsEmbeddedWingPredictor(m.ModelSpec(), foil, ForbiddenSLD()).eval()
        with torch.inference_mode():
            first, used = model.prior(batch)
            expected_conditions = batch["sld"] / batch["cos_sweep"].reshape(1, 1).square()
            observed = torch.cat(foil.seen)[:, 1].reshape(1, 128)
            self.assertTrue(torch.allclose(observed, expected_conditions, rtol=1e-6))
            self.assertTrue(torch.equal(used, batch["sld"]))
            self.assertTrue(torch.allclose(first[:, 0], batch["sld"][:, :, None].expand(-1, -1, 256), rtol=1e-6))
            altered = dict(batch, sld=batch["sld"] + .2)
            second, _ = model.prior(altered)
            self.assertTrue(torch.allclose(second[:, 0] - first[:, 0], torch.full_like(first[:, 0], .2), atol=1e-6))

    def test_model_rejects_malformed_external_sld_before_using_fallback(self):
        root = foilpara_fixture(self.root / "input")
        batch = next(iter(torch.utils.data.DataLoader(f.FoilParaData(root), batch_size=1)))
        model = m.PhysicsEmbeddedWingPredictor(m.ModelSpec(), RecordingFoil(), ForbiddenSLD()).eval()
        for value in (torch.zeros(1, 127), torch.full((1, 128), float("nan"))):
            with self.subTest(shape=tuple(value.shape)):
                with self.assertRaises(ValueError):
                    model.prior(dict(batch, sld=value))

    def test_export_is_forward_only_finite_and_preserves_all_weights_and_sources(self):
        root = foilpara_fixture(self.root / "input")
        # Invalid sentinels make accidental dependence on any source target
        # field fail, even if a source reader bypasses the np.load guard below.
        for folder in (root, root / "foiloc", root / "foilgeo", root / "foilsld"):
            (folder / "data.npy").write_bytes(b"CFD target arrays must not be read")
        before_sources = source_hashes(root)
        spec = m.ModelSpec()
        model = m.PhysicsEmbeddedWingPredictor(spec).eval()
        before_weights = state_hash(model)
        info = {"format": m.CHECKPOINT_FORMAT, "stage": "wing", "training": {
            "training_steps": 1, "airfoil_training_steps": 1, "sld_training_steps": 1,
        }}
        original_load = np.load

        def guarded_load(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).name == "data.npy":
                raise AssertionError("Prediction attempted to read CFD data.npy")
            return original_load(path, *args, **kwargs)

        output = self.root / "output"
        output.mkdir()
        format_content = "# Output format\n\nPreserve this document during prediction.\n".encode("utf-8")
        (output / "format.md").write_bytes(format_content)
        with patch.object(np, "load", side_effect=guarded_load), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("Prediction created Adam")), \
             patch.object(torch.optim, "SGD", side_effect=AssertionError("Prediction created SGD")), \
             patch.object(m.FeatureNorm, "fit", side_effect=AssertionError("Prediction fitted normalization")), \
             patch.object(m.FeatureNorm, "fit_bounds", side_effect=AssertionError("Prediction fitted normalization")), \
             patch.object(model.lift, "forward", side_effect=AssertionError("Prediction ignored external SLD")):
            f.predict_foilpara(model, spec, info, root, output, batch_size=2)
        values = np.load(output / "data.npy", allow_pickle=False)
        self.assertEqual(values.shape, (1, 3, 128, 256))
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(np.load(output / "index.npy"), np.load(root / "foiloc" / "index.npy")[:, :6])
        metadata = json.loads((output / "prediction_info.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["stored_multipliers"], [1, 150, 300])
        self.assertEqual(metadata["index_layout"], "geometry_conditions_v1")
        self.assertEqual(metadata["sld_source"], "external_foilsld_sectional_cl")
        self.assertEqual(metadata["training_steps_performed"], 0)
        self.assertTrue(metadata["model_state_unchanged"])
        self.assertTrue(metadata["normalization_unchanged"])
        for name in ("configs.dat", "geom0.npy", "origingeom.npy"):
            self.assertEqual((root / "foilgeo" / name).read_bytes(), (output / name).read_bytes())
        self.assertFalse((output / "data.npy.partial").exists())
        self.assertEqual(before_weights, state_hash(model))
        self.assertEqual(before_sources, source_hashes(root))
        self.assertEqual((output / "format.md").read_bytes(), format_content)

    def test_output_directory_allows_only_a_regular_format_document(self):
        self.assertTrue(m.prediction_directory_available(self.root / "new-output"))
        empty = self.root / "empty"
        empty.mkdir()
        self.assertTrue(m.prediction_directory_available(empty))
        documented = self.root / "documented"
        documented.mkdir()
        (documented / "format.md").write_text("# Format\n", encoding="utf-8")
        self.assertTrue(m.prediction_directory_available(documented))
        (documented / "data.npy").write_bytes(b"existing result")
        self.assertFalse(m.prediction_directory_available(documented))
        for name in ("other-file", "other-directory", "format-directory", "file-output"):
            path = self.root / name
            if name == "file-output":
                path.write_text("not a directory", encoding="utf-8")
            else:
                path.mkdir()
                if name == "other-file":
                    (path / "notes.md").write_text("existing content", encoding="utf-8")
                else:
                    (path / ("format.md" if name == "format-directory" else "nested")).mkdir()
            with self.subTest(case=name):
                self.assertFalse(m.prediction_directory_available(path))
                with self.assertRaises(ValueError):
                    f._destination(path, self.root / "input")

    def test_export_requires_trained_checkpoint_metadata(self):
        root = foilpara_fixture(self.root / "input")
        spec = m.ModelSpec()
        model = m.PhysicsEmbeddedWingPredictor(spec).eval()
        for number, info in enumerate(({}, {"stage": "wing", "training": {"training_steps": 0}})):
            with self.subTest(info=info):
                output = self.root / ("rejected" + str(number))
                with self.assertRaises(ValueError):
                    f.predict_foilpara(model, spec, info, root, output)
                self.assertFalse((output / "data.npy").exists())

    def test_cli_defaults_and_forward_dispatch_are_separate_from_training(self):
        parser = m.build_parser()
        args = parser.parse_args(["predict-foilpara", "--checkpoint", "wing.pt"])
        self.assertEqual(args.input_root, Path("D:/data/foilpara"))
        self.assertEqual(args.output_root, Path("D:/data/foilperf/network"))
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(f.FOILPARA_ROOT, args.input_root)
        self.assertEqual(parser.parse_args(["test-3d"]).device, "cpu")
        report = {"optimizer_steps": 0, "model_state_unchanged": True}
        with patch.object(f, "forward_test", return_value=report) as forward, \
             patch.object(m, "fit_model", side_effect=AssertionError("Test CLI entered training")), \
             contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(m.main(["test-3d", "--device", "cpu"]), 0)
        forward.assert_called_once_with("cpu")
        self.assertEqual(json.loads(captured.getvalue()), report)

    def test_forward_diagnostic_reports_zero_steps_and_unchanged_state(self):
        old_threads = torch.get_num_threads()
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("Diagnostic created Adam")), \
             patch.object(torch.optim, "SGD", side_effect=AssertionError("Diagnostic created SGD")), \
             patch.object(f.m.FeatureNorm, "fit", side_effect=AssertionError("Diagnostic fitted normalization")), \
             patch.object(f.m.FeatureNorm, "fit_bounds", side_effect=AssertionError("Diagnostic fitted normalization")), \
             patch.object(f.m, "fit_model", side_effect=AssertionError("Diagnostic entered training")):
            report = f.forward_test("cpu")
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertTrue(report["model_state_unchanged"])
        self.assertTrue(report["normalization_unchanged"])
        self.assertTrue(report["finite"])
        self.assertTrue(report["synthetic_only"])
        self.assertFalse(report["production_output_written"])
        self.assertEqual(report["prediction_kind"], "diagnostic_untrained")
        self.assertEqual(report["wing_shape"], [1, 3, 128, 256])
        self.assertEqual(report["index_shape"], [1, 6])
        self.assertEqual(report["sld_source"], "external_foilsld_sectional_cl")
        self.assertEqual(torch.get_num_threads(), old_threads)


if __name__ == "__main__":
    unittest.main()
