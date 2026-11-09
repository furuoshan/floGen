"""One-case extraction/export from a read-only native SuperWing database.

All inputs are temporary synthetic fixtures. No optimizer, training, bootstrap,
production database write, or fabricated network output is used in these tests.
"""
from dataclasses import is_dataclass, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_main import m, wing_fixture
from foilpara import FoilParaData
from superwing_reader import SuperWingReader
from superwing_io import export_case


def tree_state(root):
    root = Path(root)
    if not root.exists():
        return None
    return {
        str(path.relative_to(root)): (
            "directory" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in root.rglob("*")
    }


class SuperWingIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = wing_fixture(self.root / "native")
        # Different Re for each original ID / original condition slot catches
        # lookups that accidentally use the renumbered condition ID zero.
        records = []
        for original, base in ((101, 2e7), (509, 2.2e7)):
            records.append({
                "running_id": original,
                "0_running_info": {"working_conditions": {"re": base}},
                "7_running_info": {"working_conditions": {"re": base + 1e6}},
            })
        (self.source / "running_info.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records), encoding="utf-8")

    def read_case(self, row=1, sld=None, sld_mode="cfd-reference"):
        with SuperWingReader(self.source) as reader:
            return reader.read(row, sld=sld, sld_mode=sld_mode)

    def test_selected_original_slots_become_exactly_one_41_column_case(self):
        source_configs = np.loadtxt(self.source / "configs.dat", skiprows=1, ndmin=2)
        source_index = np.load(self.source / "index.npy")
        with SuperWingReader(self.source) as reader:
            self.assertEqual(len(reader), 4)
            for row, g, original, reynolds in ((1, 0, 101, 2.1e7), (3, 1, 509, 2.3e7)):
                with self.subTest(row=row):
                    case = reader.read(row, sld_mode="cfd-reference")
                    self.assertTrue(is_dataclass(case))
                    self.assertEqual(case.configs.shape, (1, 41))
                    self.assertEqual(case.index.shape, (1, 6))
                    self.assertEqual(case.cfd_index.shape, (1, 12))
                    self.assertEqual(case.centers.shape, (1, 3, 128, 256))
                    self.assertEqual(case.vertices.shape, (1, 3, 129, 257))
                    self.assertEqual(case.field.shape, (1, 3, 128, 256))
                    self.assertEqual(case.field.dtype, np.float64)
                    self.assertEqual(case.centers.dtype, np.float64)
                    self.assertEqual(case.vertices.dtype, np.float64)
                    self.assertEqual(case.reynolds.shape, (1,))
                    self.assertEqual(case.sectional_cl.shape, (1, 128))
                    np.testing.assert_array_equal(case.index[0, :2], [0, 0])
                    np.testing.assert_array_equal(case.cfd_index[0, :2], [0, 0])
                    np.testing.assert_array_equal(case.configs[0, :39], source_configs[g, :39])
                    self.assertEqual(case.configs[0, 39], source_configs[g, 46])
                    self.assertEqual(case.configs[0, 40], source_configs[g, 54])
                    np.testing.assert_array_equal(case.index[0, 2:], source_index[row, 2:6])
                    np.testing.assert_array_equal(case.cfd_index[0, 2:], source_index[row, 2:])
                    self.assertEqual(float(case.reynolds[0]), reynolds)
                    self.assertEqual(case.metadata["source_row"], row)
                    self.assertEqual(case.metadata["source_geometry_index"], g)
                    self.assertEqual(case.metadata["source_condition_id"], 7)
                    self.assertEqual(case.metadata["original_id"], original)

    def test_explicit_reference_sld_is_cfd_integral_and_marked_as_reference(self):
        case = self.read_case(3)
        features = m.geometry_features(case.centers[0], case.vertices[0])
        expected = m.sectional_lift(
            case.vertices[0], case.field[0], features["chord"], case.index[0, 2])
        np.testing.assert_allclose(case.sectional_cl[0], expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(case.metadata["sld_source"], "cfd_reference_integral")
        self.assertTrue(np.isfinite(case.sectional_cl).all())

    def test_external_sld_accepts_only_finite_128_or_one_by_128(self):
        values = np.linspace(.3, .7, 128)
        for supplied in (values, values[None, :]):
            with self.subTest(shape=supplied.shape):
                case = self.read_case(sld=supplied)
                np.testing.assert_allclose(case.sectional_cl, values[None, :], rtol=1e-6)
                self.assertNotEqual(case.metadata["sld_source"], "cfd_reference_integral")
                self.assertIn("external", case.metadata["sld_source"])
        for bad in (np.zeros(127), np.zeros((2, 128)),
                    np.full(128, np.nan), np.full((1, 128), np.inf)):
            with self.subTest(shape=bad.shape):
                with SuperWingReader(self.source) as reader:
                    with self.assertRaises(ValueError):
                        reader.read(1, sld=bad)

    def test_reynolds_overrides_remain_actual_values_in_original_sample_order(self):
        with SuperWingReader(self.source, reynolds=8e6) as reader:
            self.assertEqual(float(reader.read(3, sld_mode="cfd-reference").reynolds[0]), 8e6)
        path = self.root / "actual_re.npy"
        np.save(path, np.array([1e7, 1.1e7, 1.2e7, 1.3e7]))
        with SuperWingReader(self.source, reynolds_file=path) as reader:
            self.assertEqual(float(reader.read(1, sld_mode="cfd-reference").reynolds[0]), 1.1e7)
            self.assertEqual(float(reader.read(3, sld_mode="cfd-reference").reynolds[0]), 1.3e7)

    def test_row_rejects_out_of_range_float_and_boolean(self):
        with SuperWingReader(self.source) as reader:
            for row in (-1, 4, 10, .5, 1., True, False):
                with self.subTest(row=row):
                    with self.assertRaises((ValueError, IndexError, TypeError)):
                        reader.read(row)

    def test_iteration_is_lazy_and_respects_start_stop(self):
        with SuperWingReader(self.source) as reader:
            with patch.object(reader, "read", wraps=reader.read) as read:
                cases = reader.iter_cases(start=1, stop=3, sld_mode="cfd-reference")
                self.assertEqual(read.call_count, 0)
                first = next(cases)
                self.assertEqual(first.metadata["source_row"], 1)
                self.assertEqual(read.call_count, 1)
                second = next(cases)
                self.assertEqual(second.metadata["source_row"], 2)
                self.assertEqual(read.call_count, 2)
                with self.assertRaises(StopIteration):
                    next(cases)
                self.assertEqual(read.call_count, 2)

    def test_closed_reader_leaves_source_unchanged_and_case_values_available(self):
        before = tree_state(self.source)
        case = self.read_case(1)
        # Returned one-case snapshots must remain usable after context exit.
        self.assertTrue(np.isfinite(case.field).all())
        self.assertTrue(np.isfinite(case.centers).all())
        if case.field.flags.writeable:
            case.field[0, 0, 0, 0] += 1
        self.assertEqual(before, tree_state(self.source))

    def test_export_without_checkpoint_roundtrips_inputs_cfd_and_provenance(self):
        case = self.read_case(3)
        before = tree_state(self.source)
        output = self.root / "delivery"
        leaf_names = ("foilpara/foilgeo", "foilpara/foiloc", "foilpara/foilsld",
                      "foilperf/cfd", "foilperf/network")
        documented = {}
        for name in leaf_names:
            leaf = output / name
            leaf.mkdir(parents=True)
            content = ("# Format\n\n" + name + "\n").encode("utf-8")
            (leaf / "format.md").write_bytes(content)
            documented[name] = content
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("Export created Adam")), \
             patch.object(torch.optim, "SGD", side_effect=AssertionError("Export created SGD")):
            result = export_case(case, output)
        self.assertIsInstance(result, dict)
        self.assertFalse(result["network_prediction_generated"])
        self.assertIsNone(result["network_root"])
        dataset = FoilParaData(output / "foilpara")
        try:
            sample = dataset[0]
            np.testing.assert_allclose(sample["sld"].numpy(), case.sectional_cl[0], rtol=1e-6)
            np.testing.assert_allclose(sample["conditions"].numpy(), case.index[0, [3, 2]], rtol=1e-6)
            self.assertEqual(float(sample["reynolds"]), 2.3e7)
        finally:
            dataset.close()
        cfd = m.SuperWingData(output / "foilperf" / "cfd", target="field", require_reynolds=True)
        np.testing.assert_array_equal(cfd.configs, case.configs)
        np.testing.assert_array_equal(cfd.index, case.cfd_index)
        np.testing.assert_array_equal(cfd.data, case.field)
        self.assertEqual(cfd.reynolds_at(0), 2.3e7)
        records = [json.loads(line) for line in
                   (output / "foilperf" / "cfd" / "running_info.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["running_id"], 509)
        self.assertEqual(records[0]["0_running_info"]["working_conditions"]["re"], 2.3e7)
        for name in leaf_names:
            self.assertEqual((output / name / "format.md").read_bytes(), documented[name])
        for name in leaf_names[:4]:
            provenance = json.loads((output / name / "source_case.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["status"], "complete")
            self.assertEqual(provenance["source_row"], 3)
            self.assertEqual(provenance["source_geometry_index"], 1)
            self.assertEqual(provenance["source_condition_id"], 7)
            self.assertEqual(provenance["original_id"], 509)
            self.assertEqual(provenance["sld_source"], "cfd_reference_integral")
        self.assertFalse((output / "foilperf" / "network" / "data.npy").exists())
        self.assertFalse((output / "foilperf" / "network" / "prediction_info.json").exists())
        self.assertEqual(before, tree_state(self.source))
        # Release native-reader mappings before Windows TemporaryDirectory cleanup.
        for value in (cfd.index, cfd.centers, cfd.vertices, cfd.data):
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def test_export_preflights_all_leaves_before_creating_any_output(self):
        case = self.read_case()
        for number, leaf_name in enumerate(("foilpara/foilgeo", "foilpara/foiloc", "foilpara/foilsld",
                                            "foilperf/cfd")):
            with self.subTest(leaf=leaf_name):
                output = self.root / ("blocked-" + str(number))
                leaf = output / leaf_name
                leaf.mkdir(parents=True)
                (leaf / "existing.npy").write_bytes(b"do not overwrite")
                before = tree_state(output)
                with self.assertRaises((ValueError, FileExistsError)):
                    export_case(case, output)
                self.assertEqual(before, tree_state(output))

    def test_without_checkpoint_existing_network_is_untouched(self):
        case = self.read_case()
        output = self.root / "delivery"
        network = output / "foilperf" / "network"
        network.mkdir(parents=True)
        (network / "data.npy").write_bytes(b"existing prediction must stay untouched")
        (network / "format.md").write_text("# Existing format\n", encoding="utf-8")
        (network / "notes").mkdir()
        (network / "notes" / "record.txt").write_text("existing history", encoding="utf-8")
        before = tree_state(network)
        result = export_case(case, output)
        self.assertFalse(result["network_prediction_generated"])
        self.assertIsNone(result["network_root"])
        self.assertEqual(before, tree_state(network))

    def test_export_rejects_source_same_descendant_and_ancestor(self):
        case = self.read_case()
        before = tree_state(self.source)
        for output in (self.source, self.source / "delivery", self.source.parent):
            with self.subTest(output=str(output)):
                with self.assertRaises(ValueError):
                    export_case(case, output)
                self.assertEqual(before, tree_state(self.source))

    def test_invalid_field_or_sld_is_rejected_before_creating_outputs(self):
        case = self.read_case()
        bad_field = np.array(case.field, copy=True)
        bad_field[0, 0, 0, 0] = np.nan
        bad_sld = np.full((1, 128), np.inf)
        for number, bad in enumerate((replace(case, field=bad_field),
                                      replace(case, field=np.zeros((1, 3, 128, 255))),
                                      replace(case, sectional_cl=bad_sld),
                                      replace(case, sectional_cl=np.zeros((1, 127))))):
            with self.subTest(case=number):
                output = self.root / ("invalid-" + str(number))
                with self.assertRaises(ValueError):
                    export_case(bad, output)
                self.assertFalse(output.exists())

    def test_unknown_sld_mode_is_rejected(self):
        with SuperWingReader(self.source) as reader:
            with self.assertRaises(ValueError):
                reader.read(1, sld_mode="unknown")

    def test_cli_default_and_explicit_reference_sld_modes(self):
        from superwing_io import build_parser
        parser = build_parser()
        for command in ("inspect", "export"):
            with self.subTest(command=command):
                common = [command, "--row", "1"]
                if command == "export":
                    common += ["--output-root", str(self.root / "delivery")]
                self.assertEqual(parser.parse_args(common).sld_mode, "vlm")
                self.assertEqual(parser.parse_args(common + ["--sld-mode", "cfd-reference"]).sld_mode,
                                 "cfd-reference")


if __name__ == "__main__":
    unittest.main()
