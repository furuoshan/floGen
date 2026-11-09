"""Numerical checks using the real official PyTornado backend, without training.

No fake VLM output or surrogate solver is used. The optional real-database test
only reads the source and constructs two tiny temporary cases for independence.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_main import m
from low_fidelity_sld import estimate_spanwise_lift
from superwing_reader import SuperWingReader
from superwing_io import export_case


def flat_plate_geometry():
    """An unswept, untwisted rectangular half-wing of unit chord and half-span 2."""
    centers, vertices = m.synthetic_geometry()
    contour = np.r_[np.linspace(1., 0., 128), np.linspace(0., 1., 129)[1:]]
    contour = np.r_[contour, contour[0]]
    vertices[0] = contour[None, :]
    vertices[1] = 0.
    vertices[2] = np.linspace(0., 2., 129)[:, None]
    centers = .25 * (vertices[:, :-1, :-1] + vertices[:, 1:, :-1] +
                     vertices[:, :-1, 1:] + vertices[:, 1:, 1:])
    return centers, vertices


def area_weighted_cl(values, vertices):
    """Integrate reported sectional CL using physical strip chord and span width."""
    stations = .5 * (vertices[:, :-1] + vertices[:, 1:])
    chord = np.ptp(stations[0], axis=-1)
    dz = np.abs(np.diff(vertices[2, :, 0]))
    weights = chord * dz
    return float(np.sum(np.asarray(values)[0] * weights) / weights.sum())


class LowFidelitySLDTests(unittest.TestCase):
    def test_symmetric_flat_plate_at_zero_incidence_has_zero_lift(self):
        centers, vertices = flat_plate_geometry()
        values, info = estimate_spanwise_lift(centers, vertices, mach=.2, aoa_deg=0.,
                                              chord_panels=4, span_panels=16)
        self.assertEqual(values.shape, (1, 128))
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(float(np.abs(values).max()), 1e-6)
        self.assertLess(abs(area_weighted_cl(values, vertices)), 1e-6)
        self.assertIsInstance(info, dict)
        self.assertIn("pytornado", json.dumps(info).lower())

    def test_positive_incidence_and_span_mesh_convergence(self):
        centers, vertices = flat_plate_geometry()
        totals = []
        for panels in (16, 32, 64):
            values, info = estimate_spanwise_lift(centers, vertices, mach=.2, aoa_deg=4.,
                                                  chord_panels=4, span_panels=panels)
            self.assertEqual(values.shape, (1, 128))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(bool(np.any(values > 0)))
            total = area_weighted_cl(values, vertices)
            self.assertGreater(total, 0.)
            totals.append(total)
        fine_change = abs(totals[2] - totals[1]) / abs(totals[2])
        self.assertLess(fine_change, .05, msg=f"span panel convergence CL={totals}")
        print(json.dumps({"pytornado_flat_plate_alpha_deg": 4.,
                          "span_panels": [16, 32, 64], "chord_panels": 4,
                          "area_weighted_cl": totals, "last_relative_change": fine_change}), flush=True)

    def test_geometry_only_input_without_any_cfd_file_can_estimate_sld(self):
        centers, vertices = flat_plate_geometry()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "geom0.npy", centers[None])
            np.save(root / "origingeom.npy", vertices[None])
            self.assertFalse((root / "data.npy").exists())
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            with patch.object(m, "sectional_lift", side_effect=AssertionError("VLM used CFD integral")):
                values, info = estimate_spanwise_lift(
                    np.load(root / "geom0.npy")[0], np.load(root / "origingeom.npy")[0],
                    mach=.2, aoa_deg=3., chord_panels=4, span_panels=16)
            self.assertEqual(values.shape, (1, 128))
            self.assertTrue(np.isfinite(values).all())
            self.assertGreater(area_weighted_cl(values, vertices), 0.)
            self.assertFalse((root / "data.npy").exists())
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})

    def test_invalid_geometry_or_state_is_rejected(self):
        centers, vertices = flat_plate_geometry()
        for bad_mach, bad_alpha in ((np.nan, 3.), (.2, np.inf)):
            with self.subTest(mach=bad_mach, alpha=bad_alpha):
                with self.assertRaises(ValueError):
                    estimate_spanwise_lift(centers, vertices, bad_mach, bad_alpha)
        with self.assertRaises(ValueError):
            estimate_spanwise_lift(centers[:, :, :-1], vertices, .2, 3.)

    def test_real_superwing_default_sld_is_independent_of_cfd_values_and_exported_as_vlm(self):
        source = Path(os.environ.get("SUPERWING_ROOT", "D:/Superwing"))
        # This is a deliberate integration check against the locally supplied
        # database. A missing PyTornado dependency is never skipped or replaced.
        self.assertTrue((source / "data.npy").is_file(),
                        "Set SUPERWING_ROOT to a downloaded native SuperWing database")
        index = np.load(source / "index.npy", mmap_mode="r", allow_pickle=False)
        first_index = np.array(index[:1], copy=True)
        geom = np.load(source / "geom0.npy", mmap_mode="r", allow_pickle=False)
        vertices = np.load(source / "origingeom.npy", mmap_mode="r", allow_pickle=False)
        field = np.load(source / "data.npy", mmap_mode="r", allow_pickle=False)
        g, k = map(int, first_index[0, :2])
        # Preserve the selected original geometry/condition metadata in each tiny
        # native-format copy, changing only the CFD surface quantities.
        all_configs = np.loadtxt(source / "configs.dat", skiprows=1, ndmin=2)
        configs = all_configs[g:g + 1].copy()
        copied_geom = np.array(geom[g:g + 1], copy=True)
        copied_vertices = np.array(vertices[g:g + 1], copy=True)
        copied_field = np.array(field[:1], copy=True)
        first_index[0, 0] = 0
        original = int(configs[0, 0])
        log = None
        with (source / "running_info.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record["running_id"]) == original:
                    log = record
                    break
        self.assertIsNotNone(log)
        for mapped in (index, geom, vertices, field):
            mapped._mmap.close()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cases = []
            for number in range(2):
                root = base / str(number)
                root.mkdir()
                np.savetxt(root / "configs.dat", configs, header="native 55 columns", comments="")
                np.save(root / "index.npy", first_index)
                np.save(root / "geom0.npy", copied_geom)
                np.save(root / "origingeom.npy", copied_vertices)
                altered = copied_field if number == 0 else copied_field * 3. + 7.
                np.save(root / "data.npy", altered)
                (root / "running_info.jsonl").write_text(json.dumps(log) + "\n", encoding="utf-8")
                with SuperWingReader(root) as reader:
                    case = reader.read(0)
                self.assertEqual(case.metadata["sld_source"], "pytornado_vlm")
                self.assertIs(case.metadata["sld_requires_cfd"], False)
                self.assertIn("sld_details", case.metadata)
                self.assertTrue(np.isfinite(case.sectional_cl).all())
                cases.append(case)
            self.assertFalse(np.array_equal(cases[0].field, cases[1].field))
            np.testing.assert_allclose(cases[0].sectional_cl, cases[1].sectional_cl,
                                       rtol=1e-12, atol=1e-12)
            result = export_case(cases[0], base / "delivery")
            self.assertFalse(result["network_prediction_generated"])
            for relative in ("foilpara/foilgeo", "foilpara/foiloc", "foilpara/foilsld", "foilperf/cfd"):
                provenance = json.loads((base / "delivery" / relative / "source_case.json").read_text(encoding="utf-8"))
                self.assertEqual(provenance["sld_source"], "pytornado_vlm")
                self.assertIs(provenance["sld_requires_cfd"], False)
                self.assertIs(provenance["reference_sld_uses_cfd"], False)
            self.assertFalse((base / "delivery" / "foilperf" / "network" / "data.npy").exists())


if __name__ == "__main__":
    unittest.main()
