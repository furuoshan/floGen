"""CPU checks for the actual floGen bridge; every training run is 1-2 steps.

bootstrap_fixture(base) is also used by the DSPy cross-process integration test.
All generated arrays/checkpoints live under the caller's temporary directory.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tested_flogen_bridge",ROOT/"bridge.py")
b = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = b
_spec.loader.exec_module(b)
m = b.m
_fixture_path = ROOT/"tests"/"test_main.py"
if not _fixture_path.is_file():
    _fixture_path = Path("D:/floGen/tests/test_main.py")
_fixture_spec = importlib.util.spec_from_file_location("bridge_source_test_fixtures",_fixture_path)
f = importlib.util.module_from_spec(_fixture_spec)
sys.modules[_fixture_spec.name] = f
_fixture_spec.loader.exec_module(f)
torch.set_num_threads(2)


def bootstrap_fixture(base):
    """Return real three-stage 1-step trained checkpoint and native small arrays."""
    base = Path(base).resolve()
    base.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(base/"torch-cache"))
    torch.set_num_threads(2)
    m.seed_everything(42)
    wing = f.wing_fixture(base/"cfd")
    foil = f.foil_fixture(base/"foil")
    spec = m.ModelSpec()
    train = np.asarray([0,1],dtype=np.int64)
    validation = np.asarray([2,3],dtype=np.int64)
    airfoil = m.AirfoilPredictor(spec)
    lift = m.SpanwiseLiftPredictor(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        air_info = m.fit_model(airfoil,m.SuperFoilData(foil,"VT-GU",3),train,validation,
                               "airfoil","cpu",1,1,max_steps=1)
        lift_info = m.fit_model(lift,m.SuperWingData(wing,target="sld"),train,validation,
                                "sld","cpu",1,1,max_steps=1)
        model = m.PhysicsEmbeddedWingPredictor(spec,airfoil,lift)
        wing_info = m.fit_model(model,m.SuperWingData(wing,require_reynolds=True),train,validation,
                                "wing","cpu",1,1,max_steps=1)
    wing_info.update(train_shape_ids=[101],validation_shape_ids=[509],sld_train_shape_ids=[101],
                     airfoil_training_steps=air_info["training_steps"],sld_training_steps=lift_info["training_steps"],
                     airfoil_reynolds_input={"source":"user_confirmed_default","value":2e7,"unit":"actual_reynolds"})
    checkpoint = base/"initial.pt"
    m.save_checkpoint(checkpoint,"wing",model,spec,wing_info,[wing,foil])
    return {"cfd_root":str(wing),"checkpoint":str(checkpoint),
            "train_rows":[0,1],"validation_rows":[2,3]}


def job_fixture(base, fixture, round_id=0, checkpoint_in=None, fingerprint=None, steps=1):
    base = Path(base).resolve()
    base.mkdir(parents=True,exist_ok=True)
    return {"schema_version":"wing-loop-v1","operation":"train_predict","job_id":f"round-{round_id:04d}",
            "round_id":round_id,"checkpoint_in":str(Path(checkpoint_in or fixture["checkpoint"]).resolve()),
            "checkpoint_out":str(base/"next.pt"),
            "train":[{"root":fixture["cfd_root"],"rows":fixture["train_rows"]}],
            "validation":{"root":fixture["cfd_root"],"rows":fixture["validation_rows"],"expected_fingerprint":fingerprint},
            "prediction":{"input_root":fixture["cfd_root"],"output_root":str(base/"network")},
            "plan":{"learning_rate":1e-4,"steps":steps,"batch_size":1},"device":"cpu","seed":42,
            "training_data_policy":"cumulative","result_path":str(base/"worker_result.json")}


def write_job(path, job):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(job,indent=2),encoding="utf-8")
    return path


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory(prefix="flogen-bridge-fixture-")
        cls.fixture = bootstrap_fixture(Path(cls.shared.name)/"bootstrap")

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="flogen-bridge-job-")
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def run_quiet(self, job_path):
        with contextlib.redirect_stdout(io.StringIO()):
            return b.run_job(job_path)

    def test_actual_train_predict_and_optimizer_resume_with_fixed_normalizers(self):
        source = Path(self.fixture["cfd_root"])
        hashes_before = f.hashes(source)
        initial,_,initial_payload = m.load_checkpoint(self.fixture["checkpoint"],"wing")
        norm_before = b.normalizer_digest(initial)
        prior_before = b.prior_digest(initial)
        first = job_fixture(self.root/"r0",self.fixture,steps=1)
        job_path = write_job(self.root/"r0"/"job.json",first)
        result = self.run_quiet(job_path)
        self.assertEqual(result["status"],"completed",result)
        self.assertEqual(result["request_sha256"],hashlib.sha256(job_path.read_bytes()).hexdigest())
        self.assertEqual(result["steps_performed"],1)
        self.assertTrue(result["normalization_unchanged"])
        self.assertEqual(result["checkpoint_sha256"],b.sha256_file(first["checkpoint_out"]))
        self.assertEqual(set(result["feedback"]),{"round_id","validation_error_before","validation_error_after","metrics","notes"})
        self.assertTrue(all(type(value) is float for value in result["feedback"]["metrics"].values()))
        self.assertTrue(np.isfinite(result["feedback"]["validation_error_after"]))
        changed,spec,payload = m.load_checkpoint(first["checkpoint_out"],"wing")
        self.assertEqual(payload["training"]["cumulative_training_steps"],2)
        self.assertTrue(payload["training"]["bridge_optimizer_reset"])
        self.assertEqual(b.normalizer_digest(changed),norm_before)
        self.assertEqual(b.prior_digest(changed),prior_before)
        self.assertNotEqual(b.state_digest(initial.residual.state_dict()),b.state_digest(changed.residual.state_dict()))
        output = Path(result["prediction_dir"])
        self.assertEqual(np.load(output/"data.npy").shape,(4,3,128,256))
        for name in ("index.npy","geom0.npy","origingeom.npy","configs.dat","running_info.jsonl"):
            self.assertEqual((source/name).read_bytes(),(output/name).read_bytes())
        second = job_fixture(self.root/"r1",self.fixture,round_id=1,checkpoint_in=first["checkpoint_out"],
                             fingerprint=result["validation_fingerprint"],steps=2)
        result2 = self.run_quiet(write_job(self.root/"r1"/"job.json",second))
        self.assertEqual(result2["status"],"completed",result2)
        self.assertEqual(result2["steps_performed"],2)
        self.assertEqual(result2["validation_fingerprint"],result["validation_fingerprint"])
        resumed,_,next_payload = m.load_checkpoint(second["checkpoint_out"],"wing")
        self.assertFalse(next_payload["training"]["bridge_optimizer_reset"])
        self.assertEqual(next_payload["training"]["cumulative_training_steps"],4)
        self.assertEqual(next_payload["training"]["checkpoint_in_sha256"],result["checkpoint_sha256"])
        self.assertEqual(b.normalizer_digest(resumed),norm_before)
        self.assertEqual(b.prior_digest(resumed),prior_before)
        self.assertEqual(hashes_before,f.hashes(source))
        with patch.object(b,"execute_job",side_effect=AssertionError("must not execute twice")):
            self.assertEqual(b.run_job(job_path),result)
        self.assertTrue(Path(first["result_path"]).with_suffix(".started.json").is_file())

    def test_expected_fingerprint_mismatch_fails_without_training_or_feedback(self):
        job = job_fixture(self.root,self.fixture,fingerprint="wrong")
        result = self.run_quiet(write_job(self.root/"job.json",job))
        self.assertEqual(result["status"],"failed")
        self.assertIn("fingerprint",result["error"]["message"])
        self.assertNotIn("feedback",result)
        self.assertFalse(Path(job["checkpoint_out"]).exists())
        with self.assertRaisesRegex(ValueError,"cannot rerun"):
            b.run_job(self.root/"job.json")

    def test_current_or_historical_validation_leakage_rejected(self):
        job = job_fixture(self.root,self.fixture)
        job["train"][0]["rows"] = [2]
        result = self.run_quiet(write_job(self.root/"job.json",job))
        self.assertEqual(result["status"],"failed")
        self.assertIn("overlap current training",result["error"]["message"])
        model,spec,payload = m.load_checkpoint(self.fixture["checkpoint"],"wing")
        payload["training"]["train_shape_ids"] = [509]
        dataset = m.SuperWingData(self.fixture["cfd_root"],require_reynolds=True)
        with self.assertRaisesRegex(ValueError,"historical wing"):
            b.check_no_validation_leakage(payload,[(dataset,np.array([0]))],dataset,np.array([2]))

    def test_duplicate_canonical_training_case_is_rejected(self):
        job = job_fixture(self.root,self.fixture)
        job["train"].append(dict(job["train"][0]))
        result = self.run_quiet(write_job(self.root/"job.json",job))
        self.assertEqual(result["status"],"failed")
        self.assertIn("Duplicate training canonical",result["error"]["message"])

    def test_existing_claim_prevents_training_replay(self):
        job = job_fixture(self.root,self.fixture)
        path = write_job(self.root/"job.json",job)
        claim = Path(job["result_path"]).with_suffix(".started.json")
        claim.write_text("{}",encoding="utf-8")
        with patch.object(b,"execute_job",side_effect=AssertionError("must not execute")):
            with self.assertRaises(FileExistsError):
                b.run_job(path)
        self.assertFalse(Path(job["result_path"]).exists())

    def test_validation_fingerprint_covers_actual_reynolds_and_selected_truth(self):
        first = m.SuperWingData(self.fixture["cfd_root"],require_reynolds=True)
        rows = np.array(self.fixture["validation_rows"],dtype=np.int64)
        original = b.validation_fingerprint(first,rows)
        changed_re = m.SuperWingData(self.fixture["cfd_root"],require_reynolds=True,reynolds=3e7)
        self.assertNotEqual(original,b.validation_fingerprint(changed_re,rows))
        self.assertEqual(original,b.validation_fingerprint(first,rows))
        other = f.wing_fixture(self.root/"other")
        data = np.load(other/"data.npy");data[2,0,0,0] += .01
        np.save(other/"data.npy",data)
        changed = m.SuperWingData(other,require_reynolds=True)
        self.assertNotEqual(original,b.validation_fingerprint(changed,rows))

    def test_unsafe_output_or_noninteger_rows_rejected(self):
        job = job_fixture(self.root,self.fixture)
        job["prediction"]["output_root"] = self.fixture["cfd_root"]
        with self.assertRaises(ValueError):
            b.validate_job(job)
        with self.assertRaisesRegex(ValueError,"integer"):
            b.checked_rows([True],4,"training")
        with self.assertRaisesRegex(ValueError,"duplicates"):
            b.checked_rows([0,0],4,"training")


class ProtocolValidationTests(unittest.TestCase):
    """Fast input checks; no model construction or training is needed."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="flogen-bridge-protocol-")
        self.root = Path(self.temporary.name).resolve()
        fixture = {"cfd_root":str(self.root/"source"),"checkpoint":str(self.root/"initial.pt"),
                   "train_rows":[0],"validation_rows":[1]}
        self.job = job_fixture(self.root/"job",fixture)

    def tearDown(self):
        self.temporary.cleanup()

    def test_indexed_cuda_and_seed_boundaries(self):
        for device in ("cpu","cuda","cuda:0","cuda:12"):
            self.job["device"] = device
            b.validate_job(self.job)
        for device in ("auto","cuda:","cuda:-1","cuda:abc","cuda:1.0"):
            self.job["device"] = device
            with self.assertRaisesRegex(ValueError,"device"):
                b.validate_job(self.job)
        self.job["device"] = "cpu"
        for seed in (0,2**32-1):
            self.job["seed"] = seed
            b.validate_job(self.job)
        for seed in (-1,2**32,True):
            self.job["seed"] = seed
            with self.assertRaisesRegex(ValueError,"seed"):
                b.validate_job(self.job)

    def test_expected_input_hash_rejected_before_model_load(self):
        checkpoint = Path(self.job["checkpoint_in"])
        checkpoint.write_bytes(b"fixture bytes; must never be loaded as a model")
        self.job["checkpoint_in_sha256"] = "0"*64
        paths = b.validate_job(self.job)
        with patch.object(m,"load_checkpoint",side_effect=AssertionError("must not load mismatched checkpoint")):
            with self.assertRaisesRegex(ValueError,"checkpoint_in_sha256"):
                b.execute_job(self.job,paths,"request")
        self.job["checkpoint_in_sha256"] = "invalid"
        with self.assertRaisesRegex(ValueError,"SHA256"):
            b.validate_job(self.job)


def geometry_fixture(base,fixture,index_columns=6):
    """Geometry-only handoff: metadata/mesh and actual Re, never CFD fields."""
    base = Path(base).resolve()
    base.mkdir(parents=True,exist_ok=True)
    source = Path(fixture["cfd_root"])
    for name in ("configs.dat","geom0.npy","origingeom.npy"):
        (base/name).write_bytes((source/name).read_bytes())
    np.save(base/"index.npy",np.load(source/"index.npy")[:,:index_columns])
    files = {name:b.sha256_file(base/name) for name in ("configs.dat","index.npy","geom0.npy","origingeom.npy")}
    manifest = {"schema_version":"wing-geometry-v1","files":files,"reynolds":2e7}
    path = base/"geometry_manifest.json"
    path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return base,b.sha256_file(path)


def prediction_job_fixture(base,fixture,geometry,fingerprint):
    base = Path(base).resolve()
    base.mkdir(parents=True,exist_ok=True)
    return {"schema_version":"wing-loop-v1","operation":"predict","job_id":"round-0000-predict","round_id":0,
            "checkpoint_in":fixture["checkpoint"],"checkpoint_in_sha256":b.sha256_file(fixture["checkpoint"]),
            "geometry":{"root":str(geometry),"expected_fingerprint":fingerprint},
            "prediction":{"output_root":str(base/"network")},"batch_size":2,"device":"cpu","seed":42,
            "result_path":str(base/"prediction_result.json")}


class IndependentPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory(prefix="flogen-independent-predict-")
        existing = os.environ.get("FLOGEN_BRIDGE_FIXTURE")
        cls.fixture = json.loads(Path(existing).read_text(encoding="utf-8")) if existing else bootstrap_fixture(Path(cls.shared.name)/"bootstrap")

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def test_real_geometry_only_prediction_and_manifest_tamper_rejection(self):
        base = Path(self.shared.name).resolve()
        geometry,fingerprint = geometry_fixture(base/"foilgeo"/"geometry",self.fixture)
        self.assertFalse((geometry/"data.npy").exists())
        job = prediction_job_fixture(base/"predict",self.fixture,geometry,fingerprint)
        job_path = write_job(base/"predict"/"job.json",job)
        before = f.hashes(geometry)
        original_read = m.read_npy
        def no_cfd_fields(path):
            if Path(path).name == "data.npy":
                raise AssertionError("Pure prediction must never open CFD flow data")
            return original_read(path)
        with patch.object(m,"read_npy",side_effect=no_cfd_fields), \
             patch.object(m,"fit_normalizers",side_effect=AssertionError("must not train")), \
             patch.object(torch.optim,"Adam",side_effect=AssertionError("must not construct optimizer")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = b.run_job(job_path)
        self.assertEqual(result["status"],"completed",result)
        self.assertEqual(result["operation"],"predict")
        self.assertEqual(result["geometry_fingerprint"],fingerprint)
        self.assertEqual(result["steps_performed"],0)
        self.assertNotIn("feedback",result)
        self.assertEqual(result["checkpoint_sha256"],job["checkpoint_in_sha256"])
        self.assertEqual(before,f.hashes(geometry))
        output = Path(result["prediction_dir"])
        fields = np.load(output/"data.npy")
        self.assertEqual(fields.shape,(4,3,128,256))
        self.assertTrue(np.isfinite(fields).all())
        np.testing.assert_array_equal(np.load(output/"index.npy"),np.load(geometry/"index.npy"))
        self.assertEqual(np.load(output/"index.npy").shape,(4,6))
        marker = json.loads((output/"prediction_info.json").read_text())
        self.assertEqual(marker["status"],"complete")
        self.assertEqual(marker["stored_multipliers"],[1,150,300])
        self.assertEqual(marker["index_layout"],"geometry_conditions_v1")
        self.assertEqual(marker["index_coefficients_role"],"unavailable_not_measured_not_predicted")
        self.assertEqual(marker["geometry_fingerprint"],fingerprint)
        self.assertEqual(b.sha256_file(output/"source_geometry_manifest.json"),fingerprint)
        # A compatible legacy metadata array is read only through its first six columns.
        compatible,_ = geometry_fixture(base/"compatible",self.fixture,index_columns=12)
        manifest,_ = b.geometry_manifest(compatible)
        dataset = b.GeometryData(compatible,manifest)
        self.assertEqual(dataset.index.shape,(4,6))
        self.assertNotIn("target",dataset[0])
        del dataset
        # Bind the same job to modified geometry: reject before loading any model.
        tampered = base/"tampered"
        tampered_geometry,tampered_fingerprint = geometry_fixture(tampered/"geometry",self.fixture)
        path = tampered_geometry/"configs.dat"
        path.write_bytes(path.read_bytes()+b"\n")
        bad_job = prediction_job_fixture(tampered/"job",self.fixture,tampered_geometry,tampered_fingerprint)
        with patch.object(m,"load_checkpoint",side_effect=AssertionError("must verify geometry first")):
            failed = b.run_job(write_job(tampered/"job"/"job.json",bad_job))
        self.assertEqual(failed["status"],"failed")
        self.assertIn("Geometry file SHA256 mismatch",failed["error"]["message"])
        self.assertNotIn("feedback",failed)
        self.assertFalse(Path(bad_job["prediction"]["output_root"]).exists())


if __name__ == "__main__":
    unittest.main()
