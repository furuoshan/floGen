"""Synthetic artifacts exercise verified compaction; no CFD is claimed here."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import openfoam_case_archive as archiving


class CaseArchiveTests(unittest.TestCase):
    def setUp(self):
        (ROOT/"build").mkdir(exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=ROOT/"build",prefix="foam-archive-test-")
        self.addCleanup(self.temporary.cleanup)
        self.folder=Path(self.temporary.name)
        self.case=self.folder/"case"
        self.case.mkdir()

    def fixture(self,status="success"):
        x=np.r_[np.linspace(1,0,11),np.linspace(.1,1,10)]
        conditions=np.array([.65,.75,1.5e7])
        provenance=np.zeros((),dtype=[("source_row","i8"),("family_id","i8")])
        fingerprint="a"*64
        np.savez_compressed(self.case/"request.npz",x=x,geometry=np.zeros(len(x)),conditions=conditions,
                            provenance=provenance,fingerprint=np.array(fingerprint))
        result={"status":status,"fingerprint":fingerprint,"solver":"OpenFOAM-v2412-rhoSimpleFoam",
                "turbulence_model":"kOmegaSST"}
        (self.case/"result.json").write_text(json.dumps(result))
        for directory in ("0","100","constant/polyMesh","postProcessing/forces/0",".runtime/tmp","system"):
            (self.case/directory).mkdir(parents=True)
        (self.case/"100/p").write_text("native internal field omitted on purpose\n"*4000)
        (self.case/"0/U").write_text("initial velocity field")
        (self.case/"constant/polyMesh/points").write_text("reconstructible points")
        (self.case/"constant/thermophysicalProperties").write_text("thermo dictionary exact text")
        (self.case/"system/fvSolution").write_text("SIMPLE { nNonOrthogonalCorrectors 2; }")
        (self.case/"postProcessing/forces/0/coefficient.dat").write_text("# Time Cl Cd\n100 .75 .012\n")
        (self.case/".runtime/tmp/program.log").write_text("native transient diagnostic")
        (self.case/"solver_alpha_00.log").write_text("Time = 100\nEnd\n"*1000)
        (self.case/"checkMesh.log").write_text("Cell volumes OK.\n")
        (self.case/"mesh_metadata.json").write_text(json.dumps({"span":.01}))
        (self.case/"case_conditions.json").write_text(json.dumps({"p_inf":101325}))
        # Unknown caller data is preserved, including a log not owned by solver prefixes.
        (self.case/"notes.txt").write_text("do not remove user notes")
        (self.case/"notes.log").write_text("do not remove user log")
        (self.case/"unrelated").mkdir()
        (self.case/"unrelated/original.npy").write_bytes(b"user source data")
        if status=="success":
            np.savez_compressed(self.case/"labels.npz",labels=np.zeros((2,len(x))))
            (self.case/"quality.json").write_text(json.dumps({"converged":True}))
        n=5
        raw={"face_centres":np.arange(n*3,dtype=float).reshape(n,3),"tangent":np.tile([1.,0.,0.],(n,1)),
             "p_pa":np.linspace(100000.,102000.,n),"native_wall_shear":np.arange(n*3,dtype=float).reshape(n,3)/10,
             "tau_body_pa":-np.arange(n*3,dtype=float).reshape(n,3)/10,
             "cp":np.linspace(-1,1,n),"cf":np.linspace(-.001,.003,n)}
        return {"converged":True,"raw_wall":raw,
                "force_history":{"time":np.array([1.,2.]),"cl":np.array([.74,.75]),"cd":np.array([.01,.012])},
                "residuals":{"history":[{"time":2,"field":"p","initial":1e-6,"final":1e-8,"linear_iterations":2}]}}

    def test_success_keeps_exact_wall_and_diagnostics_before_removing_only_owned_bulk(self):
        solution=self.fixture()
        record=archiving.compact_case(ROOT,self.case,solution)
        self.assertEqual(record["phase"],"complete")
        self.assertTrue(record["crc_verified"] and record["per_file_sha256_verified"])
        self.assertGreater(record["bytes_reclaimed"],0)
        for path in ("0","100","constant/polyMesh","postProcessing",".runtime","solver_alpha_00.log","checkMesh.log"):
            self.assertFalse((self.case/path).exists(),path)
        for path in ("result.json","quality.json","labels.npz","request.npz","raw_surface.npz","system/fvSolution",
                     "constant/thermophysicalProperties","mesh_metadata.json","notes.txt","notes.log","unrelated/original.npy"):
            self.assertTrue((self.case/path).is_file(),path)
        with np.load(self.case/"raw_surface.npz",allow_pickle=False) as raw:
            for name,value in solution["raw_wall"].items():
                np.testing.assert_array_equal(raw[name],value)
        manifest=archiving._verify_archive(self.case/"raw_diagnostics.zip")
        self.assertIn("solver_alpha_00.log",manifest["files"])
        self.assertIn("postProcessing/forces/0/coefficient.dat",manifest["files"])
        self.assertIn("derived/native_solution_histories.json",manifest["files"])
        self.assertIn("reproduction_code/openfoam_airfoil_mesh.py",manifest["files"])
        self.assertNotIn("100/p",manifest["files"])
        self.assertNotIn("constant/polyMesh/points",manifest["files"])
        again=archiving.compact_case(ROOT,self.case)
        self.assertTrue(again["already_compacted"])

    def test_failed_case_keeps_actual_diagnostics_without_inventing_wall_or_labels(self):
        self.fixture("failed")
        record=archiving.compact_case(ROOT,self.case)
        self.assertEqual(record["case_status"],"failed")
        self.assertIsNone(record["raw_surface"])
        self.assertFalse((self.case/"raw_surface.npz").exists())
        self.assertFalse((self.case/"labels.npz").exists())
        manifest=archiving._verify_archive(self.case/"raw_diagnostics.zip")
        self.assertIn("solver_alpha_00.log",manifest["files"])
        self.assertNotIn("derived/native_solution_histories.json",manifest["files"])

    def test_zip_verification_failure_does_not_remove_original_fields(self):
        solution=self.fixture()
        with patch.object(archiving,"_verify_archive",side_effect=ValueError("CRC verification failed")):
            with self.assertRaisesRegex(ValueError,"CRC"):
                archiving.compact_case(ROOT,self.case,solution)
        self.assertTrue((self.case/"100/p").is_file())
        self.assertTrue((self.case/"constant/polyMesh/points").is_file())
        self.assertTrue((self.case/"solver_alpha_00.log").is_file())
        self.assertFalse((self.case/"archive_record.json").exists())

    def test_interrupted_cleanup_resumes_from_verified_archive_without_new_solution(self):
        solution=self.fixture()
        with patch.object(archiving,"_cleanup",side_effect=RuntimeError("simulated power loss")):
            with self.assertRaisesRegex(RuntimeError,"power loss"):
                archiving.compact_case(ROOT,self.case,solution)
        partial=json.loads((self.case/"archive_record.json").read_text())
        self.assertEqual(partial["phase"],"archive_verified")
        recovered=archiving.compact_case(ROOT,self.case)
        self.assertEqual(recovered["phase"],"complete")
        self.assertFalse((self.case/"100").exists())

    def test_interrupted_archive_publication_recovers_its_sha_bound_pending_zip(self):
        solution=self.fixture()
        real_replace=archiving.os.replace
        def interrupted(source,target):
            if Path(target).name=="raw_diagnostics.zip":
                raise RuntimeError("simulated publication interruption")
            return real_replace(source,target)
        with patch.object(archiving.os,"replace",side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,"publication"):
                archiving.compact_case(ROOT,self.case,solution)
        self.assertTrue((self.case/"100/p").is_file())
        self.assertFalse((self.case/"raw_diagnostics.zip").exists())
        record=json.loads((self.case/"archive_record.json").read_text())
        self.assertTrue((self.case/record["archive_pending_file"]).is_file())
        recovered=archiving.compact_case(ROOT,self.case)
        self.assertEqual(recovered["phase"],"complete")
        self.assertTrue((self.case/"raw_diagnostics.zip").is_file())
        self.assertFalse((self.case/"100").exists())

    def test_missing_success_wall_data_never_deletes_the_volume_fields(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError,"genuine converged raw-wall"):
            archiving.compact_case(ROOT,self.case)
        self.assertTrue((self.case/"100/p").exists())

    def test_symlink_to_user_data_is_rejected_before_any_archive_or_deletion(self):
        self.fixture("failed")
        source=self.folder/"original-user-data"
        source.mkdir()
        (source/"keep.txt").write_text("must survive")
        try:
            (self.case/"200").symlink_to(source,target_is_directory=True)
        except OSError:
            self.skipTest("Host does not permit creating test symlinks")
        with self.assertRaisesRegex(ValueError,"symlink"):
            archiving.compact_case(ROOT,self.case)
        self.assertEqual((source/"keep.txt").read_text(),"must survive")
        self.assertTrue((self.case/"100/p").exists())
        self.assertFalse((self.case/"raw_diagnostics.zip").exists())

    def test_changed_preserved_fingerprint_cannot_resume_compaction(self):
        solution=self.fixture()
        archiving.compact_case(ROOT,self.case,solution)
        path=self.case/"result.json"
        result=json.loads(path.read_text());result["fingerprint"]="b"*64
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError,"fingerprint mismatch"):
            archiving.compact_case(ROOT,self.case)


if __name__=="__main__":
    unittest.main()
