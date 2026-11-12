"""Parser/path/failure tests with explicit synthetic text, never a native solve."""
from pathlib import Path
import json
import tempfile
import hashlib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_superwing_xfoil as runner


class XfoilRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def outputs(self, cp_y_column=False, corrupt_wake_match=False):
        half = np.linspace(0,1,11)
        x = np.r_[half[::-1],half[1:]]
        upper = np.arange(len(x)) <= 10
        y = np.where(upper,.05*np.sin(np.pi*x),-.05*np.sin(np.pi*x))
        cp = np.where(upper,-.5-.1*x,.2+.1*x)
        cf = np.where(upper,.003,.004)
        ue = np.where(upper,.8,-.8)
        cf[17] = -.001  # A real wall-backflow sign must remain after conversion.
        ue[10],cf[10] = 0,0
        cp_path,bl_path = self.root/"cp.dat",self.root/"bl.dat"
        if cp_y_column:
            np.savetxt(cp_path,np.c_[x,y,cp],header="x y Cp")
        else:
            np.savetxt(cp_path,np.c_[x,cp],header="x Cp")
        bl = np.c_[np.arange(len(x))*.1,x,y,ue,np.full(len(x),.01),
                   np.full(len(x),.005),cf,np.full(len(x),1.5)]
        if corrupt_wake_match:
            bl[16,1] += .01
        # Includes two wake rows whose coordinates are inside the naive x<=1
        # filter, proving that CPWR matching, not that heuristic, removes them.
        wake = np.array([[3,.99,0,.8,.01,.005,99,1.5],
                         [3.1,1.2,0,.8,.01,.005,99,1.5]])
        np.savetxt(bl_path,np.r_[bl,wake],header="s x y Ue/Vinf Dstar Theta Cf H")
        return cp_path,bl_path,x,cp

    def test_cpwr_formats_wake_exclusion_and_signed_backflow(self):
        for columns in (False,True):
            with self.subTest(cp_has_y=columns):
                cp_path,bl_path,x,original_cp = self.outputs(columns)
                labels,details = runner.interpolate_labels(cp_path,bl_path,x)
                self.assertEqual(labels.shape,(2,21))
                np.testing.assert_allclose(labels[0],original_cp[::-1],atol=1e-6)
                self.assertLess(labels[1,1],0)  # ordinary lower-surface flow
                self.assertGreater(labels[1,3],0)  # lower wall backflow
                self.assertGreater(labels[1,-2],0)  # ordinary upper flow
                self.assertLess(np.abs(labels[1]).max(),.01)  # no wake Cf=99
                self.assertEqual(details["surface_node_counts"]["boundary_layer"],21)

    def test_dump_node_mismatch_is_rejected(self):
        cp_path,bl_path,x,_ = self.outputs(corrupt_wake_match=True)
        with self.assertRaisesRegex(ValueError,"not aligned"):
            runner.interpolate_labels(cp_path,bl_path,x)

    def test_polar_requires_one_target_cl_solution(self):
        path = self.root/"polar.dat"
        path.write_text("alpha CL CD CDp CM Top_Xtr Bot_Xtr\n-----\n2.3 .73 .01 .003 -.05 .7 .8\n")
        self.assertAlmostEqual(runner.read_polar(path,.73)["cl"],.73)
        with self.assertRaisesRegex(ValueError,"does not match"):
            runner.read_polar(path,.8)
        path.write_text(path.read_text()+"2.4 .74 .01 .003 -.05 .7 .8\n")
        with self.assertRaisesRegex(ValueError,"exactly one"):
            runner.read_polar(path,.74)

    def test_paths_cannot_escape_project(self):
        self.assertEqual(runner.inside(self.root,"a/b"),self.root/"a/b")
        with self.assertRaisesRegex(ValueError,"escapes"):
            runner.inside(self.root,"../outside")
        with tempfile.TemporaryDirectory() as outside:
            try:
                (self.root/"link").symlink_to(outside,target_is_directory=True)
            except OSError:
                return  # Windows may require special symlink privileges.
            with self.assertRaisesRegex(ValueError,"escapes"):
                runner.inside(self.root,"link/labels.npz")

    def test_failed_xfoil_does_not_create_labels_and_resume_binds_inputs(self):
        dtype = np.dtype([("family_id","i8"),("source_row","i8")])
        provenance = np.array((8,9),dtype=dtype)[()]
        x = np.r_[np.linspace(1,0,11),np.linspace(.1,1,10)]
        y = .05*np.sin(np.linspace(-np.pi,np.pi,21))
        conditions = np.array([.62,.73,1.7e7])
        config = {"panels":240,"iterations":200,"ncrit":9.,"trip_upper":1.,
                  "trip_lower":1.,"timeout_seconds":30.,"cl_tolerance":5e-4}
        output = self.root/"results"
        def failure(*args,**kwargs):
            self.assertEqual(kwargs["cwd"],output/"cases/train/0000000")
            self.assertTrue(Path(kwargs["env"]["HOME"]).is_relative_to(self.root))
            self.assertTrue(Path(kwargs["env"]["TMPDIR"]).is_relative_to(self.root))
            self.assertNotIn("shell",kwargs)
            kwargs["stdout"].write(b"VISCAL: Convergence failed\n")
            return SimpleNamespace(returncode=0)
        with patch.object(runner.subprocess,"run",side_effect=failure):
            result = runner.run_case(self.root,output,Path("/test/xfoil"),x,y,conditions,
                                     provenance,"train",0,config)
        self.assertEqual(result["status"],"failed")
        self.assertFalse((output/"cases/train/0000000/labels.npz").exists())
        changed = conditions.copy();changed[1] += .01
        with self.assertRaisesRegex(ValueError,"different input"):
            runner.run_case(self.root,output,Path("/test/xfoil"),x,y,changed,
                            provenance,"train",0,config)

    def test_pilot_preserves_requested_size_and_covers_extrema(self):
        conditions = np.c_[np.arange(100)[::-1],np.arange(100),np.ones(100)]
        rows = runner.choose_rows(conditions,8)
        self.assertEqual(len(rows),8)
        self.assertIn(0,rows)
        self.assertIn(99,rows)
        self.assertEqual(len(np.unique(rows)),8)

    def test_bounded_submission_does_not_queue_entire_job_list(self):
        submitted = []
        def submit(pool,job):
            submitted.append(job)
            return pool.submit(lambda x:x,job)
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = runner.bounded_results(pool,range(100),submit,4)
            first = next(values)
            self.assertLessEqual(len(submitted),4)
            results = [first,*values]
        self.assertEqual(sorted(results),list(range(100)))

    def test_raw_compaction_verifies_and_preserves_exact_file_bytes(self):
        case = self.root/"case";case.mkdir()
        expected = {}
        for name in runner.RAW_FILES:
            expected[name] = ("raw " + name + "\n").encode()
            (case/name).write_bytes(expected[name])
        result = runner.compact_raw_outputs(self.root,case)
        with zipfile.ZipFile(case/result["archive"]) as zipped:
            self.assertEqual({name:zipped.read(name) for name in zipped.namelist()},expected)
        self.assertFalse(any((case/name).exists() for name in expected))
        self.assertEqual(len(result["files"]),len(expected))

    def test_pack_only_checked_successes_exact_ratio_and_cf_scale(self):
        inputs = self.root/"inputs";inputs.mkdir()
        output = self.root/"results";output.mkdir()
        x = np.r_[np.linspace(1,0,11),np.linspace(.1,1,10)]
        np.save(inputs/"x.npy",x)
        dtype = np.dtype([(name,"i8") for name in (
            "source_row","source_geometry_index","original_id","source_condition_id","family_id","section_index")])
        counts = {"train":12,"test":2}
        for name in ("manifest","COMPLETE"):
            (inputs/(name+".json")).write_text(json.dumps({"status":"complete","counts":counts}))
        config = {"schema":runner.SCHEMA,"unit_test_synthetic":True}
        signature = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        (output/"solver_config.json").write_text(json.dumps(config))
        for split,n in counts.items():
            folder = inputs/split;folder.mkdir()
            geometry = np.zeros((n,1,21),dtype=np.float32)
            conditions = np.tile([.62,.7,1e7],(n,1))
            p = np.zeros(n,dtype=dtype)
            p["family_id"] = 1 if split == "train" else 2
            p["source_row"] = np.arange(n)
            np.save(folder/"geometry.npy",geometry)
            np.save(folder/"conditions.npy",conditions)
            np.save(folder/"provenance.npy",p)
            for row in range(n):
                case = output/"cases"/split/f"{row:07d}"
                case.mkdir(parents=True)
                # Synthetic test fixture: 11 train successes, 1 test success.
                status = "failed" if row == n-1 else "success"
                record = {"split":split,"input_row":row,"status":status,"config_sha256":signature,
                          "input_sha256":runner.input_digest(x,geometry[row,0],conditions[row]),
                          "polar":{"alpha_deg":2.,"cl":.7,"cd":.01,"cm":-.05},
                          "cp_min":-.8,"quality_flags":[]}
                (case/"result.json").write_text(json.dumps(record))
                if status == "success":
                    np.savez_compressed(case/"labels.npz",x=x,conditions=conditions[row],
                                        cp=np.full(21,-.8),cf=np.full(21,.003))
        destination = self.root/"packed"
        manifest = runner.pack_labels(self.root,inputs,output,destination)
        self.assertEqual(manifest["counts"],{"train":9,"test":1})
        self.assertTrue(manifest["exact_9_to_1"])
        train = np.load(destination/"train/data.npy")
        self.assertEqual(train.shape,(9,3,21))
        np.testing.assert_allclose(train[:,1],-.8)
        np.testing.assert_allclose(train[:,2],3.)
        self.assertEqual(len(np.load(destination/"train/unselected_successful_input_rows.npy")),2)
        rows = np.load(destination/"train/index.npy")[:,0]
        self.assertNotIn(11,rows)
        self.assertEqual(manifest["family_overlap_count"],0)
        self.assertEqual(manifest["source_case_states"]["train"]["failed"],1)


if __name__ == "__main__":
    unittest.main()
