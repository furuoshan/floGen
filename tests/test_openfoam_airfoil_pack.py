"""Explicit synthetic RANS records test packing; no native CFD is claimed."""
from pathlib import Path
import json
import tempfile
import unittest
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import openfoam_airfoil_pack as packing


class RansPackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def fixture(self,counts=None,failed_last_train=True,unattempted_last_test=True,solver="rhoSimpleFoam"):
        counts = counts or {"train":12,"test":2}
        inputs,results = self.root/"inputs",self.root/"results"
        inputs.mkdir();results.mkdir()
        x = np.r_[np.linspace(1,0,11),np.linspace(.1,1,10)]
        np.save(inputs/"x.npy",x)
        for name in ("manifest","COMPLETE"):
            (inputs/(name+".json")).write_text(json.dumps({"status":"complete","counts":counts}))
        config = {"cl_tolerance":.001,"synthetic_unit_test":True}
        if solver == "rhoPimpleFoam":
            config.update(solver=solver,time_stepping="localEuler")
        (results/"config.json").write_text(json.dumps(config))
        dtype = np.dtype([(name,"i8") for name in packing.INDEX_COLUMNS[1:]]+[("temperature_kelvin","f8")])
        for split,n in counts.items():
            folder = inputs/split;folder.mkdir()
            y = np.zeros((n,1,len(x)),dtype=np.float32)
            y[:,0] = np.linspace(-.05,.05,len(x))
            oc = np.tile([.65,.75,1.5e7],(n,1))
            p = np.zeros(n,dtype=dtype)
            p["source_row"] = np.arange(n)
            p["source_geometry_index"] = np.arange(n)
            p["original_id"] = np.arange(n)+100
            p["family_id"] = 1 if split == "train" else 2
            p["temperature_kelvin"] = 300.
            for name,value in (("geometry",y),("conditions",oc),("provenance",p)):
                np.save(folder/(name+".npy"),value)
            for row in range(n):
                if split == "test" and row == n-1 and unattempted_last_test:
                    continue
                case = results/"cases"/split/f"{row:07d}"
                case.mkdir(parents=True)
                status = "failed" if split == "train" and row == n-1 and failed_last_train else "success"
                record = {"status":status,"split":split,"input_row":row,
                          "solver":packing._solver_identity(config)[0],"turbulence_model":"kOmegaSST",
                          "conditions":oc[row].tolist(),"cl":.7501,"cd":.012,
                          "provenance":{name:p[row][name].item() for name in p.dtype.names},
                          "fingerprint":packing._fingerprint(x,np.array(y[row,0],dtype=float),oc[row],p[row],config)}
                (case/"result.json").write_text(json.dumps(record))
                if status == "success":
                    np.savez_compressed(case/"labels.npz",geometry=y[row,0],conditions=oc[row],
                                        provenance=p[row],labels=np.stack((np.full(len(x),-.8),np.full(len(x),.003))))
                    (case/"quality.json").write_text(json.dumps({"converged":True,"cl":.7501,"cd":.012,
                                                                  "yplus":{"max":.7,"mean":.3}}))
        return inputs,results

    def test_success_only_scale_ratio_provenance_and_partial_coverage(self):
        inputs,results = self.fixture()
        destination = self.root/"packed"
        manifest = packing.pack_dataset(self.root,inputs,results,destination)
        self.assertEqual(manifest["counts"],{"train":9,"test":1})
        self.assertEqual(manifest["coverage_status"],"partial_checkpoint")
        self.assertTrue(manifest["checkpoint_only"])
        self.assertFalse(manifest["all_source_inputs_attempted"])
        self.assertFalse(manifest["all_source_inputs_succeeded"])
        self.assertEqual(manifest["failed_source_case_count"],1)
        self.assertEqual(manifest["source_case_states"]["test"]["not_attempted"],1)
        data = np.load(destination/"train/data.npy")
        self.assertEqual(data.shape,(9,3,21))
        np.testing.assert_allclose(data[:,1],-.8)
        np.testing.assert_allclose(data[:,2],3.)
        selected = np.load(destination/"train/selected_input_rows.npy")
        self.assertNotIn(11,selected)
        self.assertEqual(len(np.load(destination/"train/unselected_successful_input_rows.npy")),2)
        source = np.load(inputs/"train/provenance.npy")
        np.testing.assert_array_equal(np.load(destination/"train/provenance.npy"),source[selected])
        np.testing.assert_array_equal(np.load(destination/"train/index.npy")[:,0],selected)
        self.assertEqual(manifest["solver"],packing.SOLVER)
        self.assertEqual(manifest["family_overlap_count"],0)

    def test_xfoil_or_wrong_turbulence_is_rejected(self):
        inputs,results = self.fixture()
        path = results/"cases/train/0000000/result.json"
        record = json.loads(path.read_text());record["solver"]="XFOIL"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,"mixed/XFOIL"):
            packing.pack_dataset(self.root,inputs,results,self.root/"mixed")
        self.assertFalse((self.root/"mixed").exists())

    def test_changed_npz_cannot_pass_current_source_identity(self):
        inputs,results = self.fixture({"train":9,"test":1},False,False)
        path = results/"cases/train/0000000/labels.npz"
        with np.load(path,allow_pickle=False) as existing:
            fields = {name:existing[name] for name in existing.files}
        fields["conditions"] = fields["conditions"].copy()
        fields["conditions"][0] += .01
        np.savez_compressed(path,**fields)
        with self.assertRaisesRegex(ValueError,"NPZ conditions differ"):
            packing.pack_dataset(self.root,inputs,results,self.root/"changed")
        self.assertFalse((self.root/"changed/COMPLETE.json").exists())

    def test_complete_coverage_does_not_mean_every_source_succeeded(self):
        inputs,results = self.fixture({"train":10,"test":1},True,False)
        manifest = packing.pack_dataset(self.root,inputs,results,self.root/"complete")
        self.assertTrue(manifest["all_source_inputs_attempted"])
        self.assertFalse(manifest["all_source_inputs_succeeded"])
        self.assertFalse(manifest["checkpoint_only"])
        self.assertEqual(manifest["failed_source_case_count"],1)
        with self.assertRaisesRegex(ValueError,"escapes"):
            packing.pack_dataset(self.root,inputs,results,self.root.parent/"outside")

    def test_lts_identity_is_explicit_and_cannot_mix_simple_records(self):
        inputs,results = self.fixture({"train":9,"test":1},False,False,solver="rhoPimpleFoam")
        metadata = packing.pack_dataset(self.root,inputs,results,self.root/"lts")
        self.assertEqual(metadata["solver"],"OpenFOAM-v2412-rhoPimpleFoam-LTS+SST")
        self.assertEqual(metadata["time_stepping"],"localEuler")
        self.assertTrue(metadata["is_pseudo_time"])
        path = results/"cases/train/0000000/result.json"
        record = json.loads(path.read_text());record["solver"]="OpenFOAM-v2412-rhoSimpleFoam"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,"mixed/XFOIL"):
            packing.pack_dataset(self.root,inputs,results,self.root/"mixed_lts")


if __name__ == "__main__":
    unittest.main()
