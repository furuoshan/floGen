"""Regression checks for the paper entry point; no production datasets are written."""
import contextlib
import hashlib
import gc
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# The staging checkout imports the unchanged source repository.
if not (ROOT/"flowvae").is_dir(): sys.path.insert(0,"D:/flogen")
spec=importlib.util.spec_from_file_location("flogen_paper_entry",ROOT/"main.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
torch.set_num_threads(2)


def hashes(root):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(root).iterdir() if p.is_file()}


def wing_fixture(root):
    root.mkdir(parents=True)
    configs=np.zeros((2,55),dtype=np.float64)
    configs[:,0]=[101,509]
    configs[:,1:39]=np.arange(38)[None,:]/100
    configs[:,39:47]=.78+np.arange(8)[None,:]*.001
    configs[:,47:55]=2+np.arange(8)[None,:]*.1
    index=np.zeros((4,12),dtype=np.float64)
    for n,(g,k) in enumerate([(0,0),(0,7),(1,0),(1,7)]):
        index[n]=[g,k,configs[g,47+k],configs[g,39+k],1.6,2,.4,.02,.1,.4,.02,.1]
    centers,vertices=m.synthetic_geometry()
    geom=np.stack((centers,centers.copy()))
    verts=np.stack((vertices,vertices.copy()))
    geom[1,0]+=.1;verts[1,0]+=.1
    data=np.empty((4,3,128,256),dtype=np.float64)
    data[:,0]=1.;data[:,1]=.3;data[:,2]=-.6
    for name,values in [("index",index),("data",data),("geom0",geom),("origingeom",verts)]:
        np.save(root/(name+".npy"),values)
    np.savetxt(root/"configs.dat",configs,header="VARIABLES =",comments="")
    np.savetxt(root/"training_samples_index.txt",np.arange(4),fmt="%d")
    records=[]
    for original in [101,509]:
        row={"running_id":original}
        for k in [0,7]: row[f"{k}_running_info"]={"working_conditions":{"re":2e7}}
        records.append(row)
    (root/"running_info.jsonl").write_text("\n".join(json.dumps(r) for r in records),encoding="utf-8")
    return root


def foil_fixture(root,subset="VT-GU"):
    folder=root/"VT"/subset;folder.mkdir(parents=True)
    index=np.zeros((4,13),dtype=np.float64)
    index[:,0]=[0,0,1,1];index[:,1]=[0,1,0,1]
    index[:,3]=[1.,2.,3.,4.];index[:,5]=.72
    index[:,6]=.6;index[:,9]=[.4,.5,.6,.7];index[:,10]=.02
    index[:,11]=[22121,22121,22098,22098];index[:,12]=.1
    if subset=="VT-DD":
        index[:,3]=[.71,.72,.73,.74];index[:,9]=[2.,3.,4.,5.]
    data=np.empty((4,3,401),dtype=np.float64)
    x=m.clustcos();y=.05*np.sin(np.pi*x)
    data[:,0]=np.r_[-y[::-1],y[1:]]
    data[:,1]=np.sin(np.linspace(0,2*np.pi,401))
    data[:,2]=2
    np.save(folder/"index.npy",index);np.save(folder/"data.npy",data)
    np.savetxt(folder/"traindataindex.txt",np.arange(4),fmt="%d")
    return root


class ConstantFoil(nn.Module):
    def __init__(self):
        super().__init__();self.seen=[]
    def forward(self,y,conditions):
        self.seen.append(conditions.detach().cpu())
        out=torch.ones(len(y),2,401,device=y.device);out[:,1]=2
        return out


class ConstantSLD(nn.Module):
    def forward(self,parameters,conditions):
        return torch.full((len(parameters),128),.5,device=parameters.device)


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_minmax_features_constant_variables_and_extrapolation(self):
        normalizer=m.FeatureNorm(3)
        values=torch.tensor([[1.,10.,4.],[5.,30.,4.]])
        normalizer.fit(values)
        self.assertTrue(torch.equal(normalizer(values),torch.tensor([[0.,0.,0.],[1.,1.,0.]])))
        self.assertTrue(torch.equal(normalizer.inverse(normalizer(values)),values))
        outside=torch.tensor([[9.,50.,4.]])
        self.assertTrue(torch.equal(normalizer(outside),torch.tensor([[2.,2.,0.]])))
        self.assertTrue(torch.equal(normalizer.inverse(normalizer(outside)),outside))

    def test_field_ranges_are_shared_across_samples_and_spatial_points(self):
        normalizer=m.FeatureNorm(2,channel_dim=1)
        values=torch.tensor([[[1.,5.],[10.,30.]],[[3.,7.],[20.,50.]]])
        normalizer.fit(values)
        self.assertTrue(torch.equal(normalizer.minimum,torch.tensor([1.,10.])))
        self.assertTrue(torch.equal(normalizer.scale,torch.tensor([6.,40.])))
        normalized=normalizer(values)
        self.assertTrue(torch.equal(normalized.amin(dim=(0,2)),torch.zeros(2)))
        self.assertTrue(torch.equal(normalized.amax(dim=(0,2)),torch.ones(2)))
        self.assertTrue(torch.allclose(normalizer.inverse(normalized),values))

    def test_normalization_fits_only_training_rows_without_changing_source(self):
        root=foil_fixture(self.root/"foil")
        path=root/"VT"/"VT-GU"/"data.npy"
        values=np.load(path)
        values[0,1]=np.linspace(-2,2,401);values[1,1]=np.linspace(-1,3,401)
        values[0,2]=2;values[1,2]=6
        values[2:]=1000  # Held-out extrema must not enter any normalization range.
        np.save(path,values)
        before=hashes(root/"VT"/"VT-GU")
        data=m.SuperFoilData(root,"VT-GU",2)
        model=m.AirfoilPredictor(m.ModelSpec(condition_dim=2))
        m.fit_normalizers(model,data,np.array([0,1]),"airfoil",batch_size=1)
        self.assertTrue(torch.equal(model.output_normalizer.minimum,torch.tensor([-2.,2.])))
        self.assertTrue(torch.equal(model.output_normalizer.scale,torch.tensor([5.,4.])))
        self.assertAlmostEqual(float(model.normalizer.minimum[1]),.4,places=6)
        self.assertAlmostEqual(float(model.normalizer.scale[1]),.1,places=6)
        self.assertLess(float(model.geometry_normalizer.scale[0]),1)
        self.assertEqual(before,hashes(root/"VT"/"VT-GU"))

    def test_wing_loss_uses_target_ranges_without_residual_minimum(self):
        model=nn.Module();model.residual=nn.Module()
        normalizer=m.FeatureNorm(3,channel_dim=1)
        normalizer.fit_bounds([10.,20.,30.],[12.,24.,38.])
        model.residual.output_normalizer=normalizer
        target=torch.tensor([10.,20.,30.]).reshape(1,3,1,1)
        delta=torch.tensor([2.,4.,8.]).reshape(1,3,1,1)
        error=m.normalize_target(model,target+delta,"wing")-m.normalize_target(model,target,"wing")
        self.assertTrue(torch.equal(error,torch.ones_like(error)))
        self.assertEqual(float(error.square().mean()),1.)
        self.assertTrue(torch.equal(normalizer.inverse_delta(torch.zeros_like(delta)),torch.zeros_like(delta)))
        self.assertTrue(torch.equal(normalizer.inverse_delta(torch.ones_like(delta)),delta))

    def test_airfoil_predictions_are_restored_before_sweep_transforms(self):
        class ZeroNormalizedFoil(nn.Module):
            def forward(self,geometry,code):
                return [torch.zeros(len(geometry),2,401,device=geometry.device)]
        foil=m.AirfoilPredictor(m.ModelSpec())
        foil.network=ZeroNormalizedFoil()
        foil.output_normalizer.fit_bounds([4.,2.],[8.,6.])
        model=m.PhysicsEmbeddedWingPredictor(m.ModelSpec(),foil,ConstantSLD()).eval()
        centers,vertices=m.synthetic_geometry()
        features=m.geometry_features(centers,vertices)
        batch={k:(torch.from_numpy(v.copy())[None] if v.dtype==bool else m.tensor(v)[None])
               for k,v in features.items()}
        batch.update(parameters=torch.zeros(1,38),conditions=torch.tensor([[.8,3.]]),
                     reynolds=torch.tensor([2e7]))
        prior,_=model.prior(batch);cos=float(batch["cos_sweep"])
        self.assertTrue(torch.allclose(prior[:,0],torch.full_like(prior[:,0],4*cos*cos)))
        mask=batch["lower_mask"]
        self.assertTrue(torch.allclose(prior[:,1][mask],torch.full_like(prior[:,1][mask],-.3*cos*cos)))

    def test_minmax_checkpoint_roundtrip_and_legacy_format_rejection(self):
        model=m.SpanwiseLiftPredictor(m.ModelSpec()).eval()
        model.normalizer.fit(torch.stack((torch.arange(40.),torch.arange(40.)+2)))
        model.output_normalizer.fit_bounds([.4],[.9])
        parameters=torch.zeros(2,38);conditions=torch.ones(2,2)
        expected=model(parameters,conditions).detach()
        path=self.root/"minmax.pt"
        m.save_checkpoint(path,"sld",model,m.ModelSpec(),{"training_steps":1},[])
        restored,_,payload=m.load_checkpoint(path,"sld")
        self.assertTrue(torch.equal(restored(parameters,conditions),expected))
        self.assertTrue(torch.equal(restored.normalizer.minimum,model.normalizer.minimum))
        self.assertTrue(torch.equal(restored.output_normalizer.scale,model.output_normalizer.scale))
        self.assertEqual(payload["conventions"]["normalization"],m.NORMALIZATION)
        payload["format"]="physics_embedded_superwing_v1"
        legacy=self.root/"legacy.pt";torch.save(payload,legacy)
        with self.assertRaisesRegex(ValueError,"min-max"):
            m.load_checkpoint(legacy,"sld")

    def test_airfoil_uses_confirmed_reynolds_and_preserves_raw_targets(self):
        root=foil_fixture(self.root/"foil")
        before=hashes(root/"VT"/"VT-GU")
        default=m.SuperFoilData(root,"VT-GU",3)
        np.testing.assert_array_equal(default.condition_matrix()[:,2],np.full(4,20.))
        self.assertEqual(default[0]["conditions"][2].item(),20.)
        self.assertEqual(default.reynolds_info["source"],"user_confirmed_default")
        self.assertEqual(default.reynolds_info["value"],2e7)
        np.testing.assert_array_equal(default[0]["target"].numpy(),default.data[0,1:].astype(np.float32))
        d=m.SuperFoilData(root,"VT-GU",3,reynolds=6e6)
        self.assertEqual(d[0]["target"][1,0].item(),2)
        self.assertAlmostEqual(d[0]["conditions"][2].item(),6)
        self.assertFalse(d.data.flags.writeable)
        self.assertEqual(before,hashes(root/"VT"/"VT-GU"))

    def test_explicit_reynolds_file_overrides_confirmed_default(self):
        root=foil_fixture(self.root/"foil")
        values=np.array([4e6,5e6,6e6,7e6])
        path=self.root/"reynolds.npy";np.save(path,values)
        d=m.SuperFoilData(root,"VT-GU",3,reynolds_file=path)
        np.testing.assert_array_equal(d.condition_matrix()[:,2],values/1e6)
        self.assertEqual(d.reynolds_info["source"],"file_override")
        self.assertEqual(d.reynolds_info["path"],str(path.resolve()))
        self.assertEqual(d[3]["conditions"][2].item(),7.)
        with self.assertRaisesRegex(ValueError,"scalar OR file"):
            m.SuperFoilData(root,"VT-GU",3,reynolds=2e7,reynolds_file=path)
        with self.assertRaisesRegex(ValueError,"positive"):
            m.SuperFoilData(root,"VT-GU",3,reynolds=0)

    def test_dd_uses_target_cl_not_solved_aoa(self):
        root=foil_fixture(self.root/"foil","VT-DD")
        d=m.SuperFoilData(root,"VT-DD",2)
        self.assertAlmostEqual(d[0]["conditions"][0].item(),.71,places=6)
        self.assertAlmostEqual(d[0]["conditions"][1].item(),.6,places=6)
        self.assertNotEqual(d[0]["conditions"][1].item(),2.)

    def test_shape_split_never_mixes_a_geometry_into_validation(self):
        rows=np.arange(8);groups=np.repeat([101,102,103,104],2)
        train,val=m.group_split(rows,groups,fraction=.5,seed=7)
        self.assertFalse(set(groups[train])&set(groups[val]))

    def test_wing_joins_original_id_and_condition_for_reynolds(self):
        root=wing_fixture(self.root/"wing")
        logfile=root/"running_info.jsonl"
        records=[json.loads(row) for row in logfile.read_text().splitlines()]
        records[1]["7_running_info"]["working_conditions"]["re"]=3e7
        logfile.write_text("\n".join(json.dumps(row) for row in records),encoding="utf-8")
        before=hashes(root)
        d=m.SuperWingData(root,target="sld",require_reynolds=True)
        sample=d[3]
        self.assertEqual(d.groups[3],509)
        self.assertEqual(d.shape_ids[3],1)
        self.assertEqual(sample["target"].shape,(128,))
        self.assertEqual(sample["reynolds"].item(),3e7)
        self.assertEqual(before,hashes(root))

    def test_inference_never_reads_reference_field(self):
        root=wing_fixture(self.root/"wing")
        (root/"data.npy").unlink()
        d=m.SuperWingData(root,target="none")
        self.assertIsNone(d.data)
        self.assertNotIn("target",d[0])

    def test_missing_vertices_is_not_replaced_by_guessed_geometry(self):
        root=wing_fixture(self.root/"wing");(root/"origingeom.npy").unlink()
        with self.assertRaisesRegex(ValueError,"origingeom.npy"):
            m.SuperWingData(root,target="none")

    def test_rejects_lfs_pointer_and_inconsistent_mesh(self):
        root=wing_fixture(self.root/"wing")
        geom=np.load(root/"geom0.npy");geom[0,0,0,0]+=.1
        np.save(root/"geom0.npy",geom)
        d=m.SuperWingData(root,target="none")
        with self.assertRaisesRegex(ValueError,"four-vertex"):
            d[0]
        del d
        gc.collect()  # release Windows read-only memory-map handles before replacing the fixture
        (root/"geom0.npy").write_text("version https://git-lfs.github.com/spec/v1\n",encoding="ascii")
        with self.assertRaisesRegex(ValueError,"Git LFS"):
            m.SuperWingData(root,target="none")

    def test_sweep_transforms_and_friction_conventions(self):
        foil=ConstantFoil();model=m.PhysicsEmbeddedWingPredictor(m.ModelSpec(),foil,ConstantSLD()).eval()
        centers,vertices=m.synthetic_geometry()
        f=m.geometry_features(centers,vertices)
        batch={k:(torch.from_numpy(v.copy())[None] if v.dtype==bool else m.tensor(v)[None])
               for k,v in f.items()}
        batch.update(parameters=torch.zeros(1,38),conditions=torch.tensor([[.8,3.]]),
                     reynolds=torch.tensor([2e7]))
        original={k:v.clone() for k,v in batch.items()}
        prior,sld=model.prior(batch);cos=float(batch["cos_sweep"])
        self.assertTrue(torch.allclose(prior[:,0],torch.full_like(prior[:,0],cos*cos)))
        mask=batch["lower_mask"]
        self.assertTrue(torch.allclose(prior[:,1][mask],torch.full_like(prior[:,1][mask],-.3*cos*cos)))
        upper=(~mask)&(~batch["tail_mask"])
        self.assertTrue(torch.allclose(prior[:,1][upper],torch.full_like(prior[:,1][upper],.3*cos*cos)))
        self.assertEqual(prior[0,1,0,-1],0)
        seen=torch.cat(foil.seen)
        self.assertAlmostEqual(float(seen[0,0]),.8*cos,places=6)
        self.assertAlmostEqual(float(seen[0,1]),.5/(cos*cos),places=6)
        self.assertAlmostEqual(float(seen[0,2]),20*float(batch["chord"][0,0])*cos,places=5)
        self.assertTrue(all(torch.equal(v,original[k]) for k,v in batch.items()))

    def test_interpolation_preserves_linear_fields(self):
        xp=torch.linspace(0,1,5)
        values=torch.stack((xp,2*xp))[None]
        q=torch.tensor([[.125,.4,.9]])
        out=m.torch_interp(xp,values,q)
        self.assertTrue(torch.allclose(out[0,0],q[0]))
        self.assertTrue(torch.allclose(out[0,1],2*q[0]))

    def test_real_optimizer_steps_and_freezing(self):
        result=m.smoke_test("cpu")
        self.assertEqual(result["wing_shape"],[1,3,128,256])
        self.assertTrue(result["finite"])
        self.assertTrue(result["frozen_airfoil_unchanged"])
        self.assertEqual(result["normalization"],m.NORMALIZATION)

    def test_checkpoint_rejects_untrained_or_wrong_stage(self):
        model=m.SpanwiseLiftPredictor(m.ModelSpec())
        path=self.root/"lift.pt"
        with self.assertRaisesRegex(ValueError,"untrained"):
            m.save_checkpoint(path,"sld",model,m.ModelSpec(),{"training_steps":0},[])
        m.save_checkpoint(path,"sld",model,m.ModelSpec(),{"training_steps":1},[])
        with self.assertRaisesRegex(ValueError,"Expected wing"):
            m.load_checkpoint(path,"wing")
        with self.assertRaisesRegex(ValueError,"already exists"):
            m.save_checkpoint(path,"sld",model,m.ModelSpec(),{"training_steps":1},[])

    def test_all_training_entry_points_and_dspy_compatible_export(self):
        foil=foil_fixture(self.root/"foil");wing=wing_fixture(self.root/"wing")
        before=hashes(wing);output=self.root/"network"
        air=self.root/"air.pt";sld=self.root/"sld.pt";net=self.root/"wing.pt"
        common=["--epochs","1","--batch-size","1","--max-steps","1","--device","cpu",
                "--validation-fraction","0.5"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(["pretrain-2d","--foil-root",str(foil),"--foil-subset","VT-GU",
                                     "--checkpoint",str(air)]+common),0)
            self.assertEqual(m.main(["train-sld","--wing-root",str(wing),"--checkpoint",str(sld)]+common),0)
            self.assertEqual(m.main(["train-3d","--wing-root",str(wing),
                                     "--airfoil-checkpoint",str(air),"--sld-checkpoint",str(sld),
                                     "--checkpoint",str(net)]+common),0)
            self.assertEqual(m.main(["predict","--checkpoint",str(net),"--input-root",str(wing),
                                     "--output-root",str(output),"--device","cpu"]),0)
        _,air_spec,air_payload=m.load_checkpoint(air,"airfoil")
        self.assertEqual(air_spec.condition_dim,3)
        self.assertEqual(air_payload["training"]["reynolds_input"]["value"],2e7)
        self.assertEqual(air_payload["training"]["reynolds_input"]["source"],"user_confirmed_default")
        data=np.load(output/"data.npy",mmap_mode="r")
        self.assertEqual(data.shape,(4,3,128,256));self.assertTrue(np.isfinite(data).all())
        info=json.loads((output/"prediction_info.json").read_text())
        self.assertEqual(info["status"],"complete")
        self.assertEqual(info["stored_multipliers"],[1,150,300])
        self.assertEqual(before,hashes(wing))
        for name in ["index.npy","configs.dat","geom0.npy","origingeom.npy"]:
            self.assertEqual((wing/name).read_bytes(),(output/name).read_bytes())
        dspy_python=os.environ.get("DSPY_PYTHON")
        if dspy_python:
            code=("import sys,json;sys.path.insert(0,'D:/dspy');"
                  "from superwing_io import compare_superwing_cases;"
                  f"r=compare_superwing_cases({str(output)!r},{str(wing)!r},hotspot_count=2);"
                  "print(json.dumps({'cases':len(r),'channels':r[0].layout.channels}))")
            probe=subprocess.run([dspy_python,"-B","-c",code],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(probe.stdout)["cases"],4)
        model,spec,payload=m.load_checkpoint(net,"wing")
        with self.assertRaisesRegex(ValueError,"must be empty"):
            m.predict_directory(model,spec,payload,wing,output,"cpu")
        with self.assertRaisesRegex(ValueError,"inside input"):
            m.predict_directory(model,spec,payload,wing,wing/"prediction","cpu")


    def test_paper_schedule_endpoints_and_time_scales(self):
        for stage,end,decay in [("airfoil",19,20),("wing",59,62),("sld",79,83)]:
            self.assertAlmostEqual(m.learning_rate_factor(0,stage),.1)
            self.assertAlmostEqual(m.learning_rate_factor(end,stage),1.)
            self.assertAlmostEqual(m.learning_rate_factor(decay,stage),.95)

    def test_sld_checkpoint_cannot_leak_wing_validation_shapes(self):
        m.validate_sld_split({"train_shape_ids":[101]},[509])
        with self.assertRaisesRegex(ValueError,"overlap"):
            m.validate_sld_split({"train_shape_ids":[101]},[101,509])
        with self.assertRaisesRegex(ValueError,"lacks geometry"):
            m.validate_sld_split({},[509])

    def test_cli_default_paths_and_reynolds_overrides(self):
        args=m.build_parser().parse_args(["predict","--checkpoint","trained.pt"])
        self.assertEqual(args.device,"cpu")
        self.assertEqual(args.input_root,Path("D:/data/foilperf/cfd"))
        self.assertEqual(args.output_root,Path("D:/data/foilperf/network"))
        self.assertIsNone(args.reynolds)  # Wing inference must keep its original log values.
        parser=m.build_parser()
        air=parser.parse_args(["pretrain-2d","--foil-subset","VT-GU","--checkpoint","new.pt"])
        self.assertEqual(air.foil_root,Path("D:/Superfoil"))
        self.assertEqual(air.condition_dim,3)
        self.assertIsNone(air.reynolds)  # Fallback is applied only in SuperFoilData.
        file_args=parser.parse_args(["pretrain-2d","--foil-subset","VT-GU",
                                     "--checkpoint","new.pt","--reynolds-file","actual.npy"])
        self.assertIsNone(file_args.reynolds)
        self.assertEqual(file_args.reynolds_file,Path("actual.npy"))
        self.assertEqual(parser.parse_args(["inspect"]).foil_root,Path("D:/Superfoil"))
        wing=parser.parse_args(["train-3d","--airfoil-checkpoint","air.pt",
                                "--sld-checkpoint","sld.pt","--checkpoint","wing.pt"])
        self.assertIsNone(wing.reynolds)


if __name__=="__main__":unittest.main()
