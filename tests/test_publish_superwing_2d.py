"""Delivery safety and source/label separation using explicitly synthetic files."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery',ROOT/'build/publish_superwing_2d.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def write_json(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'build')
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.work=self.base/'workspace';self.work.mkdir()
        self.target=self.base/'requested_target'
        self.addCleanup(patch.stopall)
        patch.object(d,'WORKSPACE',self.work).start()
        patch.object(d,'TARGET',self.target).start()
        self.source=self.work/'inputs';self.source.mkdir()
        self.audit=self.work/'audit.json'
        self.x=np.array([1.,.5,0.,.5,1.])
        self.counts={'train':9,'test':1}
        np.save(self.source/'x.npy',self.x)
        np.save(self.source/'family_cst_parameters.npy',np.ones((2,20)))
        np.save(self.source/'geometry_family_ids.npy',np.arange(2))
        np.save(self.source/'test_family_ids.npy',np.array([1]))
        dtype=[(n,'i8') for n in ('source_row','source_geometry_index','original_id','source_condition_id','family_id','section_index')]
        audit={'status':'passed','labels_status':'not_computed','dataset_root':str(self.source),'splits':{}}
        for split,n in self.counts.items():
            folder=self.source/split;folder.mkdir()
            g=np.tile(np.array([0.,-.05,0.,.05,0.]),(n,1,1))
            oc=np.tile([.65,.7,1e7],(n,1))
            p=np.zeros(n,dtype=dtype)
            p['source_row']=np.arange(n)+(100 if split=='test' else 0)
            p['family_id']=p['source_geometry_index']=int(split=='test')
            p['original_id']=100+p['source_geometry_index']
            for key,value in [('geometry',g),('conditions',oc),('provenance',p)]:np.save(folder/(key+'.npy'),value)
            audit['splits'][split]={'samples':n,**{k+'_sha256':d._hash(folder/(k+'.npy')) for k in ('geometry','conditions','provenance')}}
        self.manifest={'status':'complete','dataset_kind':'effective_2d_cfd_inputs_from_superwing',
                       'labels_status':'not_computed','counts':self.counts,'point_count':5}
        write_json(self.source/'manifest.json',self.manifest)
        write_json(self.source/'COMPLETE.json',self.manifest)
        (self.source/'README.txt').write_text('synthetic inputs, no CFD labels')
        write_json(self.audit,audit)

    def pack_fixture(self,selected_counts=None):
        pack=self.work/'pack';pack.mkdir()
        np.save(pack/'x.npy',self.x)
        selected_counts=selected_counts or self.counts
        for split,n in selected_counts.items():
            folder=pack/split;folder.mkdir()
            y=np.load(self.source/split/'geometry.npy')[:n,0]
            oc=np.load(self.source/split/'conditions.npy')[:n]
            p=np.load(self.source/split/'provenance.npy')[:n]
            data=np.stack([y,np.ones_like(y),np.full_like(y,2.)],axis=1)
            np.save(folder/'data.npy',data);np.save(folder/'conditions.npy',oc);np.save(folder/'provenance.npy',p)
            np.save(folder/'selected_input_rows.npy',np.arange(n))
            np.save(folder/'unselected_successful_input_rows.npy',np.array([],dtype='i8'))
            np.save(folder/'index.npy',np.column_stack([np.arange(n)]+[p[name] for name in p.dtype.names]))
            with (folder/'quality.jsonl').open('w') as f:
                for row in range(n):
                    f.write(json.dumps({'split':split,'packed_row':row,'input_row':row,
                        'quality':{'converged':True,'cl':.7,'quality':{'issues':[]},'yplus':{'max':.8}}})+'\n')
        states={s:{'success':n,'failed':0,'running':0,'not_attempted':self.counts[s]-n} for s,n in selected_counts.items()}
        partial=selected_counts!=self.counts
        m={'status':'complete','labels_status':'complete','counts':selected_counts,'point_count':5,
           'solver':'OpenFOAM-v2412-rhoPimpleFoam-LTS+SST','turbulence_model':'kOmegaSST','is_rans':True,
           'data_channels':['effective_y_over_c','Cp','1000*signed_Cf'],
           'conditions_columns':['mach_2d','cl_target_2d','reynolds_2d_actual'],
           'coverage_status':'partial_checkpoint' if partial else 'all_inputs_attempted',
           'checkpoint_only':partial,'all_source_inputs_attempted':not partial,
           'source_input_counts':self.counts,'source_case_states':states,'solver_config':{'cl_tolerance':.001}}
        write_json(pack/'manifest.json',m);write_json(pack/'COMPLETE.json',m)
        (pack/'README.txt').write_text('explicitly synthetic pack fixture')
        return pack

    def test_review_writes_nothing_and_pending_publication_has_no_fake_train(self):
        plan,inputs,pack=d.make_plan(self.source,self.audit)
        self.assertFalse(self.target.exists())
        self.assertEqual(plan['next_requested_compute_stage'],'ten_case_benchmark_then_user_decides')
        result=d.publish(plan,inputs,pack)
        self.assertEqual(result['labels_status'],'not_computed')
        self.assertFalse((self.target/'train').exists())
        self.assertFalse((self.target/'test').exists())
        self.assertTrue((self.target/'inputs/train/geometry.npy').is_file())
        self.assertEqual(d._json(self.target/'status.json')['status'],'inputs_prepared_pending_benchmark')
        self.assertFalse(d._json(self.target/'status.json')['full_source_batch_scheduled'])
        with self.assertRaisesRegex(ValueError,'overwrite'):
            d.make_plan(self.source,self.audit)

    def test_audit_tamper_or_family_leak_rejected_before_any_write(self):
        p=self.source/'train/geometry.npy'
        a=np.load(p);a[0,0,0]=.1;np.save(p,a)
        with self.assertRaisesRegex(ValueError,'SHA256'):
            d.make_plan(self.source,self.audit)
        self.assertFalse(self.target.exists())

    def test_inputs_family_overlap_is_rejected_even_with_matching_file_hash(self):
        path=self.source/'test/provenance.npy'
        values=np.load(path);values['family_id']=0;np.save(path,values)
        audit=d._json(self.audit);audit['splits']['test']['provenance_sha256']=d._hash(path);write_json(self.audit,audit)
        with self.assertRaisesRegex(ValueError,'family leakage'):
            d.make_plan(self.source,self.audit)
        self.assertFalse(self.target.exists())

    def test_verified_pack_lifecycle_preserves_source_identity(self):
        plan,inputs,_=d.make_plan(self.source,self.audit)
        d.publish(plan,inputs)
        pack=self.pack_fixture()
        plan,inputs,pack_info=d.make_plan(self.source,self.audit,pack)
        result=d.publish(plan,inputs,pack_info)
        self.assertEqual(result['labels_status'],'complete')
        self.assertTrue((self.target/'labels_publication.json').is_file())
        self.assertEqual(np.load(self.target/'train/data.npy').shape,(9,3,5))
        self.assertTrue((self.target/'inputs/train/geometry.npy').is_file())
        self.assertEqual(d._json(self.target/'status.json')['coverage_status'],'all_inputs_attempted')

    def test_xfoil_and_unconverged_pack_rejected(self):
        inputs=d.validate_inputs(self.source,self.audit)
        pack=self.pack_fixture()
        m=d._json(pack/'manifest.json');m['solver']='XFOIL';write_json(pack/'manifest.json',m)
        with self.assertRaisesRegex(ValueError,'XFOIL'):
            d.validate_pack(pack,inputs)
        m['solver']='OpenFOAM-v2412-rhoPimpleFoam-LTS+SST';write_json(pack/'manifest.json',m)
        q=pack/'test/quality.jsonl'
        a=json.loads(q.read_text());a['quality']['converged']=False;q.write_text(json.dumps(a)+'\n')
        with self.assertRaisesRegex(ValueError,'Unconverged'):
            d.validate_pack(pack,inputs)

    def test_partial_checkpoint_is_never_called_full_source_completion(self):
        for split in self.counts:
            self.counts[split]*=2
            for key in ('geometry','conditions','provenance'):
                path=self.source/split/(key+'.npy')
                a=np.load(path);np.save(path,np.concatenate((a,a)))
        self.manifest['counts']=self.counts
        write_json(self.source/'manifest.json',self.manifest);write_json(self.source/'COMPLETE.json',self.manifest)
        audit=d._json(self.audit)
        for split,n in self.counts.items():
            audit['splits'][split]['samples']=n
            for key in ('geometry','conditions','provenance'):
                audit['splits'][split][key+'_sha256']=d._hash(self.source/split/(key+'.npy'))
        write_json(self.audit,audit)
        pack=self.pack_fixture({'train':9,'test':1})
        plan,inputs,checked=d.make_plan(self.source,self.audit,pack)
        self.assertEqual(plan['coverage_status'],'partial_checkpoint')
        d.publish(plan,inputs,checked)
        status=d._json(self.target/'status.json')
        self.assertEqual(status['labels_status'],'complete')
        self.assertFalse(status['all_source_inputs_attempted'])
        self.assertFalse(status['full_source_batch_scheduled'])


if __name__=='__main__':unittest.main()
