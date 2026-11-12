"""CST alias isolation and exact-count guards for the new full-library split."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import superfoil_train_test_split as s


class FakeJoint:
    def __init__(self):
        self.groups=np.r_[np.repeat(10,3),np.repeat(11,3),
                          np.repeat(np.arange(20,29),10),np.repeat(29,8)]
        self.offsets=[0,40,75,len(self.groups)]
        self.datasets=[SimpleNamespace(subset=name) for name in ('VT-BUF','VT-DD','VT-GU')]
        # These are deliberately irrelevant to the newly authorized split.
        self.protected_test_ids=np.unique(self.groups)

    def __len__(self): return len(self.groups)

    def __getitem__(self,row): raise AssertionError('Splitting must not mutate or construct data samples')

    def counts(self,rows):
        return {d.subset:int(np.count_nonzero((rows>=self.offsets[i])&(rows<self.offsets[i+1])))
                for i,d in enumerate(self.datasets)}


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.dataset=FakeJoint()
        ids=np.unique(self.dataset.groups)
        cst=np.zeros((len(ids),20))
        cst[:,0]=ids;cst[ids==11,0]=10
        self.meta=self.folder/'meta.npz'
        np.savez(self.meta,run_ids=ids,cst=cst,tmax=np.full(len(ids),.12))

    def test_exact_ratio_partitions_every_source_and_merges_cst_aliases(self):
        before=hashlib.sha256(self.meta.read_bytes()).hexdigest()
        groups_before=self.dataset.groups.copy()
        result=s.build_split(self.dataset,self.meta,42)
        tr,te,ex=(result[k] for k in ('train_rows','test_rows','excluded_rows'))
        self.assertEqual((len(tr),len(te),len(ex)),(90,10,4))
        np.testing.assert_array_equal(np.sort(np.r_[tr,te,ex]),np.arange(104))
        g=result['family_ids']
        self.assertFalse(set(g[tr])&set(g[te]))
        self.assertFalse(set(g[ex])&set(g[te]))
        alias_rows=np.flatnonzero(np.isin(self.dataset.groups,[10,11]))
        self.assertEqual(len(np.unique(g[alias_rows])),1)
        self.assertTrue(all(not d['redundancy_claimed'] for d in result['manifest']['excluded_record_details']))
        np.testing.assert_array_equal(self.dataset.groups,groups_before)
        self.assertEqual(hashlib.sha256(self.meta.read_bytes()).hexdigest(),before)
        for family in np.unique(g[ex]):self.assertIn(family,g[tr])

    def test_seed_reproduces_both_family_selection_and_unused_indices(self):
        first=s.build_split(self.dataset,self.meta,42)
        second=s.build_split(self.dataset,self.meta,42)
        for key in ('train_rows','test_rows','excluded_rows','family_ids','test_family_ids'):
            np.testing.assert_array_equal(first[key],second[key])

    def test_two_equal_count_swaps_can_balance_a_seeded_test_set(self):
        weights=np.array([2,4,6,8,10,12])
        test,info=s.choose_test_families(weights,22,42,test_family_count=2)
        self.assertEqual(weights[test].sum(),22)
        self.assertEqual(len(test),2)
        self.assertEqual(len(info['family_swaps']),2)

    def test_subset_sum_fallback_never_splits_a_family(self):
        weights=np.array([3,4,6,8])
        test,info=s.choose_test_families(weights,10,42,test_family_count=1)
        self.assertEqual(weights[test].sum(),10)
        self.assertEqual(len(test),2)
        self.assertEqual(info['method'],'seeded whole-family subset sum')
        with self.assertRaises(ValueError):s.choose_test_families(np.array([5,5]),1)

    def test_missing_meta_is_rejected_and_existing_split_is_not_partially_overwritten(self):
        ids=np.unique(self.dataset.groups)
        with np.load(self.meta) as original:cst=original['cst'].copy()
        bad=self.folder/'bad.npz';np.savez(bad,run_ids=ids[:-1],cst=cst[:-1])
        with self.assertRaises(ValueError):s.build_split(self.dataset,bad)
        result=s.build_split(self.dataset,self.meta)
        dest=self.folder/'split';dest.mkdir()
        (dest/'test_rows.npy').write_bytes(b'preserve')
        with self.assertRaises(ValueError):s.save_split(result,dest)
        self.assertEqual((dest/'test_rows.npy').read_bytes(),b'preserve')
        self.assertEqual([p.name for p in dest.iterdir()],['test_rows.npy'])


if __name__=='__main__':unittest.main()
