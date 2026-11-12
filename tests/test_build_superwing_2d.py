"""Physical invariants and split identity for effective 2-D input construction."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_superwing_2d as b


def wing(twist_deg=7.0):
    z=np.linspace(0.0,4.0,129)
    qx=np.where(z <= 2.0,0.2*z,0.4+0.7*(z-2.0))
    theta=np.deg2rad(twist_deg)
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    lower=np.linspace(1.0,0.0,128)
    upper=np.linspace(0.0,1.0,129)[1:]
    x=np.r_[lower,upper]
    y=np.r_[-0.06*np.sin(np.pi*lower)-0.002*lower,
             0.06*np.sin(np.pi*upper)+0.002*upper]
    xy=rotation @ np.stack((x-0.25,y))
    v=np.empty((3,129,257),dtype=np.float64)
    for j in range(129):
        v[:2,j,:256]=xy+np.array([qx[j],0.1*z[j]])[:,None]
        v[2,j]=z[j]
        v[:2,j,256]=v[:2,j,0]
    c=0.25*(v[:,:-1,:-1]+v[:,:-1,1:]+v[:,1:,:-1]+v[:,1:,1:])
    return c,v


class EffectiveInputsTests(unittest.TestCase):
    def test_local_sweep_resolves_two_segments_without_smearing_kink(self):
        c,v=wing()
        f=b.extract_sections(c,v)
        np.testing.assert_allclose(f['cos_sweep'][:64],1/np.hypot(1,0.2),atol=1e-12)
        np.testing.assert_allclose(f['cos_sweep'][64:],1/np.hypot(1,0.7),atol=1e-12)
        np.testing.assert_allclose(f['chord'],1.0,atol=1e-12)
        np.testing.assert_allclose(f['chord_angle_xy_deg'],7.0,atol=1e-12)

    def test_detwisting_is_rotation_invariant_and_keeps_finite_trailing_edge(self):
        c0,v0=wing(0.0)
        c1,v1=wing(12.0)
        f0=b.extract_sections(c0,v0)
        f1=b.extract_sections(c1,v1)
        np.testing.assert_allclose(f0['foil_y'],f1['foil_y'],atol=1e-12)
        self.assertEqual(f1['foil_y'].shape,(128,321))
        self.assertLess(f1['foil_y'][0,0],0)
        self.assertGreater(f1['foil_y'][0,-1],0)
        np.testing.assert_allclose(f1['foil_y'][:,160],0,atol=1e-12)

    def test_misaligned_cell_centers_are_rejected(self):
        c,v=wing()
        c[0,10,10]+=0.001
        with self.assertRaisesRegex(ValueError,'four-vertex'):
            b.extract_sections(c,v)

    def test_same_cst_family_never_crosses_train_test(self):
        configs=np.zeros((20,55))
        # Every pair of distinct wings has the same baseline-CST family.
        configs[:,18:38]=np.repeat(np.arange(10),2)[:,None]
        index=np.zeros((40,12))
        index[:,0]=np.repeat(np.arange(20),2)
        split=b.make_family_split(configs,index)
        for family in np.unique(split['row_family']):
            self.assertEqual(len(np.unique(split['is_test'][split['row_family']==family])),1)

    def test_grid_and_primary_stations_have_unambiguous_orientation(self):
        x,half=b.chordwise_grid()
        self.assertEqual(len(x),321)
        self.assertEqual((x[0],x[160],x[-1]),(1.0,0.0,1.0))
        self.assertTrue((np.diff(x[:161])<0).all())
        self.assertTrue((np.diff(x[160:])>0).all())
        js=b.primary_sections()
        self.assertEqual(js.tolist(),[5,16,26,37,48,58,69,80,90,101,112,122])
        self.assertEqual(len(np.unique(js)),12)

    def test_canonical_extra_test_sections_are_distinct_and_exactly_balance_ratio(self):
        flags=np.zeros(28856,dtype=bool)
        flags[-2885:]=True
        split={'canonical':True,'is_test':flags}
        extras=b.ratio_extra_sections(split,np.arange(28856),b.primary_sections())
        self.assertEqual(len(extras),8)
        for row,j in extras.items():
            self.assertTrue(flags[row])
            self.assertNotIn(j,b.primary_sections())
        self.assertEqual(25971*12,9*(2885*12+len(extras)))


if __name__=='__main__':
    unittest.main()
