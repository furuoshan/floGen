"""Physical startup and anisotropic mesh acceptance guards."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_superwing_openfoam import alpha_initial,assess_mesh_log


VALID="""Boundary definition OK.
Cell to face addressing OK.
Upper triangular ordering OK.
Face vertices OK.
Number of regions: 1 (OK).
Cell volumes OK.
Face pyramids OK.
Non-orthogonality check OK.
 ***High aspect ratio cells found, Max aspect ratio: 4063.4435, number of cells 17
Mesh non-orthogonality Max: 38.0328 average: 5.69
Max skewness = 2.46336 OK.
Cell determinant (wellposedness) : minimum: 1.21444e-7 average: .112
 ***Cells with small determinant (< 0.001) found, number of cells: 9182
Face interpolation weight : minimum: .091183 average: .47489
Face volume ratio : minimum: .103614 average: .8987
Failed 2 mesh checks.
"""


class OpenFoamRunGuards(unittest.TestCase):
    def test_resolved_wall_anisotropy_does_not_excuse_bad_face_coupling(self):
        accepted=assess_mesh_log(VALID)
        self.assertTrue(accepted['accepted'])
        self.assertEqual(len(accepted['wall_anisotropy_warnings']),2)
        for text in (VALID.replace('.091183','.0018'),VALID.replace('.103614','.0018'),
                     VALID.replace('Cell volumes OK.','Negative volume!'),
                     VALID+'\n ***Faces incorrectly oriented found\n'):
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError):
                assess_mesh_log(text)

    def test_thin_parabolic_camber_has_known_zero_lift_angle(self):
        half=np.linspace(0,1,501)
        x=np.r_[half[::-1],half[1:]]
        camber=4*.02*x*(1-x)
        thickness=.05*np.sin(np.pi*x)
        y=camber+np.r_[-thickness[:501],thickness[501:]]
        self.assertAlmostEqual(alpha_initial(x,y,0),np.degrees(-.04),delta=.02)


if __name__=='__main__': unittest.main()
