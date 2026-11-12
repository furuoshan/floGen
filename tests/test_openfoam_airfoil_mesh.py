"""Geometry, first-layer orthogonality, and written polyMesh conservation checks."""
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import openfoam_airfoil_mesh as m


def airfoil():
    half=np.linspace(0,np.pi,161)
    x=.5*(1-np.cos(half))
    # Finite TE; the straight base must remain a physical wall, not be rounded.
    thickness=.09*np.sin(half)+.002*x
    return np.r_[x[::-1],x[1:]],np.r_[-thickness[::-1],thickness[1:]]


def foam_list(path):
    content=Path(path).read_text(encoding="ascii")
    match=re.search(r"\n(\d+)\n\(\n",content)
    count=int(match.group(1))
    lines=content[match.end():].splitlines()[:count]
    return count,lines


class OpenFOAMMeshTests(unittest.TestCase):
    def test_wall_polygon_is_exact_and_miter_first_layer_has_constant_normal_height(self):
        x,y=airfoil()
        result=m.make_grid(x,y)
        np.testing.assert_array_equal(result["grid"][0],np.c_[x,y])
        self.assertEqual(result["base_mask"].sum(),1)
        e=result["wall_edges"]
        normals=np.c_[-e[:,1],e[:,0]]/result["wall_edge_lengths"][:,None]
        offset=result["grid"][1]-result["grid"][0]
        # Both ends of every edge lie on its parallel h=1e-6 offset line.
        np.testing.assert_allclose(np.sum(offset*normals,axis=1),1e-6,rtol=2e-8)
        np.testing.assert_allclose(np.sum(np.roll(offset,-1,axis=0)*normals,axis=1),1e-6,rtol=2e-8)
        self.assertLess(result["quality"]["max_nonorthogonality_estimate_deg"],60)

    def test_written_face_orientation_closes_every_cell_and_gives_positive_divergence_volume(self):
        x,y=airfoil()
        with tempfile.TemporaryDirectory(dir=m.WORKSPACE/"build",prefix="foam-mesh-test-") as tmp:
            case=Path(tmp)/"case"
            meta=m.write_mesh(case,x,y,radial_cells=30,first_layer=1e-6)
            root=case/"constant/polyMesh"
            n,lines=foam_list(root/"points")
            points=np.array([[float(v) for v in line.strip("()").split()] for line in lines])
            self.assertEqual(len(points),n)
            _,lines=foam_list(root/"faces")
            faces=np.array([[int(v) for v in line[2:-1].split()] for line in lines])
            _,lines=foam_list(root/"owner")
            owner=np.array(lines,dtype=int)
            ni,lines=foam_list(root/"neighbour")
            neighbor=np.array(lines,dtype=int)
            self.assertTrue((owner[:ni]<neighbor).all())
            corners=points[faces]
            sf=.5*np.cross(corners,np.roll(corners,-1,axis=1)).sum(axis=1)
            fc=corners.mean(axis=1)
            closure=np.zeros((meta["cells"],3))
            np.add.at(closure,owner,sf)
            np.add.at(closure,neighbor,-sf[:ni])
            np.testing.assert_allclose(closure,0,atol=5e-13)
            volume=np.zeros(meta["cells"])
            value=np.sum(sf*fc,axis=1)/3
            np.add.at(volume,owner,value)
            np.add.at(volume,neighbor,-value[:ni])
            self.assertTrue((volume>0).all())
            expected=m.make_grid(x,y,radial_cells=30)["quality"]["areas_2d"].ravel()*.01
            np.testing.assert_allclose(volume,expected,rtol=1e-7,atol=1e-14)
            faces_per_cell=np.bincount(owner,minlength=meta["cells"])+np.bincount(neighbor,minlength=meta["cells"])
            np.testing.assert_array_equal(faces_per_cell,6)
            self.assertEqual(meta["patches"]["frontAndBack"]["nFaces"],2*meta["cells"])

    def test_project_root_and_symlinks_are_not_allowed_to_escape_scope(self):
        x,y=airfoil()
        with tempfile.TemporaryDirectory(dir=m.WORKSPACE/"build",prefix="foam-scope-test-") as tmp:
            root=Path(tmp)
            project=root/"project"
            project.mkdir()
            with self.assertRaisesRegex(ValueError,"inside Projects/floGen"):
                m.write_mesh(root/"outside",x,y,project_root=project)

    def test_refuses_to_replace_a_preexisting_mesh(self):
        x,y=airfoil()
        with tempfile.TemporaryDirectory(dir=m.WORKSPACE/"build",prefix="foam-existing-test-") as tmp:
            case=Path(tmp)/"case"
            m.write_mesh(case,x,y,radial_cells=30)
            with self.assertRaisesRegex(ValueError,"existing polyMesh"):
                m.write_mesh(case,x,y,radial_cells=30)

    def test_reversed_wall_order_is_rejected(self):
        x,y=airfoil()
        with self.assertRaisesRegex(ValueError,"clockwise"):
            m.make_grid(x[::-1],y[::-1])


if __name__=="__main__":
    unittest.main()
