"""Physical/dimensional/configuration tests, not native OpenFOAM validation."""
from pathlib import Path
import tempfile
import unittest
import math
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import openfoam_airfoil_case as casegen


class OpenfoamCaseTests(unittest.TestCase):
    def test_prescribed_mach_reynolds_and_ideal_gas_are_self_consistent(self):
        for mach,re,temp in ((.589,2.9e6,300.),(.845,2.6e7,300.),(.7,1.7e7,250.)):
            with self.subTest(mach=mach,re=re,temp=temp):
                flow = casegen.flow_conditions(mach,re,temp,alpha_deg=5.)
                self.assertAlmostEqual(flow.speed/math.sqrt(flow.gamma*flow.gas_constant*temp),mach)
                self.assertAlmostEqual(flow.density*flow.speed*flow.chord_metres/flow.dynamic_viscosity/re,1.)
                self.assertAlmostEqual(flow.pressure/(flow.density*flow.gas_constant*temp),1.)
                np.testing.assert_allclose(np.linalg.norm(flow.velocity),flow.speed)
                self.assertAlmostEqual(np.dot(flow.lift_direction,flow.drag_direction),0.)
        flow300 = casegen.flow_conditions(.7,1e7)
        self.assertAlmostEqual(flow300.dynamic_viscosity,1.846e-5,places=14)
        self.assertAlmostEqual(flow300.Cp/(flow300.Cp-flow300.gas_constant),1.4)

    def test_compressible_solver_low_re_walls_and_force_normalization(self):
        flow = casegen.flow_conditions(.65,1.5e7,alpha_deg=10,span=.02)
        texts = casegen.case_texts(flow)
        self.assertIn("application rhoSimpleFoam;",texts["system/controlDict"])
        self.assertIn("transonic yes;",texts["system/fvSolution"])
        self.assertIn("transport sutherland;",texts["constant/thermophysicalProperties"])
        self.assertIn("equationOfState perfectGas;",texts["constant/thermophysicalProperties"])
        self.assertIn("RASModel kOmegaSST;",texts["constant/turbulenceProperties"])
        self.assertIn("Aref 0.02;",texts["system/controlDict"])
        self.assertIn("rho rho;",texts["system/controlDict"])
        self.assertIn("type noSlip;",texts["0/U"])
        self.assertIn("type freestreamVelocity;",texts["0/U"])
        self.assertIn("type freestreamPressure;",texts["0/p"])
        self.assertIn("nutLowReWallFunction",texts["0/nut"])
        self.assertIn("omegaWallFunction",texts["0/omega"])
        self.assertIn("type zeroGradient;",texts["0/T"])
        self.assertEqual(casegen.parse_dimensions(texts["0/p"]),(1,-1,-2,0,0,0,0))
        self.assertEqual(casegen.parse_dimensions(texts["0/alphat"]),(1,-1,-1,0,0,0,0))
        self.assertEqual(casegen.parse_dimensions(texts["0/pPotential"]),(0,2,-2,0,0,0,0))
        self.assertEqual(casegen.parse_dimensions(texts["0/Phi"]),(0,2,-1,0,0,0,0))
        self.assertIn("type fixedValue;\n        value uniform 0;",texts["0/pPotential"])
        self.assertNotIn("type fixedValue;",texts["0/Phi"])
        self.assertIn("potentialFlow",texts["system/fvSolution"])
        self.assertIn("PhiRefCell 0;",texts["system/fvSolution"])
        high_order = casegen.case_texts(flow,convection="limitedLinear")["system/fvSchemes"]
        self.assertIn("div(phi,U) bounded Gauss limitedLinear 1;",high_order)
        self.assertIn("div(phi,h) bounded Gauss limitedLinear 1;",high_order)
        self.assertIn("div(phi,k) bounded Gauss upwind;",high_order)
        self.assertIn("div(phi,omega) bounded Gauss upwind;",high_order)

    def test_wall_shear_units_and_wall_on_fluid_sign(self):
        flow = casegen.flow_conditions(.65,1e7)
        q = flow.dynamic_pressure
        p = np.array([flow.pressure+.3*q,flow.pressure-.7*q])
        # Both walls experience body drag toward +x. Dataset tangents point -x
        # on lower TE->LE, and +x on upper LE->TE. OpenFOAM WSS is wall-on-fluid.
        tau = np.array([[-.003*q,0,0],[-.004*q,0,0]])
        tangent = np.array([[-1.,0,0],[1.,0,0]])
        cp,cf = casegen.surface_coefficients(p,tau,tangent,flow,
                                             (1,-1,-2,0,0,0,0),(1,-1,-2,0,0,0,0))
        np.testing.assert_allclose(cp,[.3,-.7])
        np.testing.assert_allclose(cf,[-.003,.004])
        rho_wall = np.array([.7,1.3])
        cp2,cf2 = casegen.surface_coefficients(p,tau/rho_wall[:,None],tangent,flow,
                                               (1,-1,-2,0,0,0,0),(0,2,-2,0,0,0,0),rho_wall)
        np.testing.assert_allclose(cf2,cf)
        with self.assertRaisesRegex(ValueError,"local wall density"):
            casegen.surface_coefficients(p,tau,tangent,flow,
                                           (1,-1,-2,0,0,0,0),(0,2,-2,0,0,0,0))
        with self.assertRaisesRegex(ValueError,"Pressure must"):
            casegen.surface_coefficients(p,tau,tangent,flow,
                                           (0,2,-2,0,0,0,0),(1,-1,-2,0,0,0,0))

    def test_lts_is_explicit_compressible_pimple_not_changed_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            case = root/"lts"
            metadata = casegen.write_lts_case(case,.62,1.7e7,project_root=root,end_time=100,
                                               max_co=.2,max_local_dt=.001)
            self.assertEqual(metadata["solver"],"rhoPimpleFoam")
            self.assertEqual(metadata["mach"],.62)
            self.assertEqual(metadata["reynolds"],1.7e7)
            self.assertEqual(metadata["time_scheme"],"localEuler")
            self.assertIn("application rhoPimpleFoam;",(case/"system/controlDict").read_text())
            self.assertIn("default localEuler;",(case/"system/fvSchemes").read_text())
            solution = (case/"system/fvSolution").read_text()
            self.assertIn("PIMPLE\n{",solution)
            self.assertNotIn("SIMPLE\n{",solution)
            self.assertIn("maxCo 0.2;",solution)
            self.assertIn("rDeltaTDampingCoeff 0.2;",solution)
            self.assertNotIn("alphaTemp",solution)
            standard = root/"simple"
            self.assertEqual(casegen.write_case(standard,.62,1.7e7,project_root=root)["solver"],"rhoSimpleFoam")

    def test_generator_preserves_mesh_and_existing_solution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            case = root/"pilot"
            mesh = case/"constant/polyMesh";mesh.mkdir(parents=True)
            (mesh/"boundary").write_text("test mesh sentinel")
            latest = case/"100";latest.mkdir()
            (latest/"U").write_text("solution sentinel")
            result = casegen.write_case(case,.62,1.7e7,project_root=root,end_time=300)
            initial = (case/"0/U").read_text()
            result2 = casegen.write_case(case,.62,1.7e7,alpha_deg=4,project_root=root,
                                         start_from="latestTime",end_time=600)
            self.assertEqual((mesh/"boundary").read_text(),"test mesh sentinel")
            self.assertEqual((latest/"U").read_text(),"solution sentinel")
            self.assertEqual((case/"0/U").read_text(),initial)
            self.assertIn("startFrom latestTime;",(case/"system/controlDict").read_text())
            self.assertNotEqual(result["velocity"],result2["velocity"])
            self.assertEqual(result2["restart_boundary_updates"][0]["entry"],
                             "boundaryField.farfield.freestreamValue")
            self.assertFalse(result2["native_solver_verified"])
            self.assertIn("p_inf",result2)
            potential = result2["potential_initialization"]
            self.assertEqual(potential["command"],"potentialFoam -pName pPotential -initialiseUBCs -writePhi")
            self.assertIn("type fixedValue",potential["temporarily_set_U_farfield"])
            self.assertIn("type freestreamVelocity",potential["restore_U_farfield"])
            self.assertIn("-writep",potential["must_not_use_options"])
            with self.assertRaisesRegex(ValueError,"escapes"):
                casegen.write_case(root.parent/"escape",.62,1.7e7,project_root=root)


if __name__ == "__main__":
    unittest.main()
