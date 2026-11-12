"""Known wall-traction and native ASCII fixtures, not fabricated CFD validation."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openfoam_airfoil_post as post


def write_field(path, values, dimensions, vector=False, uniform=False):
    dims = " ".join(map(str, dimensions))
    if uniform:
        value = "("+" ".join(map(str, values))+")" if vector else str(values)
        text = "uniform "+value
    else:
        values = np.asarray(values)
        content = "\n".join("("+" ".join(map(str, row))+")" if vector else str(row) for row in values)
        text = "nonuniform List<"+("vector" if vector else "scalar")+">\n"+str(len(values))+"\n(\n"+content+"\n)"
    path.write_text("FoamFile { version 2.0; format ascii; class vol"+("Vector" if vector else "Scalar")+
                    "Field; }\ndimensions ["+dims+"];\ninternalField uniform "+("(0 0 0)" if vector else "0")+
                    ";\nboundaryField { farfield { type zeroGradient; } airfoil { type calculated; value "+text+"; } }\n")


class OpenFOAMPostTests(unittest.TestCase):
    def test_native_ascii_uniform_and_nonuniform_count_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"p"
            write_field(p, 101000, post.PRESSURE, uniform=True)
            result = post.read_boundary_field(p, count=3)
            np.testing.assert_array_equal(result["values"], [101000]*3)
            write_field(p, [1,2,3], post.DIMENSIONLESS)
            np.testing.assert_array_equal(post.read_boundary_field(p,count=3)["values"], [1,2,3])
            with self.assertRaises(ValueError):
                post.read_boundary_field(p,count=4)
            p.write_text(p.read_text().replace("format ascii", "format binary"))
            with self.assertRaises(ValueError):
                post.read_boundary_field(p,count=3)

    def test_flat_wall_body_traction_and_local_density_conversion(self):
        flow={"p_inf":100000,"rho_inf":1.2,"U_inf":100}
        tangent=np.array([[-1,0,0],[1,0,0]])
        # Native output is wall-on-fluid: -x for attached +x motion.
        tau=np.array([[-2,0,0],[-2,0,0]])
        result=post.wall_coefficients([100000,100600],post.PRESSURE,tau,post.PRESSURE,tangent,flow)
        np.testing.assert_allclose(result["cp"],[0,.1])
        np.testing.assert_allclose(result["cf"],[-2/6000,2/6000])
        # Different actual wall densities must lead to different stresses.
        result=post.wall_coefficients([100000,100000],post.PRESSURE,tau,post.KINEMATIC,tangent,flow,rho_wall=[.8,1.6])
        np.testing.assert_allclose(result["cf"],[-1.6/6000,3.2/6000])
        with self.assertRaises(ValueError):
            post.wall_coefficients([100000]*2,post.PRESSURE,tau,post.KINEMATIC,tangent,flow,rho_wall=1.2)
        with self.assertRaises(ValueError):
            post.wall_coefficients([100000]*2,post.KINEMATIC,tau,post.PRESSURE,tangent,flow,rho_wall=[1.2,1.2])

    def test_zero_gradient_uses_saved_internal_cell_values_with_native_owner_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            field=root/"p"
            write_field(field,[1,1,1],post.PRESSURE)
            text=field.read_text().replace("internalField uniform 0;", "internalField nonuniform List<scalar> 4 (100000 100100 100200 100300);")
            text=text.replace("type calculated; value nonuniform List<scalar>\n3\n(\n1\n1\n1\n);", "type zeroGradient;")
            field.write_text(text)
            with self.assertRaises(ValueError):
                post.read_boundary_field(field,count=3)
            values=post.read_boundary_field(field,count=3,owner_cells=[3,0,2])
            np.testing.assert_array_equal(values["values"],[100300,100000,100200])
            self.assertEqual(values["value_source"],"zeroGradient_saved_internal_owner_cells")
            poly=root/"constant/polyMesh";poly.mkdir(parents=True)
            (poly/"owner").write_text("FoamFile {format ascii; class labelList; object owner;}\n6\n(0 1 1 3 0 2)\n")
            owner=post.wall_owner_cells(root,{"wall_face_indices":[3,4,5]},3)
            np.testing.assert_array_equal(owner,[3,0,2])

    def test_closed_surface_force_integral_includes_base_with_correct_pressure_sign(self):
        # Clockwise rectangular wall contour: lower TE->LE->upper LE->TE.
        meta={"wall_points_xy":[[1,0],[0,0],[0,.1],[1,.1]],"span":.01}
        flow={"p_inf":100000,"rho_inf":1.2,"U_inf":100,"alpha_deg":0}
        pressure=np.array([106000,100000,100000,100000])
        traction=np.array([[2,0,0],[0,0,0],[2,0,0],[0,0,0]])
        result=post.integrate_wall_forces(meta,pressure,traction,flow)
        self.assertAlmostEqual(result["cl"],1)
        self.assertAlmostEqual(result["cd"],4/6000)
        self.assertTrue(result["finite_te_base_included"])

    def fixture(self, root):
        time=root/"100";time.mkdir()
        x=[1,.8,.4,.01,.01,.4,.8,1,1]
        centres=np.column_stack((x,np.zeros(9),np.zeros(9)))
        tangent=np.array([[-1,0,0]]*4+[[1,0,0]]*4+[[0,-1,0]])
        meta={"wall_face_centers":centres.tolist(),"wall_face_tangents":tangent.tolist(),
              "wall_face_base_mask":[False]*8+[True]}
        (root/"mesh_metadata.json").write_text(json.dumps(meta))
        write_field(time/"p",[100000]*9,post.PRESSURE)
        write_field(time/"wallShearStress",[[-2,0,0]]*9,post.PRESSURE,vector=True)
        write_field(time/"yPlus",[.5]*9,post.DIMENSIONLESS)
        log=[]
        for iteration in range(1,101):
            log.append("Time = "+str(iteration))
            for field in ("p","Ux","Uy","h","k","omega"):
                log.append("smoothSolver: Solving for "+field+", Initial residual = 1e-7, Final residual = 1e-9, No Iterations 1")
        (root/"log.rhoSimpleFoam").write_text("\n".join(log)+"\nEnd\n")
        force=root/"postProcessing/forceCoeffs/0";force.mkdir(parents=True)
        (force/"coefficient.dat").write_text("# Time Cd Cl CmPitch\n"+"\n".join(f"{i} .02 .4 -.1" for i in range(1,101)))
        return {"p_inf":100000,"rho_inf":1.2,"U_inf":100,"alpha_deg":2}

    def test_full_solution_certification_and_unconverged_no_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);flow=self.fixture(root)
            half=np.linspace(0,1,161);target=np.r_[half[::-1],half[1:]]
            result=post.parse_solution(root,flow,target,cl_target=.4)
            self.assertTrue(result["converged"])
            self.assertEqual(result["labels"].shape,(2,321))
            np.testing.assert_allclose(result["labels"][0],0)
            self.assertTrue((result["labels"][1,:161]<0).all())
            self.assertTrue((result["labels"][1,161:]>0).all())
            self.assertEqual(result["quality"]["interpolation"]["base_faces_excluded"],1)
            self.assertAlmostEqual(result["yplus"]["max"],.5)
            result=post.parse_solution(root,flow,target,cl_target=.7,require_converged=False)
            self.assertFalse(result["converged"])
            self.assertIsNone(result["labels"])
            with self.assertRaises(ValueError):
                post.parse_solution(root,flow,target,cl_target=.7)
            write_field(root/"100/yPlus",[2]*9,post.DIMENSIONLESS)
            result=post.parse_solution(root,flow,target,cl_target=.4,require_converged=False)
            self.assertIsNone(result["labels"])
            self.assertIn("wall_yplus_exceeds_limit",result["quality"]["issues"])
            (root/"100/yPlus").unlink()
            result=post.parse_solution(root,flow,target,cl_target=.4,require_converged=False)
            self.assertIsNone(result["labels"])
            self.assertIn("missing_wall_yplus",result["quality"]["issues"])

    def test_residuals_use_nonlinear_initial_values_and_force_named_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            path=root/"log.rhoSimpleFoam"
            path.write_text(path.read_text().replace("Initial residual = 1e-7", "Initial residual = .1"))
            parsed=post.parse_solver_log(path)
            self.assertEqual(parsed["initial_residuals"]["p"],.1)
            self.assertEqual(parsed["final_residuals"]["p"],1e-9)
            history=post.read_force_history(root)
            np.testing.assert_allclose(history["cl"],.4)
            np.testing.assert_allclose(history["cd"],.02)

    def test_nonorthogonal_pressure_correctors_do_not_hide_nonlinear_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);flow=self.fixture(root)
            path=root/"log.rhoSimpleFoam"
            # The final time block's first pressure solve remains unconverged.
            text=path.read_text().replace("\nEnd\n", "\n"+
                "GAMG: Solving for p, Initial residual = .1, Final residual = 1e-5, No Iterations 3\n"+
                "GAMG: Solving for p, Initial residual = .001, Final residual = 1e-8, No Iterations 2\n"+
                "GAMG: Solving for p, Initial residual = 1e-8, Final residual = 1e-10, No Iterations 1\nEnd\n")
            path.write_text(text)
            parsed=post.parse_solver_log(path)
            self.assertEqual(parsed["initial_residuals"]["p"],.1)
            self.assertEqual(parsed["final_residuals"]["p"],1e-10)
            half=np.linspace(0,1,161);target=np.r_[half[::-1],half[1:]]
            result=post.parse_solution(root,flow,target,cl_target=.4,require_converged=False)
            self.assertEqual(result["nonlinear_residuals"]["p"],.1)
            self.assertIn("nonlinear_residual_p",result["quality"]["issues"])
            self.assertIsNone(result["labels"])

    def test_native_exception_trapping_banner_is_not_a_floating_point_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"native.log"
            normal=("trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n"
                    "rhoInf : 1.45000295863\nTime = 1030\n"
                    "smoothSolver: Solving for Ux, Initial residual = 1e-7, Final residual = 1e-9, No Iterations 1\n"
                    "End\n")
            path.write_text(normal)
            parsed=post.parse_solver_log(path)
            self.assertTrue(parsed["ended"])
            self.assertFalse(parsed["fatal_error"])
            for failure in ("FOAM FATAL ERROR: bad field\n","[stack trace]\n", "Foam::sigFpe::sigHandler(int)\n",
                            "Floating point exception (core dumped)\n","Program received signal SIGFPE\n",
                            "Segmentation fault\n", "Initial residual = nan\n", "max: inf\n"):
                with self.subTest(failure=failure):
                    path.write_text(normal+failure)
                    self.assertTrue(post.parse_solver_log(path)["fatal_error"])


if __name__=="__main__":
    unittest.main()
