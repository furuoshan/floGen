"""Generate a compressible OpenCFD OpenFOAM v2412 airfoil RANS case.

This module writes dictionaries/initial fields only. It does not mesh, launch
OpenFOAM, open an SSH connection, or certify convergence. Mesh patches must be
airfoil (wall), farfield (patch), and frontAndBack (empty).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re

import numpy as np


SOURCES = {
    "solver":"https://doc.openfoam.com/2312/tools/processing/solvers/rtm/compressible/rhoSimpleFoam/",
    "sutherland":"https://api.openfoam.com/2412/classFoam_1_1sutherlandTransport.html",
    "farfield_velocity":"https://api.openfoam.com/2412/classFoam_1_1freestreamVelocityFvPatchVectorField.html",
    "farfield_pressure":"https://api.openfoam.com/2412/classFoam_1_1freestreamPressureFvPatchScalarField.html",
    "low_re_wall":"https://api.openfoam.com/2412/classFoam_1_1nutLowReWallFunctionFvPatchScalarField.html",
    "forces":"https://api.openfoam.com/2412/classFoam_1_1functionObjects_1_1forceCoeffs.html",
    "wall_shear_source":"https://api.openfoam.com/2306/wallShearStress_8C_source.html",
    "potential_fields":"https://api.openfoam.com/2412/solvers_2basic_2potentialFoam_2createFields_8H.html",
    "pressure_limits":"https://api.openfoam.com/2412/pressureControl_8C_source.html",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(root, path):
    root = Path(root).resolve(strict=True)
    path = Path(path)
    if not path.is_absolute():
        path = root/path
    resolved = path.resolve()
    require(resolved == root or resolved.is_relative_to(root),f"Path escapes project root: {path}")
    return resolved


@dataclass(frozen=True)
class FlowConditions:
    mach: float
    reynolds: float
    temperature_kelvin: float
    alpha_deg: float
    chord_metres: float
    span_metres: float
    gamma: float
    gas_constant: float
    speed: float
    density: float
    pressure: float
    dynamic_viscosity: float
    kinematic_viscosity: float
    dynamic_pressure: float
    sutherland_As: float
    sutherland_Ts: float
    Cp: float
    mol_weight: float
    turbulence_intensity: float
    turbulence_length_scale: float
    k: float
    omega: float
    nut: float
    Prt: float
    velocity: tuple
    lift_direction: tuple
    drag_direction: tuple


def flow_conditions(mach, reynolds, temp=300.0, alpha_deg=0.0, chord=1.0,
                    span=.01, turbulence_intensity=.01,
                    turbulence_length_scale=.01, gamma=1.4, gas_constant=287.05,
                    mu_300=1.846e-5, sutherland_Ts=110.4, Prt=.85):
    values = (mach,reynolds,temp,alpha_deg,chord,span,turbulence_intensity,
              turbulence_length_scale,gamma,gas_constant,mu_300,sutherland_Ts,Prt)
    require(all(math.isfinite(float(v)) for v in values),"Flow parameters must be finite")
    require(0 < mach < 1 and reynolds > 0 and temp > 0 and chord > 0 and span > 0,
            "Require subsonic positive Mach, Re, T, chord and span")
    require(0 < turbulence_intensity < 1 and turbulence_length_scale > 0 and gamma > 1 and Prt > 0,
            "Invalid turbulence or gas parameters")
    As = mu_300*(1+sutherland_Ts/300.0)/math.sqrt(300.0)
    mu = As*math.sqrt(temp)/(1+sutherland_Ts/temp)
    speed = mach*math.sqrt(gamma*gas_constant*temp)
    rho = reynolds*mu/(speed*chord)
    p = rho*gas_constant*temp
    angle = math.radians(alpha_deg)
    ca,sa = math.cos(angle),math.sin(angle)
    k = 1.5*(turbulence_intensity*speed)**2
    omega = math.sqrt(k)/(.09**.25*turbulence_length_scale)
    return FlowConditions(float(mach),float(reynolds),float(temp),float(alpha_deg),float(chord),float(span),
                          gamma,gas_constant,speed,rho,p,mu,mu/rho,.5*rho*speed*speed,As,sutherland_Ts,
                          gamma*gas_constant/(gamma-1),8314.46261815324/gas_constant,
                          turbulence_intensity,turbulence_length_scale,k,omega,k/omega,Prt,
                          (speed*ca,speed*sa,0.),(-sa,ca,0.),(ca,sa,0.))


def _number(value):
    return format(float(value),".15g")


def _vector(values):
    return "("+" ".join(_number(v) for v in values)+")"


def foam_header(object_name, field_class="dictionary", location=None):
    location_line = f'    location "{location}";\n' if location else ""
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class "+field_class+";\n"+
            location_line+"    object "+object_name+";\n}\n\n")


def _field(name, field_class, dimensions, internal, wall, farfield):
    return (foam_header(name,field_class,"0")+"dimensions "+dimensions+";\n"
            "internalField uniform "+internal+";\n\nboundaryField\n{\n"
            "    airfoil\n    {\n"+wall+"    }\n"
            "    farfield\n    {\n"+farfield+"    }\n"
            "    frontAndBack\n    {\n        type empty;\n    }\n}\n")


def _scalar_inlet(value):
    scalar = _number(value)
    return "        type inletOutlet;\n        inletValue uniform "+scalar+";\n        value uniform "+scalar+";\n"


def case_texts(flow, *, end_time=1000, start_from="startTime", write_interval=100,
               residual_tolerance=1e-6, convection="upwind"):
    require(start_from in ("startTime","latestTime"),"Invalid startFrom")
    require(end_time > 0 and write_interval > 0 and residual_tolerance > 0,"Invalid run controls")
    require(convection in ("upwind","limitedLinear"),"Unsupported convection scheme")
    f = flow
    U = _vector(f.velocity)
    p,T,k,omega,nut = map(_number,(f.pressure,f.temperature_kelvin,f.k,f.omega,f.nut))
    calculated = lambda value:"        type calculated;\n        value uniform "+_number(value)+";\n"
    field = {
        "U":_field("U","volVectorField","[0 1 -1 0 0 0 0]",U,
                   "        type noSlip;\n",
                   "        type freestreamVelocity;\n        freestreamValue uniform "+U+";\n        value uniform "+U+";\n"),
        "p":_field("p","volScalarField","[1 -1 -2 0 0 0 0]",p,
                   "        type zeroGradient;\n",
                   "        type freestreamPressure;\n        freestreamValue uniform "+p+";\n        value uniform "+p+";\n"),
        "T":_field("T","volScalarField","[0 0 0 1 0 0 0]",T,
                   "        type zeroGradient;\n",_scalar_inlet(f.temperature_kelvin)),
        "k":_field("k","volScalarField","[0 2 -2 0 0 0 0]",k,
                   "        type fixedValue;\n        value uniform 0;\n",_scalar_inlet(f.k)),
        "omega":_field("omega","volScalarField","[0 0 -1 0 0 0 0]",omega,
                   "        type omegaWallFunction;\n        value uniform "+omega+";\n",_scalar_inlet(f.omega)),
        "nut":_field("nut","volScalarField","[0 2 -1 0 0 0 0]",nut,
                   "        type nutLowReWallFunction;\n        value uniform 0;\n",calculated(f.nut)),
        "alphat":_field("alphat","volScalarField","[1 -1 -1 0 0 0 0]",_number(f.density*f.nut/f.Prt),
                   "        type fixedValue;\n        value uniform 0;\n",calculated(f.density*f.nut/f.Prt)),
        # Potential-flow-only auxiliary fields have different physical units
        # from compressible p. The explicit pName keeps them separate.
        "pPotential":_field("pPotential","volScalarField","[0 2 -2 0 0 0 0]","0",
                   "        type zeroGradient;\n","        type fixedValue;\n        value uniform 0;\n"),
        "Phi":_field("Phi","volScalarField","[0 2 -1 0 0 0 0]","0",
                   "        type zeroGradient;\n","        type zeroGradient;\n"),
    }
    files = {"0/"+name:text for name,text in field.items()}
    files["constant/thermophysicalProperties"] = foam_header("thermophysicalProperties")+f"""thermoType
{{
    type hePsiThermo;
    mixture pureMixture;
    transport sutherland;
    thermo hConst;
    equationOfState perfectGas;
    specie specie;
    energy sensibleEnthalpy;
}}
mixture
{{
    specie {{ molWeight {_number(f.mol_weight)}; }}
    thermodynamics {{ Cp {_number(f.Cp)}; Hf 0; }}
    transport {{ As {_number(f.sutherland_As)}; Ts {_number(f.sutherland_Ts)}; }}
}}
"""
    files["constant/turbulenceProperties"] = foam_header("turbulenceProperties")+"""simulationType RAS;
RAS
{
    RASModel kOmegaSST;
    turbulence on;
    printCoeffs on;
}
"""
    scheme = "upwind" if convection == "upwind" else "limitedLinear 1"
    files["system/fvSchemes"] = foam_header("fvSchemes")+f"""ddtSchemes {{ default steadyState; }}
gradSchemes {{ default cellLimited Gauss linear 1; }}
divSchemes
{{
    default Gauss upwind;
    div(phi,U) bounded Gauss {scheme};
    div(phi,h) bounded Gauss {scheme};
    div(phi,e) bounded Gauss {scheme};
    div(phi,K) bounded Gauss {scheme};
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div(phid,p) Gauss upwind;
    div(phiv,p) Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear limited 0.5; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default limited 0.5; }}
wallDist {{ method meshWave; }}
fluxRequired {{ default no; p; Phi; }}
"""
    files["system/fvSolution"] = foam_header("fvSolution")+f"""solvers
{{
    p
    {{
        solver PBiCGStab;
        preconditioner DILU;
        tolerance 1e-8;
        relTol 0.01;
    }}
    pFinal {{ $p; relTol 0; }}
    "(U|h|e|k|omega)"
    {{
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-8;
        relTol 0.01;
    }}
    rho {{ solver diagonal; }}
    Phi
    {{
        solver PCG;
        preconditioner DIC;
        tolerance 1e-10;
        relTol 0;
    }}
}}
potentialFlow
{{
    nNonOrthogonalCorrectors 10;
    PhiRefCell 0;
    PhiRefValue 0;
}}
SIMPLE
{{
    transonic yes;
    consistent no;
    nNonOrthogonalCorrectors 2;
    pMin {_number(.05*f.pressure)};
    pMax {_number(2*f.pressure)};
    residualControl
    {{
        p {_number(residual_tolerance)};
        U {_number(residual_tolerance)};
        h {_number(residual_tolerance)};
        k {_number(residual_tolerance)};
        omega {_number(residual_tolerance)};
    }}
}}
relaxationFactors
{{
    fields {{ p 0.1; rho 0.3; }}
    equations {{ U 0.3; h 0.3; e 0.3; k 0.3; omega 0.3; }}
}}
"""
    files["system/controlDict"] = foam_header("controlDict")+f"""application rhoSimpleFoam;
startFrom {start_from};
startTime 0;
stopAt endTime;
endTime {int(end_time)};
deltaT 1;
writeControl timeStep;
writeInterval {int(write_interval)};
purgeWrite 2;
writeFormat ascii;
writePrecision 12;
writeCompression off;
timeFormat general;
timePrecision 10;
runTimeModifiable true;
functions
{{
    forceCoeffs
    {{
        type forceCoeffs;
        libs (forces);
        patches (airfoil);
        p p;
        U U;
        rho rho;
        rhoInf {_number(f.density)};
        pRef {_number(f.pressure)};
        CofR ({_number(.25*f.chord_metres)} 0 0);
        liftDir {_vector(f.lift_direction)};
        dragDir {_vector(f.drag_direction)};
        pitchAxis (0 0 1);
        magUInf {_number(f.speed)};
        lRef {_number(f.chord_metres)};
        Aref {_number(f.span_metres*f.chord_metres)};
        coefficients (Cd Cl CmPitch);
        writeControl timeStep;
        writeInterval 1;
        log true;
    }}
    wallShearStress
    {{
        type wallShearStress;
        libs (fieldFunctionObjects);
        patches (airfoil);
        writeFields true;
        executeControl timeStep;
        executeInterval 1;
        writeControl writeTime;
    }}
    yPlus
    {{
        type yPlus;
        libs (fieldFunctionObjects);
        patches (airfoil);
        executeControl writeTime;
        writeControl writeTime;
        log true;
    }}
    residuals
    {{
        type solverInfo;
        libs (utilityFunctionObjects);
        fields (p U h k omega);
        writeResidualFields false;
        writeControl timeStep;
        writeInterval 1;
    }}
}}
"""
    return files


def write_case(case_dir, mach, reynolds, temp=300.0, alpha_deg=0.0, span=.01,
               end_time=1000, project_root=None, start_from="startTime",
               turbulence_intensity=.01, turbulence_length_scale=.01,
               write_interval=100, residual_tolerance=1e-6, convection="upwind",
               overwrite_initial=False):
    """Write dictionaries without modifying mesh files or an existing solution.

    For restart at a new angle, update latest/U's farfield freestreamValue via
    the returned restart_boundary_updates using foamDictionary. Internal fields
    are never overwritten by this function. Existing 0/ fields are retained
    unless overwrite_initial=True.
    """
    case = Path(case_dir)
    root = Path(project_root).resolve(strict=True) if project_root else case.parent.resolve(strict=True)
    case = _inside(root,case)
    flow = flow_conditions(mach,reynolds,temp,alpha_deg,1.,span,
                           turbulence_intensity,turbulence_length_scale)
    files = case_texts(flow,end_time=end_time,start_from=start_from,write_interval=write_interval,
                       residual_tolerance=residual_tolerance,convection=convection)
    # Validate all paths before the first write, including symlinks in an
    # already-prepared mesh/case directory.
    for relative in (*files,"case_conditions.json"):
        _inside(root,case/relative)
    for relative,text in files.items():
        path = case/relative
        if relative.startswith("0/") and path.exists() and not overwrite_initial:
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding="ascii")
    metadata = asdict(flow)
    metadata.update({"U_inf":flow.speed,"velocity_inf":list(flow.velocity),"rho_inf":flow.density,
                     "p_inf":flow.pressure,"q_inf":flow.dynamic_pressure,
                     "T_inf":flow.temperature_kelvin,"mu_inf":flow.dynamic_viscosity,
                     "R":flow.gas_constant})
    potential_velocity = _vector(flow.velocity)
    potential_set_fixed = "{ type fixedValue; value uniform "+potential_velocity+"; }"
    potential_restore_freestream = ("{ type freestreamVelocity; freestreamValue uniform "+potential_velocity+
                                   "; value uniform "+potential_velocity+"; }")
    metadata.update({"solver":"rhoSimpleFoam","openfoam_family":"OpenCFD","target_version":"v2412",
                     "thermo":"hePsiThermo/sutherland/hConst/perfectGas/sensibleEnthalpy",
                     "turbulence_model":"RAS/kOmegaSST","wall_treatment":"wall_resolved_low_yplus",
                     "patches":{"wall":"airfoil","farfield":"farfield","empty":"frontAndBack"},
                     "mesh_required":True,"mesh_generated_here":False,"native_solver_verified":False,
                     "convection_scheme":convection,"first_order_initialization":convection=="upwind",
                     "turbulence_convection_scheme":"bounded Gauss upwind for both k and omega in all stages",
                     "end_time":int(end_time),"start_from":start_from,
                     "residual_tolerance":residual_tolerance,
                     "convergence_requires":"residuals AND stable forces AND target_CL AND mesh/yPlus acceptance",
                     "restart_boundary_updates":[{"field":"U","entry":"boundaryField.farfield.freestreamValue",
                                                  "value":"uniform "+_vector(flow.velocity)}],
                     "potential_initialization":{
                         "purpose":"Nonpenetrating velocity initial guess only; RANS still uses prescribed Mach/Re/T",
                         "temporarily_set_U_farfield":potential_set_fixed,
                         "restore_U_farfield":potential_restore_freestream,
                         "command":"potentialFoam -pName pPotential -initialiseUBCs -writePhi",
                         "must_not_use_options":["-writep","-writephi","-withFunctionObjects"],
                         "reason_for_temporary_fixed_U":"potentialFoam zeroes U before applying BCs; angle-based freestreamVelocity can divide by |U|=0",
                         "pressure_field_unchanged":"p [Pa] is never passed to the potential pressure solve",
                         "mass_flux_warning":"Do not write volumetric phi; rhoSimpleFoam constructs mass phi from rho and initialized U",
                         "auxiliary_reference":"pPotential farfield fixedValue zero bypasses adjustPhi's roundoff-sensitive all-fixed-U outflow balance; Phi remains zeroGradient with its own reference cell",
                         "native_initialization_probe":"v2412 fixed-pressure-only auxiliary BC: continuity 2.62e-8, mean speed 1.036 Uinf; true Pa pressure unchanged",
                         "restore_preserves_internal_U":True},
                     "pressure_reference":"freestreamPressure fixes pressure on normal outflow; no all-Neumann physical-pressure reference is needed",
                     "wall_shear_conversion":"Read dimensions. Dynamic Pa direct; kinematic requires local rho. OpenFOAM wall-on-fluid traction is negated for fluid-on-body signed Cf.",
                     "official_sources":SOURCES})
    (case/"case_conditions.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return metadata


def parse_dimensions(text):
    match = re.search(r"\bdimensions\s*\[([^\]]+)\]\s*;",text)
    require(match is not None,"OpenFOAM field dimensions are missing")
    values = tuple(float(x) for x in match.group(1).split())
    require(len(values)==7,"OpenFOAM dimensions must have 7 exponents")
    return values


def _replace_block(text, name, replacement):
    match = re.search(r"(?m)^"+re.escape(name)+r"\s*\{",text)
    require(match is not None,f"Missing dictionary block {name}")
    depth = 1
    for end in range(match.end(),len(text)):
        depth += (text[end]=="{")-(text[end]=="}")
        if depth == 0:
            return text[:match.start()]+replacement+text[end+1:]
    raise ValueError("Unclosed dictionary block "+name)


def write_lts_case(case_dir, mach, reynolds, temp=300.0, alpha_deg=0.0, span=.01,
                   end_time=1000, project_root=None, start_from="startTime",
                   turbulence_intensity=.01, turbulence_length_scale=.01,
                   write_interval=100, convection="upwind", max_co=.2,
                   max_local_dt=1e-3, smoothing=.1, damping=.2,
                   n_outer_correctors=2, n_correctors=2, overwrite_initial=False):
    """Generate rhoPimpleFoam localEuler pseudo-transient compressible SST.

    Local time stepping is a steady-state acceleration, not a physical unsteady
    time series. This separate API leaves write_case's rhoSimpleFoam identity
    unchanged. No alphaTemp knob is advertised: that is not read by the native
    non-reacting rhoPimpleFoam/setRDeltaT.H implementation.
    """
    require(max_co>0 and max_local_dt>0 and 0<smoothing<=1 and 0<damping<=1,
            "Invalid local time-step controls")
    require(n_outer_correctors>=1 and n_correctors>=1,"Invalid PIMPLE corrector counts")
    metadata = write_case(case_dir,mach,reynolds,temp,alpha_deg,span,end_time,
                           project_root,start_from,turbulence_intensity,turbulence_length_scale,
                           write_interval,convection=convection,overwrite_initial=overwrite_initial)
    case = Path(case_dir)
    root = Path(project_root).resolve(strict=True) if project_root else case.parent.resolve(strict=True)
    case = _inside(root,case)
    schemes = case/"system/fvSchemes"
    schemes.write_text(schemes.read_text().replace("default steadyState;","default localEuler;"),encoding="ascii")
    solution = case/"system/fvSolution"
    solution_text = solution.read_text()
    pimple = f"""PIMPLE
{{
    momentumPredictor yes;
    transonic yes;
    consistent no;
    nOuterCorrectors {int(n_outer_correctors)};
    nCorrectors {int(n_correctors)};
    nNonOrthogonalCorrectors 1;
    turbOnFinalIterOnly true;
    maxCo {_number(max_co)};
    maxDeltaT {_number(max_local_dt)};
    rDeltaTSmoothingCoeff {_number(smoothing)};
    rDeltaTDampingCoeff {_number(damping)};
    pMin {_number(.05*metadata['p_inf'])};
    pMax {_number(2*metadata['p_inf'])};
}}
"""
    solution_text = _replace_block(solution_text,"SIMPLE",pimple)
    # Final PIMPLE correctors may select '*Final' solver entries.
    at = solution_text.index("    rho { solver diagonal; }")
    final_solvers = """    "(U|h|e|k|omega)Final"
    {
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-8;
        relTol 0;
    }
    rhoFinal { solver diagonal; }
"""
    solution_text = solution_text[:at]+final_solvers+solution_text[at:]
    solution.write_text(solution_text,encoding="ascii")
    control = case/"system/controlDict"
    control_text = control.read_text().replace("application rhoSimpleFoam;","application rhoPimpleFoam;")
    control_text = control_text.replace("runTimeModifiable true;","runTimeModifiable true;\nadjustTimeStep no;")
    control.write_text(control_text,encoding="ascii")
    metadata.update({"solver":"rhoPimpleFoam","time_scheme":"localEuler",
                     "temporal_interpretation":"local pseudo time for steady RANS; not a physical unsteady solution",
                     "native_solver_verified":False,"lts_controls":{"maxCo":max_co,"maxDeltaT":max_local_dt,
                     "rDeltaTSmoothingCoeff":smoothing,"rDeltaTDampingCoeff":damping,
                     "nOuterCorrectors":n_outer_correctors,"nCorrectors":n_correctors},
                     "lts_temperature_control":"alphaTemp is NOT a rhoPimpleFoam control; omit rather than borrow reactingFoam settings",
                     "launch_command":"rhoPimpleFoam",
                     "steady_label_acceptance":"Require stationary converged forces/residuals, target CL and mesh/yPlus acceptance after pseudo-time transients",
                     "lts_source":"https://api.openfoam.com/2412/compressible_2rhoPimpleFoam_2setRDeltaT_8H_source.html"})
    (case/"case_conditions.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return metadata


def surface_coefficients(pressure, wall_shear, contour_tangent, flow,
                         pressure_dimensions, shear_dimensions,
                         local_density=None, wss_traction="wall_on_fluid"):
    """Convert dimensional surface fields to Cp and signed fluid-on-body Cf.

    contour_tangent must follow lower TE->LE->upper TE, i.e. the target dataset
    point order. The OpenFOAM field's units and traction convention are explicit
    arguments; unsupported/missing dimensions are rejected rather than guessed.
    """
    get = lambda name:getattr(flow,name) if isinstance(flow,FlowConditions) else flow[name]
    p = np.asarray(pressure,dtype=float)
    tau = np.asarray(wall_shear,dtype=float)
    tangent = np.asarray(contour_tangent,dtype=float)
    require(tuple(pressure_dimensions)==(1,-1,-2,0,0,0,0),"Pressure must be absolute dynamic pressure units [Pa]")
    require(tau.shape==tangent.shape and tau.ndim==2 and tau.shape[1] in (2,3) and len(p)==len(tau),
            "Surface pressure, shear and tangent shapes disagree")
    require(np.isfinite(p).all() and np.isfinite(tau).all() and np.isfinite(tangent).all(),"Nonfinite surface values")
    norm = np.linalg.norm(tangent,axis=1)
    require((norm>0).all(),"Zero contour tangent")
    tangent = tangent/norm[:,None]
    dimensions = tuple(shear_dimensions)
    if dimensions == (0,2,-2,0,0,0,0):
        require(local_density is not None,"Kinematic wall shear requires local wall density")
        rho = np.asarray(local_density,dtype=float)
        require(rho.shape==p.shape and np.isfinite(rho).all() and (rho>0).all(),"Invalid local wall density")
        tau = tau*rho[:,None]
    else:
        require(dimensions==(1,-1,-2,0,0,0,0),"Unsupported wallShearStress dimensions")
    require(wss_traction in ("wall_on_fluid","fluid_on_body"),"Unknown shear traction convention")
    sign = -1 if wss_traction=="wall_on_fluid" else 1
    cp = (p-get("pressure"))/get("dynamic_pressure")
    cf = sign*np.sum(tau*tangent,axis=1)/get("dynamic_pressure")
    return cp,cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--case",type=Path,required=True)
    parser.add_argument("--mach",type=float,required=True)
    parser.add_argument("--reynolds",type=float,required=True)
    parser.add_argument("--temperature",type=float,default=300.)
    parser.add_argument("--alpha",type=float,default=0.)
    parser.add_argument("--span",type=float,default=.01)
    parser.add_argument("--end-time",type=int,default=1000)
    parser.add_argument("--start-from",choices=("startTime","latestTime"),default="startTime")
    parser.add_argument("--convection",choices=("upwind","limitedLinear"),default="upwind")
    args = parser.parse_args()
    result = write_case(args.case,args.mach,args.reynolds,args.temperature,args.alpha,args.span,args.end_time,
                        args.project_root,args.start_from,convection=args.convection)
    print(json.dumps(result,indent=2,allow_nan=False))
