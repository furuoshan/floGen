"""Read genuine OpenFOAM ASCII wall fields and certify 2-D airfoil labels.

OpenCFD v2412 wallShearStress uses (-Sf/|Sf|)&devRhoReff. Since forces uses
Sf&devRhoReff, force on the body per unit area is -wallShearStress. The native
compressible output inherits dynamic-pressure dimensions from devRhoReff.
No unsuccessful solution is converted into training labels.
"""
from __future__ import annotations

from pathlib import Path
import gzip
import json
import math
import re
import numpy as np

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PRESSURE = (1, -1, -2, 0, 0, 0, 0)
KINEMATIC = (0, 2, -2, 0, 0, 0, 0)
DENSITY = (1, -3, 0, 0, 0, 0, 0)
TEMPERATURE = (0, 0, 0, 1, 0, 0, 0)
DIMENSIONLESS = (0, 0, 0, 0, 0, 0, 0)


def require(value, message):
    if not value:
        raise ValueError(message)


def _inside(root, path):
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    require(path == root or path.is_relative_to(root), "Case file escapes case directory")
    return path


def _read(root, path):
    path = _inside(root, path)
    if not path.is_file() and path.with_suffix(path.suffix + ".gz").is_file():
        path = _inside(root, path.with_suffix(path.suffix + ".gz"))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii", errors="strict") as stream:
        return stream.read()


def _strip_comments(text):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)


def _block(text, name):
    found = re.search(r'(?<![\w.])(?:"' + re.escape(name) + r'"|' + re.escape(name) + r')\s*\{', text)
    require(found is not None, f"Missing dictionary block {name}")
    start, depth = found.end(), 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i]
    raise ValueError(f"Unclosed dictionary block {name}")


def _internal_values(text, owner_cells):
    """Evaluate a zeroGradient patch from the saved adjacent-cell solution."""
    owner_cells = np.asarray(owner_cells, dtype=np.int64)
    require(owner_cells.ndim == 1 and (owner_cells >= 0).all(), "Invalid wall owner-cell indices")
    uniform = re.search(r"\binternalField\s+uniform\s+([^;]+);", text)
    if uniform:
        values = np.asarray([float(s) for s in re.findall(NUMBER, uniform.group(1))])
        require(values.size in (1, 3), "Unsupported internal field")
        return np.full(len(owner_cells), values[0]) if values.size == 1 else np.tile(values, (len(owner_cells), 1))
    found = re.search(r"\binternalField\s+nonuniform\s+List<(scalar|vector)>\s+(\d+)\s*\(", text)
    require(found is not None, "No explicit saved internal field for zeroGradient patch")
    n, width = int(found.group(2)), 3 if found.group(1) == "vector" else 1
    start, depth, end = found.end(), 1, None
    for i in range(start, len(text)):
        depth += (text[i] == "(") - (text[i] == ")")
        if depth == 0:
            end = i
            break
    require(end is not None, "Unclosed internal field list")
    flat = np.asarray([float(s) for s in re.findall(NUMBER, text[start:end])])
    require(flat.size == n*width and (owner_cells < n).all(), "Internal field/owner indices mismatch")
    values = flat.reshape(n, width) if width > 1 else flat
    return values[owner_cells]


def read_boundary_field(path, patch="airfoil", count=None, case_root=None, owner_cells=None):
    """Return dimensions and an explicit uniform/nonuniform ASCII patch value.

    count is mandatory for a uniform field. A zeroGradient patch without a
    stored value requires explicit owner_cells and is evaluated from the saved
    internal solution, exactly as OpenFOAM's extrapolateInternal implementation.
    """
    path = Path(path)
    root = Path(case_root).resolve() if case_root else path.resolve().parent
    text = _strip_comments(_read(root, path))
    header = _block(text, "FoamFile")
    require(re.search(r"\bformat\s+ascii\s*;", header) is not None,
            "Only native ASCII OpenFOAM fields are supported")
    dims = re.search(r"\bdimensions\s*\[([^]]+)\]\s*;", text)
    require(dims is not None, "Field has no dimensions")
    dimension_values = [float(s) for s in dims.group(1).split()]
    require(len(dimension_values) == 7 and all(x == int(x) for x in dimension_values),
            "Unsupported dimension set")
    dimensions = tuple(int(x) for x in dimension_values)
    body = _block(_block(text, "boundaryField"), patch)
    if not re.search(r"\bvalue\s+(?:uniform|nonuniform)\b", body):
        boundary_type = re.search(r"\btype\s+(\w+)\s*;", body)
        require(boundary_type is not None and boundary_type.group(1) == "zeroGradient" and owner_cells is not None,
                "Patch has no explicit value; only zeroGradient with saved owner cells can be evaluated")
        result = _internal_values(text, owner_cells)
        require((count is None or len(result) == count) and np.isfinite(result).all(), "Invalid evaluated zeroGradient patch")
        return {"values":result,"dimensions":dimensions,"patch":patch,"path":str(path),
                "value_source":"zeroGradient_saved_internal_owner_cells"}
    uniform = re.search(r"\bvalue\s+uniform\s+([^;]+);", body)
    if uniform:
        require(count is not None and count >= 1, "Uniform field requires the wall face count")
        values = np.asarray([float(x) for x in re.findall(NUMBER, uniform.group(1))])
        require(values.size in (1, 3), "Only scalar/vector patch fields are supported")
        result = np.full(count, values[0]) if len(values) == 1 else np.tile(values, (count, 1))
    else:
        nonuniform = re.search(r"\bvalue\s+nonuniform\s+List<(scalar|vector)>\s+(\d+)\s*\(", body)
        require(nonuniform is not None, "Patch has no explicit computed value")
        n = int(nonuniform.group(2))
        start, depth, end = nonuniform.end(), 1, None
        for i in range(start, len(body)):
            depth += (body[i] == "(") - (body[i] == ")")
            if depth == 0:
                end = i
                break
        require(end is not None, "Unclosed nonuniform value list")
        flat = np.asarray([float(x) for x in re.findall(NUMBER, body[start:end])])
        width = 3 if nonuniform.group(1) == "vector" else 1
        require(flat.size == n * width, "Nonuniform list length differs from its declaration")
        result = flat.reshape(n, width) if width > 1 else flat
        require(count is None or n == count, "Field/mesh wall-face counts differ")
    require(np.isfinite(result).all(), "Nonfinite OpenFOAM patch field")
    return {"values": result, "dimensions": dimensions, "patch": patch, "path": str(path),
            "value_source":"explicit_boundary_value"}


def wall_owner_cells(case_dir, mesh_metadata, count):
    case = Path(case_dir).resolve()
    if "wall_face_owner_cells" in mesh_metadata:
        owners = np.asarray(mesh_metadata["wall_face_owner_cells"], dtype=np.int64)
        require(owners.shape == (count,) and (owners >= 0).all(), "Invalid metadata owner-cell indices")
        return owners
    owner_path = case/"constant/polyMesh/owner"
    if not owner_path.is_file() and not owner_path.with_suffix(".gz").is_file():
        return None
    text = _strip_comments(_read(case, owner_path))
    start = re.search(r"\bFoamFile\s*\{", text)
    require(start is not None, "Owner file has no native header")
    header = _block(text, "FoamFile")
    require(re.search(r"\bformat\s+ascii\s*;", header), "Owner labels require ASCII format")
    depth, end = 1, None
    for i in range(start.end(), len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            end = i+1
            break
    require(end is not None, "Unclosed owner header")
    found = re.fullmatch(r"\s*(\d+)\s*\((.*?)\)\s*;?\s*", text[end:], flags=re.S)
    require(found is not None, "Invalid native owner label list")
    owners = np.fromstring(found.group(2), sep=" ", dtype=np.int64)
    require(len(owners) == int(found.group(1)) and (owners >= 0).all(), "Invalid owner labels")
    indices = mesh_metadata.get("wall_face_indices")
    if indices is None:
        patch = mesh_metadata["patches"]["airfoil"]
        require(int(patch["nFaces"]) == count, "Wall patch count mismatch")
        indices = np.arange(int(patch["startFace"]), int(patch["startFace"])+count)
    indices = np.asarray(indices, dtype=np.int64)
    require(indices.shape == (count,) and (indices >= 0).all() and (indices < len(owners)).all(), "Wall face IDs exceed owner list")
    return owners[indices]


def latest_time(case_dir):
    root = Path(case_dir).resolve()
    candidates = []
    for path in root.iterdir():
        if path.is_dir() and re.fullmatch(NUMBER, path.name):
            _inside(root, path)
            value = float(path.name)
            if math.isfinite(value) and value > 0:
                candidates.append((value, path))
    require(candidates, "No computed numerical time directory")
    return max(candidates, key=lambda v: v[0])[1]


def _flow(conditions):
    p = float(conditions["p_inf"])
    rho = float(conditions["rho_inf"])
    velocity = conditions.get("U_inf", conditions.get("velocity_inf", conditions.get("speed_inf")))
    require(velocity is not None, "U_inf/velocity_inf/speed_inf is required")
    velocity = np.asarray(velocity, dtype=float)
    speed = float(np.linalg.norm(velocity)) if velocity.ndim else abs(float(velocity))
    require(np.isfinite([p, rho, speed]).all() and p > 0 and rho > 0 and speed > 0,
            "Invalid physical freestream conditions")
    return p, rho, speed, 0.5 * rho * speed**2


def wall_coefficients(pressure, pressure_dimensions, wall_shear, shear_dimensions,
                      tangent, flow_conditions, rho_wall=None,
                      allow_kinematic_pressure=False, pressure_offset_pa=None):
    """Convert native wall fields to Cp and signed Cf with explicit dimensions.

    Kinematic shear requires actual per-face wall density. A scalar freestream
    density is deliberately rejected for that conversion. Kinematic pressure
    additionally requires an explicit pressure offset, since its datum can be
    arbitrary. Dynamic p/tau from rhoSimpleFoam need no density conversion.
    """
    p_inf, _, _, q = _flow(flow_conditions)
    p, tau, tangent = map(lambda a: np.asarray(a, dtype=float), (pressure, wall_shear, tangent))
    require(p.ndim == 1 and tau.shape == tangent.shape == (len(p), 3), "Invalid wall array shapes")
    require(np.isfinite(p).all() and np.isfinite(tau).all() and np.isfinite(tangent).all(), "Nonfinite wall data")
    norm = np.linalg.norm(tangent, axis=1)
    require((norm > 0).all(), "Zero contour tangent")
    tangent = tangent / norm[:, None]
    need_rho = tuple(shear_dimensions) == KINEMATIC or tuple(pressure_dimensions) == KINEMATIC
    if need_rho:
        density = np.asarray(rho_wall, dtype=float)
        require(density.shape == p.shape and np.isfinite(density).all() and (density > 0).all(),
                "Kinematic fields require an actual per-face wall-density array")
    if tuple(pressure_dimensions) == PRESSURE:
        p_pa = p
    elif tuple(pressure_dimensions) == KINEMATIC:
        require(allow_kinematic_pressure and pressure_offset_pa is not None,
                "Kinematic pressure is rejected without explicit density and datum conversion")
        p_pa = p * density + float(pressure_offset_pa)
    else:
        raise ValueError("Unsupported pressure dimensions")
    require(np.isfinite(p_pa).all() and (p_pa > 0).all(), "Nonpositive absolute wall pressure")
    if tuple(shear_dimensions) == PRESSURE:
        tau_pa = tau
    elif tuple(shear_dimensions) == KINEMATIC:
        tau_pa = tau * density[:, None]
    else:
        raise ValueError("Unsupported wallShearStress dimensions")
    tau_body = -tau_pa
    return {"cp": (p_pa-p_inf)/q, "cf": np.sum(tau_body*tangent, axis=1)/q,
            "tau_body_pa": tau_body, "p_pa": p_pa, "dynamic_pressure_pa": q,
            "shear_sign": "body_traction=-native_wallShearStress; dot_with_dataset_contour_tangent"}


def parse_solver_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    current, time_block = None, 0
    records = []
    for line in text.splitlines():
        time_match = re.match(r"\s*Time\s*=\s*("+NUMBER+r")", line)
        if time_match:
            current = float(time_match.group(1))
            time_block += 1
        solve = re.search(r"Solving for\s+([A-Za-z][\w.]*)\s*,\s*Initial residual\s*=\s*("+NUMBER+r"),\s*Final residual\s*=\s*("+NUMBER+r"),\s*No Iterations\s+(\d+)", line)
        if solve:
            records.append({"time": current, "time_block":time_block, "field": solve.group(1),
                            "initial": float(solve.group(2)), "final": float(solve.group(3)),
                            "linear_iterations": int(solve.group(4))})
    initial, final = {}, {}
    # Pressure may be solved several times within one SIMPLE iteration because
    # of nonorthogonal correction. Its last linear solve's Initial residual can
    # be tiny while the first nonlinear pressure residual is still large. Keep
    # the maximum per field within each Time block, including the FIRST solve.
    blocks = {}
    for record in records:
        block = blocks.setdefault(record["time_block"], {"time":record["time"],"initial_max":{},"final_last":{}})
        field = record["field"]
        block["initial_max"][field] = max(block["initial_max"].get(field,0.0),record["initial"])
        block["final_last"][field] = record["final"]
    if blocks:
        last = blocks[max(blocks)]
        initial, final = last["initial_max"], last["final_last"]
    # Every normal native OpenFOAM log announces that exception trapping is
    # enabled. That startup capability message is not a numerical exception.
    fatal = bool(re.search(
        r"FOAM FATAL|\[stack trace\]|Foam::sigFpe|Program received signal SIGFPE|"
        r"Floating point exception(?!\s+trapping\s+enabled)|Segmentation fault|\bnan\b|\binf\b",
        text,re.I))
    return {"history": records, "initial_residuals": initial, "final_residuals": final,
            "time_block_residuals":[blocks[i] for i in sorted(blocks)],
            "initial_residual_policy":"maximum_per_field_within_final_Time_block_including_first_pressure_solve",
            "fatal_error": fatal, "solver_reported_converged": "SIMPLE solution converged" in text,
            "ended": bool(re.search(r"^End\s*$", text, re.M)), "path": str(path)}


def _force_table(path):
    header, rows = None, []
    for line in path.read_text(encoding="ascii", errors="strict").splitlines():
        text = line.strip()
        if text.startswith("#"):
            names = text.lstrip("# ").split()
            lower = [s.lower() for s in names]
            if "time" in lower and "cl" in lower and "cd" in lower:
                header = lower
            continue
        if not text:
            continue
        try:
            numbers = [float(x) for x in text.split()]
        except ValueError:
            continue
        if header and len(numbers) >= len(header):
            rows.append([numbers[header.index(s)] for s in ("time", "cl", "cd")])
    return rows


def read_force_history(case_dir):
    root = Path(case_dir).resolve()
    post = _inside(root, root/"postProcessing")
    candidates = []
    if post.is_dir():
        for filename in ("coefficient.dat", "forceCoeffs.dat"):
            candidates.extend(post.rglob(filename))
    rows, used = [], []
    for path in sorted(set(candidates)):
        _inside(root, path)
        values = _force_table(path)
        if values:
            rows.extend(values)
            used.append(str(path))
    require(rows, "No forceCoeffs history with named Time/Cl/Cd columns")
    # Retain the latest occurrence after a restart; retain numerical time order.
    by_time = {float(row[0]): row for row in rows}
    values = np.asarray([by_time[t] for t in sorted(by_time)])
    require(np.isfinite(values).all(), "Nonfinite force coefficient history")
    return {"time": values[:, 0], "times": values[:, 0], "cl": values[:, 1], "cd": values[:, 2], "paths": used}


def _interpolate(x, values, query):
    order = np.argsort(x, kind="stable")
    x, values = np.asarray(x)[order], np.asarray(values)[order]
    x, unique = np.unique(x, return_index=True)
    require(len(x) >= 3 and x.min() <= 0.02 and x.max() >= 0.98,
            "Wall sampling does not cover both leading/trailing edges")
    return np.interp(query, x, values[unique])


def interpolate_wall_labels(centres, cp, cf, target_x, surface_id=None, is_base=None):
    centres, target_x = np.asarray(centres), np.asarray(target_x)
    n, p = len(centres), len(target_x)
    require(centres.shape == (n, 3) and p % 2 == 1 and np.isfinite(target_x).all(), "Invalid wall/target coordinates")
    keep = np.ones(n, dtype=bool) if is_base is None else ~np.asarray(is_base, dtype=bool)
    require(keep.shape == (n,), "Invalid base-face mask")
    if surface_id is None:
        active = np.flatnonzero(keep)
        le = int(np.argmin(centres[active, 0]))
        lower, upper = active[:le+1], active[le+1:]
    else:
        side = np.asarray(surface_id)
        require(side.shape == (n,), "Invalid surface IDs")
        lower, upper = np.flatnonzero(keep & (side == "lower")), np.flatnonzero(keep & (side == "upper"))
    require(len(lower) >= 3 and len(upper) >= 3, "Cannot identify both airfoil surfaces")
    mid = (p-1)//2
    values = []
    for field in (cp, cf):
        values.append(np.r_[_interpolate(centres[lower,0], np.asarray(field)[lower], target_x[:mid+1]),
                            _interpolate(centres[upper,0], np.asarray(field)[upper], target_x[mid+1:])])
    result = np.asarray(values, dtype=np.float32)
    require(np.isfinite(result).all(), "Nonfinite interpolated wall labels")
    return result, {"method": "linear_separate_surfaces_constant_endpoint_extension",
                    "lower_x_range": [float(centres[lower,0].min()), float(centres[lower,0].max())],
                    "upper_x_range": [float(centres[upper,0].min()), float(centres[upper,0].max())],
                    "base_faces_excluded": int(np.count_nonzero(~keep))}


def _metadata(case, metadata):
    if metadata is None:
        metadata = case/"mesh_metadata.json"
    if not isinstance(metadata, dict):
        metadata = json.loads(_inside(case, metadata).read_text())
    return metadata


def integrate_wall_forces(mesh_metadata, pressure_pa, tau_body_pa, flow_conditions):
    """Independently reproduce forces' pressure and viscous wall integrals.

    The finite trailing-edge base participates in total force integration, even
    though it is excluded from the two airfoil-profile label distributions.
    """
    if "wall_points_xy" not in mesh_metadata:
        return None
    points = np.asarray(mesh_metadata["wall_points_xy"], dtype=float)
    pressure, tau = np.asarray(pressure_pa), np.asarray(tau_body_pa)
    require(points.shape == (len(pressure), 2) and tau.shape == (len(pressure), 3),
            "Wall polygon does not match wall fields")
    span = float(mesh_metadata.get("span", mesh_metadata.get("extrusion_height", 0)))
    require(span > 0, "Extruded wall integration requires positive span")
    edge = np.roll(points, -1, axis=0)-points
    signed_area = .5*float(np.sum(points[:,0]*np.roll(points[:,1],-1)-points[:,1]*np.roll(points[:,0],-1)))
    require(signed_area != 0, "Degenerate closed wall polygon")
    # Sf is outward from the FLUID domain, pointing into the solid body.
    sf_xy = -np.sign(signed_area)*np.column_stack((edge[:,1],-edge[:,0]))*span
    area = np.linalg.norm(edge,axis=1)*span
    p_inf,_,_,q = _flow(flow_conditions)
    body_force = np.column_stack(((pressure-p_inf)[:,None]*sf_xy,np.zeros(len(pressure))))+tau*area[:,None]
    chord = float(flow_conditions.get("chord_metres",1.0))
    angle = math.radians(float(flow_conditions.get("alpha_deg",0.0)))
    lift = np.array([-math.sin(angle),math.cos(angle),0.0])
    drag = np.array([math.cos(angle),math.sin(angle),0.0])
    coefficient = body_force.sum(axis=0)/(q*span*chord)
    return {"cl":float(coefficient@lift),"cd":float(coefficient@drag),
            "force_coefficient_xyz":coefficient.tolist(),"finite_te_base_included":True}


def parse_solution(case_dir, flow_conditions, target_x, cl_target=None, cl_tolerance=1e-3,
                   mesh_metadata=None, log_path=None, residual_thresholds=None,
                   force_window=50, force_drift_tolerance=1e-3, max_yplus=1.0,
                   force_integral_tolerance=5e-4, require_converged=True):
    """Return checked raw wall arrays and Cp/Cf labels only if all checks pass.

    With require_converged=False, an unfinished solution may be inspected but
    labels is None. This does not reclassify failed CFD as training data.
    """
    case = Path(case_dir).resolve()
    meta = _metadata(case, mesh_metadata)
    centres = np.asarray(meta.get("wall_face_centers", meta.get("airfoil_face_centres")), dtype=float)
    tangent = np.asarray(meta.get("wall_face_tangents", meta.get("airfoil_face_tangents")), dtype=float)
    n = len(centres)
    owner_cells = wall_owner_cells(case, meta, n)
    base_mask = meta.get("wall_face_base_mask", meta.get("airfoil_face_is_base"))
    surface_id = meta.get("airfoil_face_surface_id")
    if surface_id is None and "wall_face_source_segments" in meta:
        segments = np.asarray(meta["wall_face_source_segments"], dtype=int)
        require(segments.shape == (n, 2), "Invalid source-contour segment indices")
        mid = (len(target_x)-1)//2
        surface_id = np.full(n, "base", dtype="U5")
        surface_id[np.max(segments, axis=1) <= mid] = "lower"
        surface_id[np.min(segments, axis=1) >= mid] = "upper"
    time_dir = latest_time(case)
    p = read_boundary_field(time_dir/"p", count=n, case_root=case, owner_cells=owner_cells)
    tau = read_boundary_field(time_dir/"wallShearStress", count=n, case_root=case)
    density, density_source = None, None
    if tau["dimensions"] == KINEMATIC or p["dimensions"] == KINEMATIC:
        if (time_dir/"rho").is_file() or (time_dir/"rho.gz").is_file():
            rho = read_boundary_field(time_dir/"rho", count=n, case_root=case, owner_cells=owner_cells)
            require(rho["dimensions"] == DENSITY, "Unexpected wall density dimensions")
            density, density_source = rho["values"], "native_rho_wall"
        elif p["dimensions"] == PRESSURE:
            t = read_boundary_field(time_dir/"T", count=n, case_root=case, owner_cells=owner_cells)
            require(t["dimensions"] == TEMPERATURE, "Unexpected wall-temperature dimensions")
            gas_constant = flow_conditions.get("gas_constant", flow_conditions.get("R"))
            require(gas_constant is not None and float(gas_constant) > 0, "Gas constant is required for p/(R*T) wall density")
            density = p["values"]/(float(gas_constant)*t["values"])
            density_source = "native_wall_p_over_R_T"
    coeff = wall_coefficients(p["values"], p["dimensions"], tau["values"], tau["dimensions"],
                              tangent, flow_conditions, rho_wall=density)
    log = _inside(case, log_path or case/"log.rhoSimpleFoam")
    residual = parse_solver_log(log)
    forces = read_force_history(case)
    issues = []
    if residual["fatal_error"] or not residual["ended"]:
        issues.append("solver_log_fatal_or_incomplete")
    thresholds = residual_thresholds or {"p":1e-4,"U":1e-5,"h":1e-5,"k":1e-4,"omega":1e-4}
    normalized = {}
    for field, value in residual["initial_residuals"].items():
        key = "U" if field in ("Ux","Uy","Uz","U") else "h" if field in ("h","e") else field
        normalized[key] = max(normalized.get(key, 0.0), value)
    for field, threshold in thresholds.items():
        if field not in normalized or not np.isfinite(normalized[field]) or normalized[field] > threshold:
            issues.append("nonlinear_residual_"+field)
    require(force_window >= 4, "force_window must be >=4")
    window = min(force_window, len(forces["time"]))
    drift = {}
    if len(forces["time"]) < force_window:
        issues.append("insufficient_force_history")
    for key in ("cl","cd"):
        segment = forces[key][-window:]
        half = len(segment)//2
        difference = abs(float(segment[:half].mean())-float(segment[half:].mean())) if half else math.inf
        span = float(np.ptp(segment))
        drift[key] = {"half_window_mean_difference":difference,"peak_to_peak":span,"window":window}
        if difference > force_drift_tolerance:
            issues.append("force_drift_"+key)
    cl, cd = float(forces["cl"][-1]), float(forces["cd"][-1])
    if abs(float(forces["time"][-1])-float(time_dir.name)) > 1e-9*max(1,abs(float(time_dir.name))):
        issues.append("wall_fields_not_at_final_force_time")
    if cl_target is not None and abs(cl-float(cl_target)) > cl_tolerance:
        issues.append("target_cl_not_reached")
    independent = integrate_wall_forces(meta,coeff["p_pa"],coeff["tau_body_pa"],flow_conditions)
    if independent is not None:
        independent["cl_difference_from_history"] = independent["cl"]-cl
        independent["cd_difference_from_history"] = independent["cd"]-cd
        if max(abs(independent["cl"]-cl),abs(independent["cd"]-cd)) > force_integral_tolerance:
            issues.append("wall_force_integral_differs_from_forceCoeffs")
    yplus = None
    if (time_dir/"yPlus").is_file() or (time_dir/"yPlus.gz").is_file():
        yp = read_boundary_field(time_dir/"yPlus", count=n, case_root=case)
        require(yp["dimensions"] == DIMENSIONLESS, "Unexpected yPlus dimensions")
        yplus = {"mean":float(yp["values"].mean()),"max":float(yp["values"].max())}
        if yplus["mean"] < 0 or yplus["max"] > max_yplus:
            issues.append("wall_yplus_exceeds_limit")
    else:
        issues.append("missing_wall_yplus")
    labels, interpolation = interpolate_wall_labels(centres, coeff["cp"], coeff["cf"], target_x,
                              surface_id=surface_id, is_base=base_mask)
    converged = not issues
    if require_converged:
        require(converged, "CFD solution is not label-ready: "+", ".join(issues))
    return {"labels": labels if converged else None, "converged":converged,
            "cl":cl,"cd":cd,"alpha_deg":float(flow_conditions.get("alpha_deg",0.0)),
            "cl_target_error":None if cl_target is None else cl-float(cl_target),
            "time":time_dir.name,"residuals":residual,"nonlinear_residuals":normalized,
            "force_drift":drift,"independent_wall_force_integral":independent,
            "yplus":yplus,"quality":{"issues":issues,"interpolation":interpolation,
                  "pressure_dimensions":p["dimensions"],"shear_dimensions":tau["dimensions"],
                  "pressure_value_source":p["value_source"],
                  "wall_density_source":density_source,"shear_sign":coeff["shear_sign"],
                  "max_yplus_limit":max_yplus},
            "raw_wall":{"face_centres":centres,"tangent":tangent,"p_pa":coeff["p_pa"],
                        "native_wall_shear":tau["values"],"tau_body_pa":coeff["tau_body_pa"],
                        "cp":coeff["cp"],"cf":coeff["cf"]}, "force_history":forces}
