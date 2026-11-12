"""Deterministic, geometry-preserving 2-D airfoil O-grid for OpenFOAM.

The wall is the supplied polygon, including a straight finite-thickness TE base.
Only off-wall mesh lines are regularized: an isotonic normal-angle map gives
ordered outward rays, and a bounded normal offset preserves wall orthogonality.
No gmsh, external executable, temporary directory, or geometry fitting is used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parent


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _wall(x, y):
    x, y = np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)
    _require(x.shape == y.shape and len(x) >= 20, "Expected matching airfoil coordinate vectors")
    p = np.column_stack((x, y))
    _require(np.isfinite(p).all(), "Nonfinite airfoil coordinates")
    _require(np.ptp(x) > 0 and np.ptp(y) > 0, "Degenerate airfoil")
    source = np.arange(len(p), dtype=int)
    closed_te = np.linalg.norm(p[-1] - p[0]) < 1e-12
    if closed_te:
        p, source = p[:-1], source[:-1]
    edge = np.roll(p, -1, axis=0) - p
    length = np.linalg.norm(edge, axis=1)
    _require((length > 1e-12).all(), "Repeated wall vertex or zero-length wall edge")
    area = 0.5 * _cross(p, np.roll(p, -1, axis=0)).sum()
    _require(area < 0, "Wall must run lower TE -> LE -> upper TE (clockwise)")
    le = int(np.argmin(p[:, 0]))
    _require(0 < le < len(p)-1, "Cannot locate leading edge between both surfaces")
    normals = np.column_stack((-edge[:, 1], edge[:, 0])) / length[:, None]
    vertex_normals = normals + np.roll(normals, 1, axis=0)
    norm = np.linalg.norm(vertex_normals, axis=1)
    _require((norm > 1e-8).all(), "Wall has a reversing cusp with undefined outward normal")
    vertex_normals /= norm[:, None]
    # Miter offsets have exactly the same perpendicular height above BOTH
    # adjoining wall edges. Unit bisectors would make a thin TE-base cell a
    # wedge: its area-weighted centre then shifts along the wall by O(base_gap)
    # despite O(1e-6) wall distance, producing nearly 90-degree nonorthogonality.
    projection = np.sum(vertex_normals*normals, axis=1)
    _require((projection > 1e-5).all(), "Wall corner is too sharp for a bounded miter offset")
    vertex_normals /= projection[:, None]
    segments = np.column_stack((source, np.roll(source, -1)))
    if closed_te:
        segments[-1, 1] = len(x)-1
    base = np.zeros(len(p), dtype=bool)
    if not closed_te:
        base[-1] = True
    surface = np.where(np.arange(len(p)) < le, "lower", "upper")
    surface[base] = "base"
    return p, edge, length, vertex_normals, segments, base, surface


def _radial_distances(count, first, extent):
    _require(count >= 30 and first > 0 and extent > count * first, "Invalid radial grid parameters")
    low, high = 1.0+1e-10, 3.0
    for _ in range(90):
        middle = .5*(low+high)
        total = first*np.expm1(count*np.log(middle))/np.expm1(np.log(middle))
        if total < extent:
            low = middle
        else:
            high = middle
    growth = .5*(low+high)
    distance = np.r_[0.0, first * np.expm1(np.arange(1, count+1)*np.log(growth)) / np.expm1(np.log(growth))]
    return distance, growth


def _isotonic_decreasing(values):
    """Unweighted pool-adjacent-violators projection; NumPy-only runtime."""
    levels, counts = [], []
    for value in -np.asarray(values, dtype=float):
        levels.append(float(value))
        counts.append(1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            count = counts[-2]+counts[-1]
            level = (counts[-2]*levels[-2]+counts[-1]*levels[-1])/count
            levels[-2:] = [level]
            counts[-2:] = [count]
    return -np.repeat(levels,counts)


def _grid_quality(grid):
    a, b = grid[:-1], np.roll(grid[:-1], -1, axis=1)
    c, d = np.roll(grid[1:], -1, axis=1), grid[1:]
    quad = np.stack((a, b, c, d), axis=2)
    next_quad = np.roll(quad, -1, axis=2)
    edge = next_quad - quad
    corner = _cross(edge, np.roll(edge, -1, axis=2))
    cross = _cross(quad, next_quad)
    area = .5 * cross.sum(axis=2)
    _require((area > 0).all() and (corner > 0).all(), "O-grid has nonconvex or nonpositive cells")
    # Exact centroid of the planar polygon, hence the exact extruded cell centre.
    centre = ((quad + next_quad) * cross[..., None]).sum(axis=2) / (6*area[..., None])
    angles, skewness, face_weights = [], [], []

    def measure(start, stop, left, right=None):
        tangent = stop-start
        normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
        fc = .5*(start+stop)
        if right is None:
            delta = fc-left
        else:
            delta = right-left
        norm_n = np.linalg.norm(normal, axis=-1)
        norm_d = np.linalg.norm(delta, axis=-1)
        cosine = np.abs(np.sum(normal*delta, axis=-1))/(norm_n*norm_d)
        angles.append(np.degrees(np.arccos(np.clip(cosine, 0, 1))).ravel())
        if right is not None:
            denominator = np.sum(normal*delta, axis=-1)
            d0 = np.abs(np.sum(normal*(fc-left),axis=-1))
            d1 = np.abs(np.sum(normal*(right-fc),axis=-1))
            face_weights.append((np.minimum(d0,d1)/np.maximum(d0+d1,1e-30)).ravel())
            t = np.sum(normal*(fc-left), axis=-1)/denominator
            intersection = left+t[..., None]*delta
            scale = 2*np.minimum(np.linalg.norm(intersection-left, axis=-1), np.linalg.norm(right-intersection, axis=-1))
            skewness.append((np.linalg.norm(fc-intersection, axis=-1)/np.maximum(scale, 1e-30)).ravel())

    measure(grid[1:-1], np.roll(grid[1:-1], -1, axis=1), centre[:-1], centre[1:])
    measure(grid[:-1], grid[1:], np.roll(centre, 1, axis=1), centre)
    measure(grid[0], np.roll(grid[0], -1, axis=0), centre[0])
    measure(grid[-1], np.roll(grid[-1], -1, axis=0), centre[-1])
    values = np.concatenate(angles)
    circum_ratio = np.minimum(area,np.roll(area,-1,axis=1))/np.maximum(area,np.roll(area,-1,axis=1))
    radial_ratio = np.minimum(area[:-1],area[1:])/np.maximum(area[:-1],area[1:])
    return {"min_cell_area_2d": float(area.min()), "max_cell_area_2d": float(area.max()),
            "min_corner_jacobian_2d": float(corner.min()),
            "max_nonorthogonality_estimate_deg": float(values.max()),
            "mean_nonorthogonality_estimate_deg": float(values.mean()),
            "p99_nonorthogonality_estimate_deg": float(np.percentile(values, 99)),
            "max_internal_skewness_estimate": float(np.concatenate(skewness).max()),
            "min_internal_face_weight_estimate":float(np.concatenate(face_weights).min()),
            "min_neighbor_volume_ratio":float(min(circum_ratio.min(),radial_ratio.min())),
            "cell_centres_xy": centre, "areas_2d": area}


def make_grid(x, y, radial_cells=100, first_layer=1e-6, farfield=20.0,
              normal_weight=.8, normal_offset_scale=.03, normal_angle_smoothing=2.0):
    """Build and quality-check every quadrilateral before any case file is written."""
    p, edge, length, normal, segments, base, surface = _wall(x, y)
    _require(0 < normal_weight < 1, "normal_weight must be strictly between zero and one")
    angles = np.unwrap(np.arctan2(normal[:, 1], normal[:, 0]))
    monotone = _isotonic_decreasing(angles)
    _require(monotone[0] - monotone[-1] < 2*np.pi, "Wall normal winding is not simple")
    arc = np.r_[0.0, np.cumsum(length[:-1])]/length.sum()
    ordered = normal_weight*monotone+(1-normal_weight)*(monotone[0]-2*np.pi*arc)
    # Smooth the PERIODIC LIFT of the angular map, not the wall geometry. Abrupt
    # curvature jumps at the TE base/LE otherwise fan one off-wall column into a
    # large wedge next to a tiny column, causing bad volume ratios/face weights.
    _require(normal_angle_smoothing > 0,"Angular smoothing scale must be positive")
    radius = int(math.ceil(4*normal_angle_smoothing))
    offsets = np.arange(-radius,radius+1)
    weights = np.exp(-.5*(offsets/normal_angle_smoothing)**2)
    weights /= weights.sum()
    ids = np.arange(len(p))
    ordered = sum(weight*(ordered[(ids+offset)%len(p)]-2*np.pi*np.floor_divide(ids+offset,len(p)))
                  for offset,weight in zip(offsets,weights))
    rays = np.column_stack((np.cos(ordered), np.sin(ordered)))
    distances, growth = _radial_distances(radial_cells, first_layer, farfield)
    # a'(h)>=0 and [h-a(h)]'>=0. This avoids the negative radial Jacobian
    # introduced by naive exponential blending of normal and far-field rays.
    h = distances[:, None, None]
    a = normal_offset_scale*np.tanh(h/normal_offset_scale)
    grid = p[None, :, :]+a*normal[None, :, :]+(h-a)*rays[None, :, :]
    quality = _grid_quality(grid)
    _require(quality["max_nonorthogonality_estimate_deg"] < 75,
             f"Mesh nonorthogonality too large: {quality['max_nonorthogonality_estimate_deg']:.3f} deg")
    _require(quality["max_internal_skewness_estimate"] < 4,
             f"Mesh skewness too large: {quality['max_internal_skewness_estimate']:.3f}")
    _require(quality["min_neighbor_volume_ratio"] > .01,
             f"Adjacent volume ratio too small: {quality['min_neighbor_volume_ratio']:.5g}")
    _require(quality["min_internal_face_weight_estimate"] > .05,
             f"Internal face weight too small: {quality['min_internal_face_weight_estimate']:.5g}")
    return {"grid": grid, "wall": p, "wall_edges": edge, "wall_edge_lengths": length,
            "source_segments": segments, "base_mask": base, "surface_id": surface,
            "growth": growth, "distances": distances, "quality": quality,
            "normal_weight": normal_weight, "normal_offset_scale": normal_offset_scale,
            "normal_angle_smoothing":normal_angle_smoothing}


def _header(kind, name):
    return ("/* Geometry-preserving extruded airfoil O-grid */\nFoamFile\n{\n"
            f"    version 2.0;\n    format ascii;\n    class {kind};\n    object {name};\n}}\n\n")


def _write_list(path, kind, values, formatter, project):
    _require(path.resolve().is_relative_to(project),f"Output file escapes the allowed project: {path.name}")
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(_header(kind, path.name))
        stream.write(f"{len(values)}\n(\n")
        for value in values:
            stream.write(formatter(value)+"\n")
        stream.write(")\n")


def write_mesh(case_dir, x, y, radial_cells=100, first_layer=1e-6, farfield=20.0,
               span=.01, project_root=None, **options):
    """Write constant/polyMesh; return JSON-compatible wall/quality metadata.

    All output must lie below this module's project directory. The input wall
    polygon is unchanged: reported geometric error is exactly zero at vertices.
    Quality numbers are analytical estimates; native checkMesh remains required.
    """
    case = Path(case_dir).resolve()
    project = Path(project_root).resolve() if project_root is not None else WORKSPACE
    _require(case.is_relative_to(project), "Case output must stay inside Projects/floGen")
    _require(span > 0, "Extrusion span must be positive")
    mesh = make_grid(x, y, radial_cells, first_layer, farfield, **options)
    grid, wall = mesh["grid"], mesh["wall"]
    nr, n = radial_cells, len(wall)
    plane_count = (nr+1)*n
    points = np.empty((2*plane_count, 3))
    for k, z in enumerate((-span/2, span/2)):
        points[k*plane_count:(k+1)*plane_count, :2] = grid.reshape(-1, 2)
        points[k*plane_count:(k+1)*plane_count, 2] = z

    def point(k, j, i):
        return k*plane_count+j*n+(i % n)

    internal = []
    for j in range(nr):
        for i in range(n):
            owner, neighbor = j*n+i, j*n+(i+1) % n
            face = [point(0,j,i+1),point(0,j+1,i+1),point(1,j+1,i+1),point(1,j,i+1)]
            if owner > neighbor:
                owner, neighbor, face = neighbor, owner, face[::-1]
            internal.append((owner, neighbor, face))
    for j in range(nr-1):
        for i in range(n):
            face = [point(0,j+1,i+1),point(0,j+1,i),point(1,j+1,i),point(1,j+1,i+1)]
            internal.append((j*n+i,(j+1)*n+i,face))
    internal.sort(key=lambda item: (item[0], item[1]))
    faces = [item[2] for item in internal]
    owners = [item[0] for item in internal]
    neighbors = [item[1] for item in internal]
    patches = {}
    patches["airfoil"] = {"type":"wall","startFace":len(faces),"nFaces":n}
    for i in range(n):
        faces.append([point(0,0,i),point(0,0,i+1),point(1,0,i+1),point(1,0,i)])
        owners.append(i)
    patches["farfield"] = {"type":"patch","startFace":len(faces),"nFaces":n}
    for i in range(n):
        faces.append([point(0,nr,i+1),point(0,nr,i),point(1,nr,i),point(1,nr,i+1)])
        owners.append((nr-1)*n+i)
    patches["frontAndBack"] = {"type":"empty","startFace":len(faces),"nFaces":2*nr*n}
    for j in range(nr):
        for i in range(n):
            faces.append([point(0,j,i),point(0,j+1,i),point(0,j+1,i+1),point(0,j,i+1)])
            owners.append(j*n+i)
            faces.append([point(1,j,i),point(1,j,i+1),point(1,j+1,i+1),point(1,j+1,i)])
            owners.append(j*n+i)
    poly = case / "constant" / "polyMesh"
    _require(poly.resolve().is_relative_to(project), "polyMesh symlink escapes the allowed project")
    _require(not poly.exists(), "Refusing to replace an existing polyMesh")
    metadata_path = case / "mesh_metadata.json"
    _require(metadata_path.resolve().is_relative_to(project), "Metadata symlink escapes the allowed project")
    _require(not metadata_path.exists(), "Refusing to replace existing mesh metadata")
    poly.mkdir(parents=True)
    _write_list(poly/"points", "vectorField", points, lambda p: "("+" ".join(f"{v:.17g}" for v in p)+")",project)
    _write_list(poly/"faces", "faceList", faces, lambda f: "4("+" ".join(map(str,f))+")",project)
    _write_list(poly/"owner", "labelList", owners, str,project)
    _write_list(poly/"neighbour", "labelList", neighbors, str,project)
    _require((poly/"boundary").resolve().is_relative_to(project),"Boundary file escapes the allowed project")
    with (poly/"boundary").open("x",encoding="ascii",newline="\n") as stream:
        stream.write(_header("polyBoundaryMesh", "boundary"))
        stream.write("3\n(\n")
        for name, patch in patches.items():
            stream.write(f"{name}\n{{\n    type {patch['type']};\n    nFaces {patch['nFaces']};\n    startFace {patch['startFace']};\n}}\n")
        stream.write(")\n")
    centres = np.column_stack((.5*(wall+np.roll(wall,-1,axis=0)),np.zeros(n)))
    tangent = np.column_stack((mesh["wall_edges"]/mesh["wall_edge_lengths"][:,None],np.zeros(n)))
    quality = {key:value for key,value in mesh["quality"].items() if not isinstance(value,np.ndarray)}
    normals = np.column_stack((-mesh["wall_edges"][:,1],mesh["wall_edges"][:,0]))/mesh["wall_edge_lengths"][:,None]
    first_centre_distance = np.abs(np.sum((mesh["quality"]["cell_centres_xy"][0]-centres[:,:2])*normals,axis=1))
    metadata = {
        "mesh_kind":"extruded_geometry_preserving_structured_O_grid", "cells":nr*n,
        "points":len(points),"faces":len(faces),"internal_faces":len(neighbors),
        "radial_cells":nr,"circumferential_cells":n,"span":float(span),
        "extrusion_height":float(span),"first_layer_vertex_spacing":float(first_layer),
        "radial_growth":float(mesh["growth"]),"farfield_offset":float(farfield),
        "normal_weight":mesh["normal_weight"],"normal_offset_scale":mesh["normal_offset_scale"],
        "normal_angle_smoothing":mesh["normal_angle_smoothing"],
        "first_cell_centre_wall_distance_min":float(first_centre_distance.min()),
        "first_cell_centre_wall_distance_max":float(first_centre_distance.max()),
        "patches":patches,"wall_points_xy":wall.tolist(),
        "airfoil_face_centres":centres.tolist(),"airfoil_face_centers":centres.tolist(),
        "airfoil_face_tangents":tangent.tolist(),"airfoil_face_is_base":mesh["base_mask"].tolist(),
        "airfoil_face_surface_id":mesh["surface_id"].tolist(),
        "wall_face_centers":centres.tolist(),"wall_face_tangents":tangent.tolist(),
        "wall_face_base_mask":mesh["base_mask"].tolist(),
        "wall_face_source_segments":mesh["source_segments"].tolist(),
        "wall_face_indices":list(range(patches["airfoil"]["startFace"],patches["airfoil"]["startFace"]+n)),
        "wall_order":"lower_TE_to_LE_then_upper_to_TE_then_finite_gap_base",
        "wall_vertex_geometry_error":0.0,"wall_geometry_rule":"Exact supplied polygon; finite TE closed by straight wall base; no rounding/fitting",
        "min_cell_volume":quality["min_cell_area_2d"]*span,
        "quality_estimates":quality,"native_checkMesh_required":True,
    }
    metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=WORKSPACE/"build/superwing_2d_inputs_complete")
    parser.add_argument("--split",choices=("train","test"),default="train")
    parser.add_argument("--row",type=int,default=0)
    parser.add_argument("--case-dir",type=Path,required=True)
    parser.add_argument("--radial-cells",type=int,default=100)
    args=parser.parse_args()
    x=np.load(args.input_root/"x.npy",allow_pickle=False)
    y=np.load(args.input_root/args.split/"geometry.npy",mmap_mode="r",allow_pickle=False)[args.row,0]
    result=write_mesh(args.case_dir,x,y,radial_cells=args.radial_cells)
    print(json.dumps({key:result[key] for key in ("cells","points","faces","patches","quality_estimates","min_cell_volume","wall_vertex_geometry_error")},indent=2))


if __name__ == "__main__":
    main()
