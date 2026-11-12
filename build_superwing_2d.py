"""Build provenance-preserving effective 2-D CFD inputs from SuperWing.

The output contains geometry and prescribed operating conditions, not invented
2-D pressure/friction labels. A separate, genuine 2-D solver must produce those
labels. Source arrays are always opened read-only with memory mapping.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = WORKSPACE / "build" / "superwing_2d_inputs"
PROVENANCE_DTYPE = np.dtype([
    ("source_row", "<i8"), ("source_geometry_index", "<i4"),
    ("original_id", "<i8"), ("source_condition_id", "<i2"),
    ("family_id", "<i4"), ("section_index", "<i2"),
    ("extra_ratio_adjustment", "?"), ("eta", "<f8"),
    ("eta_exposed", "<f8"), ("cos_sweep", "<f8"),
    ("quarter_chord_sweep_deg", "<f8"), ("chord", "<f8"),
    ("sectional_cl_3d", "<f8"), ("mach_3d", "<f8"),
    ("reynolds_3d", "<f8"), ("temperature_kelvin", "<f8"),
    ("aoa_3d_deg", "<f8"), ("chord_angle_xy_deg", "<f8"),
    ("leading_edge_x", "<f8"), ("leading_edge_y", "<f8"),
    ("span_z", "<f8"),
])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def chordwise_grid(points=321):
    require(points >= 5 and points % 2 == 1, "Point count must be odd and >=5")
    k = np.linspace(0.0, 1.0, (points + 1) // 2)
    a0, a1 = 0.0079, 0.96
    half = ((np.cos(a0 * np.pi) - np.cos((a0 * (1-k) + a1*k) * np.pi))
            / (np.cos(a0 * np.pi) - np.cos(a1 * np.pi)))
    return np.r_[half[::-1], half[1:]], half


def primary_sections(span_cells=128, count=12):
    require(0 < count <= span_cells, "Invalid section count")
    # Pick each equal-width span-bin's center, avoiding exact root/tip planes.
    selected = np.floor((np.arange(count) + 0.5) * span_cells / count).astype(np.int64)
    require(len(np.unique(selected)) == count, "Repeated primary section index")
    return selected


def _frame(points):
    te = 0.5 * (points[0] + points[-1])
    le_index = int(np.argmin(points[:, 0]))
    direction = te - points[le_index]
    require(np.linalg.norm(direction) > 0, "Zero initial section chord")
    le_index = int(np.argmin(points @ direction))
    require(0 < le_index < len(points)-1, "Leading edge must separate lower and upper surfaces")
    le = points[le_index]
    direction = te - le
    chord = float(np.linalg.norm(direction))
    require(chord > 0, "Zero section chord")
    ex = direction / chord
    ey = np.array([-ex[1], ex[0]])
    xy = np.stack(((points-le) @ ex / chord, (points-le) @ ey / chord), axis=-1)
    return le_index, le, direction, chord, xy


def _interp_surface(surface, x):
    order = np.argsort(surface[:, 0], kind="stable")
    xs, ys = surface[order].T
    xs, unique = np.unique(xs, return_index=True)
    require(len(xs) > 1, "Degenerate sectional surface")
    return np.interp(x, xs, ys[unique])


def extract_sections(centers, vertices, points=321):
    """Extract detwisted sections and strip-local quarter-chord sweep in float64."""
    c, v = np.asarray(centers), np.asarray(vertices)
    require(c.shape == (3,128,256) and v.shape == (3,129,257), "Unexpected native geometry shape")
    require(np.isfinite(c).all() and np.isfinite(v).all(), "Nonfinite native geometry")
    require(np.ptp(v[2], axis=-1).max() < 1e-8, "Native span sections must be planar in x-y")
    require(np.allclose(v[:,:,0], v[:,:,-1], atol=1e-8, rtol=0), "Native contour is not closed")
    mid = 0.5 * (v[:,:-1,:] + v[:,1:,:])
    expected = 0.5 * (mid[:,:,:-1] + mid[:,:,1:])
    require(np.allclose(c, expected, atol=1e-8, rtol=0), "Centers are not four-vertex averages")
    z_boundaries = np.asarray(v[2,:,0], dtype=np.float64)
    dz = np.diff(z_boundaries)
    require((dz > 0).all(), "Span coordinates must increase")
    quarters = []
    for j in range(129):
        _, le, direction, _, _ = _frame(np.asarray(v[:2,j,:-1], dtype=np.float64).T)
        quarters.append(le + 0.25*direction)
    slope = np.diff(np.asarray(quarters)[:,0]) / dz
    cosine = 1.0 / np.sqrt(1.0 + slope*slope)
    _, half = chordwise_grid(points)
    y, chords, twist, leading = [], [], [], []
    for j in range(128):
        le_index, le, direction, chord, xy = _frame(mid[:2,j,:-1].T)
        lower = _interp_surface(xy[:le_index+1], half)
        upper = _interp_surface(xy[le_index:], half)
        profile = np.r_[lower[::-1], upper[1:]]
        require(np.isfinite(profile).all(), "Nonfinite normalized section")
        y.append(profile)
        chords.append(chord)
        twist.append(math.degrees(math.atan2(direction[1], direction[0])))
        leading.append(le)
    return {"foil_y": np.asarray(y), "chord": np.asarray(chords),
            "cos_sweep": cosine, "quarter_chord_sweep_deg": np.degrees(np.arctan(slope)),
            "chord_angle_xy_deg": np.asarray(twist), "leading_edge": np.asarray(leading),
            "span_z": 0.5*(z_boundaries[:-1]+z_boundaries[1:]),
            "eta_exposed": (np.arange(128)+0.5)/128.0}


def make_family_split(configs, index, seed=42):
    """Keep every rounded baseline-CST family, wing and condition in one split."""
    unique, geometry_family = np.unique(np.round(configs[:,18:38], 10), axis=0, return_inverse=True)
    family_count = len(unique)
    count = max(1, min(family_count-1, round(family_count*0.1))) if family_count > 1 else 0
    test_families = np.random.default_rng(seed).permutation(family_count)[:count]
    adjustments = []
    canonical = (family_count == 1510 and len(configs) == 4239 and len(index) == 28856 and seed == 42)
    if canonical:
        require(1104 in test_families and 195 not in test_families,
                "Canonical CST-family split differs; source may have changed")
        test_families = np.r_[test_families[test_families != 1104], 195]
        adjustments = [{"removed_from_test": 1104, "added_to_test": 195,
                        "reason": "Balance source-case counts while keeping whole CST families"}]
    row_family = geometry_family[np.asarray(index[:,0], dtype=np.int64)]
    is_test = np.isin(row_family, test_families)
    if canonical:
        require(int(is_test.sum()) == 2885, "Canonical test-source-case count differs")
    return {"geometry_family": geometry_family, "row_family": row_family,
            "is_test": is_test, "test_family_ids": np.sort(test_families),
            "family_count": family_count, "canonical": canonical,
            "family_swaps": adjustments, "unique_cst": unique}


def ratio_extra_sections(split, selected_rows, primary, seed=20261109):
    """Eight additional test inputs yield the requested exact final 9:1 ratio."""
    if not split["canonical"] or len(selected_rows) != len(split["is_test"]) or len(primary) != 12:
        return {}
    require(12*25971 == 9*(12*2885+8), "Ratio arithmetic is inconsistent")
    rng = np.random.default_rng(seed)
    test_rows = np.flatnonzero(split["is_test"])
    chosen = rng.choice(test_rows, size=8, replace=False)
    available = np.setdiff1d(np.arange(128), primary)
    return {int(row): int(rng.choice(available)) for row in chosen}


def read_working_conditions(source):
    records = {}
    with (source / "running_info.jsonl").open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            original = int(value["running_id"])
            require(original not in records, "Duplicate running ID in native logs")
            records[original] = {int(key.split("_",1)[0]): info["working_conditions"]
                                 for key, info in value.items()
                                 if key.endswith("_running_info") and isinstance(info, dict)
                                 and "working_conditions" in info}
    return records


def load_source(source):
    source = Path(source).resolve()
    configs = np.loadtxt(source / "configs.dat", skiprows=1, ndmin=2)
    require(configs.shape[1] == 55, "Expected native 55-column configurations")
    arrays = {name: np.load(source / (name+".npy"), mmap_mode="r", allow_pickle=False)
              for name in ("index", "geom0", "origingeom", "data")}
    n, g = len(arrays["index"]), len(configs)
    require(arrays["index"].shape == (n,12), "Unexpected native index shape")
    require(arrays["geom0"].shape == (g,3,128,256), "Unexpected native center shape")
    require(arrays["origingeom"].shape == (g,3,129,257), "Unexpected native vertex shape")
    require(arrays["data"].shape == (n,3,128,256), "Unexpected native field shape")
    require(np.isfinite(configs).all() and np.isfinite(arrays["index"]).all(), "Invalid native metadata")
    require(np.equal(arrays["index"][:,:2], np.round(arrays["index"][:,:2])).all(), "Noninteger source IDs")
    require(((arrays["index"][:,0] >= 0) & (arrays["index"][:,0] < g)).all(), "Invalid geometry indices")
    arrays["configs"] = configs
    return arrays


def _metadata(source, arrays):
    result = {}
    for name in ("configs.dat", "index.npy", "geom0.npy", "origingeom.npy", "data.npy", "running_info.jsonl"):
        path = source / name
        stat = path.stat()
        result[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if stat.st_size < 30_000_000:
            result[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def generate(source, output, points=321, sections=12, seed=42, max_source_cases=None,
             progress_seconds=10.0):
    import torch
    from main import sectional_lift

    source, output = Path(source).resolve(), Path(output).resolve()
    require(output.is_relative_to(WORKSPACE), "This builder only writes inside the floGen workspace")
    require(output != source and not output.is_relative_to(source) and not source.is_relative_to(output),
            "Source and destination must not overlap")
    require(not output.exists(), "Destination already exists; choose a fresh staging directory")
    arrays = load_source(source)
    configs, index = arrays["configs"], arrays["index"]
    split = make_family_split(configs, index, seed)
    total_source_rows = len(index)
    rows = np.arange(total_source_rows, dtype=np.int64)
    if max_source_cases is not None:
        require(max_source_cases > 0, "max-source-cases must be positive")
        rows = rows[:max_source_cases]
    primary = primary_sections(128, sections)
    extras = ratio_extra_sections(split, rows, primary)
    counts = {name: int(np.count_nonzero(split["is_test"][rows] == flag))*sections
              for name, flag in (("train",False),("test",True))}
    counts["test"] += len(extras)
    source_metadata = _metadata(source, arrays)
    logs = read_working_conditions(source)
    # Validate every selected case's conditions before creating output files.
    for row in rows:
        g, k = map(int, index[row,:2])
        original = int(configs[g,0])
        wc = logs[original][k]
        require(float(wc["re"]) > 0 and float(wc["temp"]) > 0, "Invalid native Re/temperature")
        require(np.isclose(float(wc["M3"]), index[row,3], atol=2e-9, rtol=0), "Logged Mach differs")
        require(np.isclose(float(wc["AA"]), index[row,2], atol=2e-9, rtol=0), "Logged AoA differs")
    output.mkdir(parents=True)
    _json(output / "BUILD_STARTED.json", {"status_at_start": "building", "labels_status_at_start": "not_computed"})
    x, _ = chordwise_grid(points)
    np.save(output / "x.npy", x.astype(np.float64), allow_pickle=False)
    np.save(output / "geometry_family_ids.npy", split["geometry_family"], allow_pickle=False)
    np.save(output / "test_family_ids.npy", split["test_family_ids"], allow_pickle=False)
    np.save(output / "family_cst_parameters.npy", split["unique_cst"], allow_pickle=False)
    buffers = {}
    for name in ("train","test"):
        folder = output / name
        folder.mkdir()
        n = counts[name]
        buffers[name] = {
            "geometry": np.lib.format.open_memmap(folder / "geometry.npy", mode="w+", dtype="float32", shape=(n,1,points)),
            "conditions": np.lib.format.open_memmap(folder / "conditions.npy", mode="w+", dtype="float64", shape=(n,3)),
            "provenance": np.lib.format.open_memmap(folder / "provenance.npy", mode="w+", dtype=PROVENANCE_DTYPE, shape=(n,)),
        }
    positions = {"train":0,"test":0}
    ranges = {name: {key:[math.inf,-math.inf] for key in ("mach_2d","cl_2d","reynolds_2d","tc_effective")}
              for name in buffers}
    begin, last_print = time.monotonic(), time.monotonic()
    geometry_cache = {}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for done, row in enumerate(rows,1):
            row = int(row)
            g, k = map(int,index[row,:2])
            original = int(configs[g,0])
            name = "test" if split["is_test"][row] else "train"
            if g not in geometry_cache:
                # Native rows are contiguous by shape; a small bounded cache also
                # supports arbitrary valid order without retaining the whole grid.
                if len(geometry_cache) >= 8:
                    geometry_cache.pop(next(iter(geometry_cache)))
                geometry_cache[g] = extract_sections(arrays["geom0"][g],arrays["origingeom"][g],points)
            f = geometry_cache[g]
            field = np.array(arrays["data"][row], dtype=np.float64, copy=True)
            require(np.isfinite(field).all(), f"Nonfinite source CFD field at row {row}")
            cl3d = sectional_lift(arrays["origingeom"][g], field, f["chord"], index[row,2])
            js = np.r_[primary,extras[row]] if row in extras else primary
            cos = f["cos_sweep"][js]
            wc = logs[original][k]
            re3d = float(wc["re"])
            y = f["foil_y"][js] / cos[:,None]
            conditions = np.column_stack((index[row,3]*cos, cl3d[js]/cos**2, re3d*f["chord"][js]*cos))
            require(np.isfinite(conditions).all() and (conditions[:,[0,2]] > 0).all(), "Invalid transformed conditions")
            a, b = positions[name], positions[name] + len(js)
            buffers[name]["geometry"][a:b,0] = y
            buffers[name]["conditions"][a:b] = conditions
            p = np.zeros(len(js), dtype=PROVENANCE_DTYPE)
            p["source_row"], p["source_geometry_index"] = row,g
            p["original_id"], p["source_condition_id"] = original,k
            p["family_id"] = split["geometry_family"][g]
            p["section_index"] = js
            p["extra_ratio_adjustment"] = ~np.isin(js,primary)
            p["eta"] = f["span_z"][js] / index[row,5]
            p["eta_exposed"] = f["eta_exposed"][js]
            p["cos_sweep"] = cos
            p["quarter_chord_sweep_deg"] = f["quarter_chord_sweep_deg"][js]
            p["chord"] = f["chord"][js]
            p["sectional_cl_3d"] = cl3d[js]
            p["mach_3d"],p["reynolds_3d"] = index[row,3],re3d
            p["temperature_kelvin"],p["aoa_3d_deg"] = float(wc["temp"]),index[row,2]
            p["chord_angle_xy_deg"] = f["chord_angle_xy_deg"][js]
            p["leading_edge_x"],p["leading_edge_y"] = f["leading_edge"][js].T
            p["span_z"] = f["span_z"][js]
            buffers[name]["provenance"][a:b] = p
            positions[name] = b
            half_len = (points+1)//2
            thickness = (y[:,half_len-1:] - y[:,:half_len][:,::-1]).max(axis=1)
            for key,values in zip(("mach_2d","cl_2d","reynolds_2d","tc_effective"),
                                  (*conditions.T,thickness)):
                ranges[name][key][0] = min(ranges[name][key][0],float(values.min()))
                ranges[name][key][1] = max(ranges[name][key][1],float(values.max()))
            now = time.monotonic()
            if now-last_print >= progress_seconds or done == len(rows):
                print(json.dumps({"source_cases_done":done,"source_cases_total":len(rows),
                                  "written":positions,"elapsed_seconds":round(now-begin,1)}),flush=True)
                last_print = now
    finally:
        for split_buffers in buffers.values():
            for values in split_buffers.values():
                values.flush()
        torch.set_num_threads(previous_threads)
    require(positions == counts,"Final write counts differ")
    for name in ranges:
        if counts[name] == 0:
            ranges[name] = {key: None for key in ranges[name]}
    canonical_full = split["canonical"] and len(rows) == total_source_rows and sections == 12
    if canonical_full:
        require(counts["train"] == 9*counts["test"],"Final train/test ratio is not 9:1")
    require(source_metadata == _metadata(source,arrays),"Source files changed during read-only build")
    manifest = {
        "status":"complete", "created_utc":datetime.now(timezone.utc).isoformat(),
        "dataset_kind":"effective_2d_cfd_inputs_from_superwing", "labels_status":"not_computed",
        "source_root":str(source), "source_files":source_metadata,
        "source_case_count":total_source_rows,"selected_source_case_count":len(rows),
        "source_geometry_count":len(configs),"cst_family_count":split["family_count"],
        "split_seed":seed,"split_group":"baseline_CST_20_parameters_rounded_10_decimals",
        "test_family_count":len(split["test_family_ids"]),"family_swaps":split["family_swaps"],
        "counts":counts,"exact_9_to_1":counts["train"] == 9*counts["test"],
        "primary_section_indices":primary.tolist(),"extra_test_sections":extras,
        "extra_section_seed":20261109,"point_count":points,
        "conditions_columns":["mach_2d","cl_2d","reynolds_2d_actual"],
        "temperature_source":"native_running_info_working_conditions_temp_kelvin",
        "c_ref":1.0,"reynolds_unit":"actual_dimensionless_number_not_millions",
        "sld_source":"cfd_reference_integral","sld_requires_3d_cfd":True,
        "sweep_definition":"quarter_chord_delta_x_over_delta_z_per_native_span_strip",
        "sweep_extension":"Local strip sweep generalizes the paper's single-segment global sweep to kink wings",
        "geometry_definition":"untwist chord frame, normalize by local chord, then divide y/c by cos(sweep)",
        "geometry_order":"lower_TE_to_LE_then_upper_LE_excluded_to_TE",
        "provenance_dtype":PROVENANCE_DTYPE.descr,"ranges":ranges,
        "elapsed_seconds":time.monotonic()-begin,"source_files_modified":False,
        "solver_labels_required":["Cp_2d","Cf_2d"],
        "warning":"Three-dimensional Cp/Cf are not independent two-dimensional CFD labels. No placeholder labels are written."
    }
    _json(output / "manifest.json",manifest)
    (output / "README.txt").write_text(_readme(manifest),encoding="utf-8")
    # The initial marker is retained as an immutable record, and COMPLETE is
    # written last; consumers must check manifest/COMPLETE, not marker absence.
    _json(output / "COMPLETE.json",{"status":"complete","labels_status":"not_computed","counts":counts})
    return manifest


def _json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def _readme(m):
    return ("SuperWing effective 2-D CFD inputs\n\n"
            "These are independent 2-D simulation INPUTS, not completed 2-D CFD training labels.\n"
            "No Cp/Cf placeholders or inverse-transformed 3-D labels are supplied.\n"
            "Run a real 2-D solver with the prescribed geometry, Mach, CL, Re and temperature.\n\n"
            "x.npy: shared chord-normalized circumferential x positions.\n"
            "train|test/geometry.npy: float32 [N,1,321] by default, effective y/c.\n"
            "train|test/conditions.npy: float64 [N,3], columns [Mach2D, CL2D, actual_Re2D].\n"
            "train|test/provenance.npy: structured source/section/condition metadata, no pickle.\n"
            "provenance.temperature_kelvin: native freestream temperature; chord_angle_xy_deg is atan2(TEy-LEy,TEx-LEx).\n"
            "eta=z/half_span; eta_exposed=(section_index+0.5)/128.\n"
            "family_cst_parameters.npy and geometry_family_ids.npy define rounded baseline-CST identity.\n\n"
            "Source Re uses c_ref=1. Effective Re=source_Re*local_chord*cos(sweep); actual units, not millions.\n"
            "Effective Ma=source_Ma*cos(sweep); CL=integrated_source_section_CL/cos(sweep)^2.\n"
            "The chord-frame profile is untwisted/normalized before y is divided by cos(sweep).\n"
            "Sweep is local quarter-chord-line dx/dz per source span strip, an extension for kink wings.\n"
            "Source CL is integrated from reference 3-D CFD, including real Cf multipliers 150/300.\n"
            "The original files remain unchanged. Complete source paths, stats/hashes and rules are in manifest.json.\n\n"
            f"Counts: train={m['counts']['train']}, test={m['counts']['test']}; exact9to1={m['exact_9_to_1']}.\n"
            "Whole baseline-CST families stay in one split, including all wings, conditions and sections.\n"
            "Canonical full build uses 12 evenly binned section centers per source case plus 8 extra test sections.\n"
            "CST family seed=42; one explicit whole-family swap balances source counts; extra section seed=20261109.\n"
            "For subset/synthetic runs exact ratio is not asserted. Check COMPLETE.json and manifest.status before use.\n")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("inspect","generate"))
    p.add_argument("--source-root",type=Path,default=Path("D:/SuperWing"))
    p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--points",type=int,default=321)
    p.add_argument("--sections",type=int,default=12)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-source-cases",type=int)
    p.add_argument("--progress-seconds",type=float,default=10.0)
    args=p.parse_args(argv)
    if args.command == "generate":
        result=generate(args.source_root,args.output_root,args.points,args.sections,args.seed,
                        args.max_source_cases,args.progress_seconds)
    else:
        a=load_source(args.source_root)
        split=make_family_split(a["configs"],a["index"],args.seed)
        result={"source_root":str(args.source_root.resolve()),
                "source_array_shapes":{k:list(v.shape) for k,v in a.items()},
                "family_count":split["family_count"],"test_family_count":len(split["test_family_ids"]),
                "source_train_cases":int((~split["is_test"]).sum()),
                "source_test_cases":int(split["is_test"].sum()),"family_swaps":split["family_swaps"],
                "primary_section_indices":primary_sections(128,args.sections).tolist(),
                "labels_status":"not_computed","files_written":False}
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        print(f"error: {exc}",file=sys.stderr,flush=True)
        raise SystemExit(2)
