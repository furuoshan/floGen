"""Run genuine XFOIL cases for the effective SuperWing 2-D input dataset.

All task files, including subprocess cwd/HOME/temp, must stay below the explicitly
selected project root. The installed XFOIL executable and its OS runtime are the
only expected external program resources. No remote connection is made here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import numpy as np


SCHEMA = "superwing-xfoil-v1"
RAW_FILES = ("foil.dat", "commands.in", "xfoil.log", "polar.dat", "cp.dat", "bl.dat")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, path):
    """Reject lexical traversal and symlinks resolving outside the task root."""
    root = Path(root).resolve(strict=True)
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    require(resolved == root or resolved.is_relative_to(root),
            f"Task path escapes project root: {path}")
    return resolved


def json_write(path, value):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def json_once(path, value):
    """Initialize immutable shared run metadata, safe for disjoint shards."""
    try:
        with path.open("x",encoding="utf-8") as handle:
            handle.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        return
    except FileExistsError:
        pass
    # A sibling coordinator might still be finishing its short initial write.
    for attempt in range(25):
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
            require(prior == value,"Output directory uses different immutable metadata")
            return
        except json.JSONDecodeError:
            if attempt == 24:
                raise
            time.sleep(.02)


def as_json(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def parse_table(path):
    """Read XFOIL text tables while retaining declared column names.

    Boundary-layer wake lines can have fewer columns than surface lines. Only
    rows containing every requested field will subsequently be used.
    """
    headers, rows = [], []
    for line in Path(path).read_text(encoding="ascii", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            words = text.lstrip("# ").split()
            if words:
                headers.append(words)
            continue
        try:
            row = [float(s.replace("D", "E").replace("d", "e"))
                   for s in text.split()]
        except ValueError:
            continue
        if row:
            rows.append(row)
    require(rows, f"No numerical rows in {path.name}")
    return headers, rows


def _column_header(headers, needed):
    for header in reversed(headers):
        lookup = {name.lower(): i for i, name in enumerate(header)}
        if all(name in lookup for name in needed):
            return lookup
    raise ValueError(f"XFOIL table has no recognized header for {needed}")


def read_cp(path):
    headers, rows = parse_table(path)
    header = _column_header(headers, ("x", "cp"))
    last = max(header["x"], header["cp"])
    data = np.asarray([[r[header["x"]], r[header["cp"]]]
                       for r in rows if len(r) > last], dtype=float)
    require(len(data) >= 20 and np.isfinite(data).all(), "Incomplete/nonfinite Cp output")
    return data


def read_boundary_layer(path):
    headers, rows = parse_table(path)
    header = _column_header(headers, ("s", "x", "y", "cf"))
    ue_names = ("ue/vinf", "ue/uinf", "ue", "ue/v")
    ue = next((header[name] for name in ue_names if name in header), None)
    require(ue is not None, "Boundary-layer output lacks signed edge velocity")
    indices = [header[name] for name in ("s", "x", "y", "cf")] + [ue]
    last = max(indices)
    data = np.asarray([[r[i] for i in indices] for r in rows if len(r) > last], dtype=float)
    require(len(data) >= 20 and np.isfinite(data).all(), "Incomplete/nonfinite boundary-layer output")
    return data  # s, x, y, Cf_raw, signed Ue/Vinf


def _interp(x, values, query):
    order = np.argsort(x, kind="stable")
    x, values = np.asarray(x)[order], np.asarray(values)[order]
    x, first = np.unique(x, return_index=True)
    require(len(x) >= 4 and x.min() <= 0.02 and x.max() >= 0.98,
            "Insufficient chordwise surface coverage")
    # Constant extension only bridges the small missing LE/TE endpoint interval;
    # no operating conditions, flow states, or failed solutions are invented.
    return np.interp(query, x, values[first])


def interpolate_labels(cp_path, dump_path, target_x):
    # CPWR contains only the airfoil surface. Match those same leading rows in
    # DUMP by their x coordinates, rather than guessing the wake boundary from
    # x<=1 (a wake point can still have x<=1 on nonstandard geometries).
    cp = read_cp(cp_path)
    bl = read_boundary_layer(dump_path)
    require(len(bl) >= len(cp), "Boundary-layer output has fewer nodes than CPWR")
    bl = bl[:len(cp)]
    require(np.allclose(bl[:,1],cp[:,0],atol=3e-5,rtol=0),
            "CPWR/DUMP surface nodes are not aligned; cannot safely remove wake")
    le_cp = int(np.argmin(cp[:,0]))
    le_bl = int(np.argmin(bl[:,1]))
    require(le_cp == le_bl and 3 <= le_cp <= len(cp)-4,
            "CPWR/DUMP leading edges disagree or surfaces are incomplete")
    require(cp[0,0] >= 0.98 and cp[-1,0] >= 0.98 and cp[le_cp,0] <= 0.02,
            "XFOIL output does not cover LE and both trailing edges")
    x = np.asarray(target_x, dtype=float)
    le_target = int(np.argmin(x))
    require(le_target*2+1 == len(x), "Expected symmetric lower/upper target grid")
    upper_x, lower_x = x[le_target:], x[:le_target+1]
    cp_lower = _interp(cp[le_cp:,0], cp[le_cp:,1], lower_x)
    cp_upper = _interp(cp[:le_cp+1,0], cp[:le_cp+1,1], upper_x)
    # XFOIL's positive Cf follows boundary-layer downstream direction. Signed
    # Ue is positive on the upper branch and negative on the lower in ordinary
    # attached flow. Multiplication also handles a shifted stagnation point;
    # negative Cf (wall backflow) is preserved, never replaced by abs(Cf).
    signed_cf = bl[:,3] * np.sign(bl[:,4])
    cf_lower = _interp(bl[le_bl:,1], signed_cf[le_bl:], lower_x)
    cf_upper = _interp(bl[:le_bl+1,1], signed_cf[:le_bl+1], upper_x)
    labels = np.stack((np.r_[cp_lower,cp_upper[1:]],
                       np.r_[cf_lower,cf_upper[1:]])).astype(np.float32)
    require(np.isfinite(labels).all(), "Nonfinite interpolated labels")
    return labels, {"cp_min":float(cp[:,1].min()),
                    "interpolation":"linear_separate_surfaces_constant_endpoint_extension",
                    "surface_node_counts":{"cp":len(cp),"boundary_layer":len(bl)}}


def read_polar(path, target_cl, cl_tolerance=5e-4):
    # Polar headers are un-commented text. Rows begin after the dashed divider.
    lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()
    rows = []
    for line in lines:
        try:
            row = [float(s.replace("D","E")) for s in line.split()]
        except ValueError:
            continue
        if len(row) >= 7:
            rows.append(row)
    require(len(rows) == 1, "Expected exactly one converged polar operating point")
    row = np.asarray(rows[0])
    require(np.isfinite(row).all(), "Nonfinite polar output")
    tolerance = max(cl_tolerance, abs(target_cl)*1e-4)
    require(abs(row[1]-target_cl) <= tolerance,
            f"Converged CL={row[1]} does not match target CL={target_cl}")
    require(row[2] >= 0, "Negative total drag in polar output")
    return {"alpha_deg":float(row[0]),"cl":float(row[1]),"cd":float(row[2]),
            "cd_pressure":float(row[3]),"cm":float(row[4]),
            "transition_upper_xc":float(row[5]),"transition_lower_xc":float(row[6])}


def xfoil_script(mach, cl, reynolds, panels, iterations, ncrit, trip_upper, trip_lower):
    return ("PLOP\nG\n\nLOAD foil.dat\nPPAR\nN " + str(panels) +
            "\n\n\nPANE\nOPER\nTYPE 1\nVISC " + format(reynolds,".12g") +
            "\nMACH " + format(mach,".12g") + "\nVPAR\nN " + format(ncrit,".8g") +
            "\nXTR " + format(trip_upper,".8g") + " " + format(trip_lower,".8g") +
            "\n\nITER " + str(iterations) + "\nPACC\npolar.dat\n\nCL " +
            format(cl,".12g") + "\nCPWR cp.dat\nDUMP bl.dat\nPACC\n\nQUIT\n")


def local_sonic_cp(mach):
    if not 0 < mach < 1:
        return None
    gamma = 1.4
    pressure_ratio = ((1+0.5*(gamma-1)*mach*mach)/(1+0.5*(gamma-1)))**(gamma/(gamma-1))
    return 2*(pressure_ratio-1)/(gamma*mach*mach)


def input_digest(x, y, conditions):
    sha = hashlib.sha256()
    for item in (x,y,conditions):
        sha.update(np.asarray(item,dtype="<f8").tobytes())
    return sha.hexdigest()


def compact_raw_outputs(root, case):
    """Losslessly archive raw files, verify bytes, then remove duplicate copies."""
    case = inside(root,case)
    archive = inside(root,case/"raw_outputs.zip")
    partial = inside(root,case/"raw_outputs.zip.partial")
    raw = [inside(root,case/name) for name in RAW_FILES if (case/name).is_file()]
    if not raw:
        require(archive.is_file(), "Compacted case has no raw archive")
        return {"archive":"raw_outputs.zip","files":None}
    require(not archive.exists(), "Refusing to replace an existing raw archive")
    hashes = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in raw}
    with zipfile.ZipFile(partial,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for path in raw:
            zipped.write(path,arcname=path.name)
    with zipfile.ZipFile(partial,"r") as zipped:
        require(zipped.testzip() is None, "Raw archive CRC verification failed")
        for name,digest in hashes.items():
            require(hashlib.sha256(zipped.read(name)).hexdigest() == digest,
                    "Raw archive differs from original output")
    os.replace(partial,archive)
    # The uncompressed copies are redundant after successful byte/CRC checks.
    # All names are fixed local case files, never recursive deletion targets.
    for path in raw:
        path.unlink()
    return {"archive":"raw_outputs.zip","sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "files":hashes}


def load_inputs(project_root, input_root):
    root, folder = Path(project_root).resolve(strict=True), inside(project_root,input_root)
    for name in ("manifest.json","COMPLETE.json","x.npy"):
        inside(root,folder/name)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((folder/"COMPLETE.json").read_text(encoding="utf-8"))
    require(manifest["status"] == complete["status"] == "complete", "Input build is incomplete")
    require(manifest["counts"] == complete["counts"], "Input completion counts disagree")
    x = np.load(folder/"x.npy",allow_pickle=False)
    require(x.ndim == 1 and np.isfinite(x).all(), "Invalid shared x grid")
    data = {}
    for split in ("train","test"):
        data[split] = {}
        n = manifest["counts"][split]
        for name in ("geometry","conditions","provenance"):
            path = inside(root,folder/split/(name+".npy"))
            data[split][name] = np.load(path,mmap_mode="r" if n else None,allow_pickle=False)
        require(data[split]["geometry"].shape == (n,1,len(x)), "Invalid geometry shape")
        require(data[split]["conditions"].shape == (n,3), "Invalid conditions shape")
        require(data[split]["provenance"].shape == (n,), "Invalid provenance shape")
    train_f = np.unique(data["train"]["provenance"]["family_id"])
    test_f = np.unique(data["test"]["provenance"]["family_id"])
    require(not np.intersect1d(train_f,test_f).size, "Train/test share a source CST family")
    return x,data,manifest


def choose_rows(conditions, count):
    """Deterministic rank samples covering Mach and CL, without relabelling splits."""
    n = len(conditions)
    if count is None or count >= n:
        return np.arange(n,dtype=np.int64)
    require(count >= 0, "Sample count cannot be negative")
    if count == 0:
        return np.array([],dtype=np.int64)
    chosen = []
    # Allocate quantiles of CL and Mach so a pilot is not just early source rows.
    orders = [np.argsort(conditions[:,i],kind="stable") for i in (1,0)]
    quota = [(count+1)//2,count//2]
    for order,k in zip(orders,quota):
        ranks = np.linspace(0,n-1,k,dtype=np.int64) if k else []
        for rank in ranks:
            value = int(order[rank])
            if value not in chosen:
                chosen.append(value)
            if len(chosen) == count:
                return np.sort(chosen)
    # Overlapping extrema in the two rankings must not reduce the pilot size.
    for value in np.linspace(0,n-1,n,dtype=np.int64):
        if int(value) not in chosen:
            chosen.append(int(value))
        if len(chosen) == count:
            break
    return np.asarray(sorted(chosen),dtype=np.int64)


def run_case(project_root, output_root, executable, x, y, conditions, provenance,
             split, row, config, retry_failed=False, compact_success=False):
    root = Path(project_root).resolve(strict=True)
    case = inside(root,output_root/"cases"/split/f"{row:07d}")
    case.mkdir(parents=True,exist_ok=True)
    # Check each path after mkdir as well, so a preexisting symlink cannot escape.
    for filename in ("result.json","labels.npz","foil.dat","commands.in","xfoil.log",
                     "polar.dat","cp.dat","bl.dat","raw_outputs.zip","raw_outputs.zip.partial"):
        inside(root,case/filename)
    digest = input_digest(x,y,conditions)
    old_path = case/"result.json"
    signature = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        require(old.get("input_sha256") == digest and old.get("config_sha256") == signature,
                f"Existing case {split}/{row} belongs to different input/settings")
        if old["status"] == "success":
            require((case/"labels.npz").is_file(), "Successful case lost its labels")
            if compact_success and "raw_outputs_archive" not in old:
                old["raw_outputs_archive"] = compact_raw_outputs(root,case)
                json_write(case/"result.json",old)
            return dict(old,resumed=True)
        if not retry_failed:
            return dict(old,resumed=True)
        # Archive, rather than overwrite, failed outputs for transparent retries.
        number = 1
        while (case/f"attempt_{number:03d}").exists():
            number += 1
        archive = inside(root,case/f"attempt_{number:03d}")
        archive.mkdir()
        for name in ("result.json",*RAW_FILES,"raw_outputs.zip"):
            source = case/name
            if source.exists():
                source.rename(archive/name)
    require(not (case/"labels.npz").exists(), "Refusing untracked existing labels")
    mach,cl,reynolds = map(float,conditions)
    record = {"schema":SCHEMA,"split":split,"input_row":int(row),"input_sha256":digest,
              "config_sha256":signature,"config":config,"conditions":{"mach":mach,"cl_target":cl,
              "reynolds_actual":reynolds},"solver":"XFOIL","is_rans":False,
              "provenance":{name:as_json(provenance[name]) for name in provenance.dtype.names},
              "started_utc":datetime.now(timezone.utc).isoformat(),"resumed":False}
    started = time.monotonic()
    try:
        require(np.isfinite(y).all() and np.isfinite(conditions).all(), "Nonfinite case inputs")
        require(0 <= mach < 1 and reynolds > 0, "XFOIL requires subsonic Mach and positive Re")
        # Standard XFOIL order: upper TE -> LE -> lower TE, the reverse of the
        # dataset's lower TE -> LE -> upper TE order. Neither shape nor Mach/CL
        # is altered to make a failed calculation converge.
        np.savetxt(case/"foil.dat",np.column_stack((x[::-1],y[::-1])),fmt="%.12g",
                   header=f"SuperWing_{split}_{row}",comments="")
        script = xfoil_script(mach,cl,reynolds,config["panels"],config["iterations"],
                              config["ncrit"],config["trip_upper"],config["trip_lower"])
        (case/"commands.in").write_text(script,encoding="ascii")
        home = inside(root,output_root/"runtime-home")
        temporary = inside(root,output_root/"tmp")
        cache = inside(root,output_root/"cache")
        for directory in (home,temporary,cache):
            directory.mkdir(parents=True,exist_ok=True)
        env = dict(os.environ,HOME=str(home),TMPDIR=str(temporary),TMP=str(temporary),
                   TEMP=str(temporary),XDG_CACHE_HOME=str(cache),HISTFILE=str(case/"history.disabled"),
                   OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
        env.pop("DISPLAY",None)
        # Plain executable invocation without shell avoids shell startup files
        # and shell history. XFOIL's relative defaults/output stay in this case.
        with (case/"xfoil.log").open("wb") as log:
            process = subprocess.run([str(executable)],input=script.encode("ascii"),
                                     cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     timeout=config["timeout_seconds"],check=False)
        require(process.returncode == 0,f"XFOIL exit code {process.returncode}")
        text = (case/"xfoil.log").read_text(encoding="ascii",errors="replace")
        require(not re.search(r"convergence\s+failed|floating.point|\bNaN\b|\bInfinity\b|segmentation\s+fault",
                              text,re.IGNORECASE), "XFOIL reported numerical/convergence failure")
        polar = read_polar(case/"polar.dat",cl,config["cl_tolerance"])
        labels,details = interpolate_labels(case/"cp.dat",case/"bl.dat",x)
        sonic = local_sonic_cp(mach)
        flags = ["xfoil_not_rans"]
        if sonic is not None and details["cp_min"] <= sonic:
            flags.append("local_sonic_or_supersonic_accuracy_not_assured")
        # The NPZ exists only for a converged, checked solution. Cf has no storage
        # scaling factor; consumers can explicitly multiply by 1000 if needed.
        np.savez_compressed(case/"labels.npz",cp=labels[0],cf=labels[1],x=x.astype(np.float64),
                            conditions=np.asarray(conditions,dtype=np.float64))
        record.update(status="success",polar=polar,quality_flags=flags,
                      sonic_cp_threshold=sonic,**details)
    except (ValueError,OSError,subprocess.TimeoutExpired) as error:
        record.update(status="failed",error={"type":type(error).__name__,"message":str(error)})
    record["elapsed_seconds"] = time.monotonic()-started
    if compact_success and record["status"] == "success":
        record["raw_outputs_archive"] = compact_raw_outputs(root,case)
    json_write(case/"result.json",record)
    return record


def bounded_results(pool, jobs, submit, pending_limit):
    """Keep only a small bounded number of queued argument arrays/futures."""
    require(pending_limit > 0,"Pending-future limit must be positive")
    iterator = iter(jobs)
    pending = set()
    exhausted = False
    while pending or not exhausted:
        while len(pending) < pending_limit and not exhausted:
            try:
                job = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending.add(submit(pool,job))
        if pending:
            finished,pending = wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                yield future.result()


def pack_labels(project_root, input_root, results_root, destination, seed=42):
    """Pack only checked successes, retaining the original family split.

    The smaller successful split determines a strict 9:1 selection. All raw
    solver results remain in results_root; unselected successful input rows are
    recorded explicitly, rather than deleted or moved across splits.
    """
    root = Path(project_root).resolve(strict=True)
    inputs = inside(root,input_root)
    results = inside(root,results_root)
    destination = inside(root,destination)
    require(destination != root and not destination.exists(),
            "Packing destination must be a fresh project subdirectory")
    for source in (inputs,results):
        require(destination != source and not destination.is_relative_to(source)
                and not source.is_relative_to(destination),"Packing paths must not overlap")
    x,data,input_manifest = load_inputs(root,inputs)
    config_path = inside(root,results/"solver_config.json")
    solver_config = json.loads(config_path.read_text(encoding="utf-8"))
    config_digest = hashlib.sha256(json.dumps(solver_config,sort_keys=True).encode()).hexdigest()
    available,states = {},{}
    begin = time.monotonic()
    for split in ("train","test"):
        available[split] = []
        states[split] = {"success":0,"failed":0,"not_attempted":0}
        for row in range(len(data[split]["conditions"])):
            path = inside(root,results/"cases"/split/f"{row:07d}"/"result.json")
            if not path.is_file():
                states[split]["not_attempted"] += 1
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            require(record["split"] == split and record["input_row"] == row,
                    "Result case identity differs from input row")
            require(record["config_sha256"] == config_digest,"Mixed solver settings in result tree")
            require(record["status"] in ("success","failed"),"Unknown solver result status")
            states[split][record["status"]] += 1
            if record["status"] == "success":
                available[split].append(row)
        print(json.dumps({"pack_scan_split":split,"states":states[split]},ensure_ascii=False),flush=True)
    n_test = min(len(available["test"]),len(available["train"])//9)
    require(n_test > 0,"At least 9 successful train cases and 1 successful test case are required")
    counts = {"train":9*n_test,"test":n_test}
    selection = {}
    for split,offset in (("train",0),("test",1)):
        rows = np.asarray(available[split],dtype=np.int64)
        chosen = np.random.default_rng(seed+offset).choice(rows,size=counts[split],replace=False)
        selection[split] = np.sort(chosen)
    train_families = np.unique(data["train"]["provenance"][selection["train"]]["family_id"])
    test_families = np.unique(data["test"]["provenance"][selection["test"]]["family_id"])
    require(not np.intersect1d(train_families,test_families).size,"Selected CST families leak")
    destination.mkdir(parents=True)
    json_write(destination/"PACK_STARTED.json",{"status":"packing","counts":counts})
    np.save(destination/"x.npy",x,allow_pickle=False)
    quality_dtype = np.dtype([("alpha_deg","<f8"),("cl_achieved","<f8"),("cd","<f8"),
                              ("cm","<f8"),("cp_min","<f8"),("local_sonic_risk","?")])
    index_columns = ["input_row","source_row","source_geometry_index","original_id",
                     "source_condition_id","family_id","section_index"]
    quality_counts = {}
    for split in ("train","test"):
        folder = inside(root,destination/split)
        folder.mkdir()
        n = counts[split]
        arrays = {
            "data":np.lib.format.open_memmap(folder/"data.npy",mode="w+",dtype="float32",shape=(n,3,len(x))),
            "conditions":np.lib.format.open_memmap(folder/"conditions.npy",mode="w+",dtype="float64",shape=(n,3)),
            "provenance":np.lib.format.open_memmap(folder/"provenance.npy",mode="w+",dtype=data[split]["provenance"].dtype,shape=(n,)),
            "index":np.lib.format.open_memmap(folder/"index.npy",mode="w+",dtype="int64",shape=(n,len(index_columns))),
            "quality":np.lib.format.open_memmap(folder/"quality.npy",mode="w+",dtype=quality_dtype,shape=(n,)),
        }
        sonic_count = 0
        for packed_row,input_row in enumerate(selection[split]):
            row = int(input_row)
            case = inside(root,results/"cases"/split/f"{row:07d}")
            record = json.loads(inside(root,case/"result.json").read_text(encoding="utf-8"))
            conditions = np.asarray(data[split]["conditions"][row])
            y = np.asarray(data[split]["geometry"][row,0])
            require(record["status"] == "success" and record["input_sha256"] == input_digest(x,y,conditions),
                    "Successful result is stale or belongs to changed inputs")
            with np.load(inside(root,case/"labels.npz"),allow_pickle=False) as labels:
                cp,cf = labels["cp"],labels["cf"]
                require(cp.shape == cf.shape == x.shape and np.isfinite(cp).all()
                        and np.isfinite(cf).all(),"Invalid successful Cp/Cf labels")
                require(np.array_equal(labels["x"],x) and np.array_equal(labels["conditions"],conditions),
                        "NPZ coordinates or conditions differ from input case")
                arrays["data"][packed_row] = np.stack((y,cp,1000.0*cf))
            arrays["conditions"][packed_row] = conditions
            provenance = data[split]["provenance"][row]
            arrays["provenance"][packed_row] = provenance
            arrays["index"][packed_row] = [row]+[int(provenance[name]) if name in provenance.dtype.names else -1
                                                       for name in index_columns[1:]]
            sonic = "local_sonic_or_supersonic_accuracy_not_assured" in record.get("quality_flags",[])
            sonic_count += int(sonic)
            polar = record["polar"]
            arrays["quality"][packed_row] = (polar["alpha_deg"],polar["cl"],polar["cd"],polar["cm"],
                                               record["cp_min"],sonic)
            if (packed_row+1) % 10000 == 0:
                print(json.dumps({"pack_split":split,"packed":packed_row+1,"total":n}),flush=True)
        for array in arrays.values():
            array.flush()
        quality_counts[split] = {"local_sonic_risk":sonic_count,"selected":n}
        np.save(folder/"selected_input_rows.npy",selection[split],allow_pickle=False)
        np.save(folder/"unselected_successful_input_rows.npy",
                np.setdiff1d(np.asarray(available[split],dtype=np.int64),selection[split]),allow_pickle=False)
        # Small sidecar directly documents the two common magnitude conventions.
        (folder/"format.txt").write_text(
            "data.npy [N,3,P]: effective_y/c, Cp, 1000*signed_Cf; float32.\n"
            "conditions.npy [N,3]: Mach, target_CL, actual_Re (not millions); float64.\n"
            "index.npy [N,7]: "+", ".join(index_columns)+".\n"
            "provenance.npy retains the complete input provenance. selected_input_rows.npy links every output row.\n"
            "quality.npy retains achieved polar coefficients and local-sonic applicability risk.\n"
            "These are genuine XFOIL solutions, not RANS or inverse-transformed 3-D surface labels.\n",
            encoding="utf-8")
    manifest = {"schema":SCHEMA,"status":"complete","labels_status":"computed_with_xfoil",
                "dataset_kind":"effective_superwing_2d_genuine_xfoil","is_rans":False,
                "created_utc":datetime.now(timezone.utc).isoformat(),"counts":counts,
                "train_test_ratio":[9,1],"exact_9_to_1":counts["train"]==9*counts["test"],
                "selection_seed":seed,"point_count":len(x),"source_input_root":str(inputs),
                "source_solver_results":str(results),"source_input_counts":input_manifest["counts"],
                "source_case_states":states,"solver_config":solver_config,
                "data_channels":["effective_y_over_c","Cp","1000*signed_Cf"],
                "conditions_columns":["mach_2d","cl_target_2d","reynolds_2d_actual"],
                "index_columns":index_columns,"selected_train_family_count":len(train_families),
                "selected_test_family_count":len(test_families),"family_overlap_count":0,
                "quality_counts":quality_counts,"unselected_successes_preserved":True,
                "raw_solver_results_preserved":True,"all_source_inputs_attempted":
                    all(state["not_attempted"]==0 for state in states.values()),
                "warning":"XFOIL is not the paper's RANS solver; local sonic/supersonic and shocked-flow accuracy is not assured.",
                "elapsed_seconds":time.monotonic()-begin}
    json_write(destination/"manifest.json",manifest)
    json_write(destination/"COMPLETE.json",{"status":"complete","labels_status":"computed_with_xfoil","counts":counts})
    return manifest


def run(args):
    if os.name == "posix":
        # Applied once in the coordinator, before worker threads are started.
        # Child numerical crashes must not create multi-megabyte core files for
        # hundreds of thousands of cases or invoke a core-file destination.
        import resource
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    root = args.project_root.resolve(strict=True)
    require(root.is_dir(), "Project root must be an existing directory")
    inputs = inside(root,args.input_root)
    output = inside(root,args.output_root)
    require(inputs != output and not inputs.is_relative_to(output) and not output.is_relative_to(inputs),
            "Input and result directories must not overlap")
    require(args.xfoil is not None, "--xfoil is required for solver execution")
    executable = args.xfoil.expanduser().resolve(strict=True)
    require(executable.is_file(), "XFOIL executable is missing")
    require(args.workers > 0 and args.panels >= 80 and args.panels <= 300 and args.iterations > 0,
            "Invalid worker/panel/iteration setting")
    require(args.timeout > 0 and args.ncrit > 0 and 0 <= args.trip_upper <= 1 and 0 <= args.trip_lower <= 1,
            "Invalid timeout/transition setting")
    x,data,manifest = load_inputs(root,inputs)
    output.mkdir(parents=True,exist_ok=True)
    config = {"panels":args.panels,"iterations":args.iterations,"ncrit":args.ncrit,
              "trip_upper":args.trip_upper,"trip_lower":args.trip_lower,
              "timeout_seconds":args.timeout,"cl_tolerance":args.cl_tolerance,
              "xfoil_executable":str(executable),"schema":SCHEMA}
    settings = inside(root,output/"solver_config.json")
    json_once(settings,config)
    json_once(inside(root,output/"README.json"),{
        "dataset_kind":"genuine_xfoil_viscous_solutions","is_rans":False,
        "input_root":str(inputs),"source_input_counts":manifest["counts"],
        "labels":{"cp":"freestream_dynamic_pressure_normalization",
                  "cf":"raw_unscaled_freestream_dynamic_pressure_normalization_signed_in_dataset_contour_direction"},
        "geometry_order":"lower_TE_to_LE_then_upper_to_TE",
        "case_success_requires":"one_converged_polar_row_matching_target_CL_and_finite_Cp_Cf_on_both_surfaces",
        "warning":"XFOIL uses an integral boundary layer and potential-flow compressibility correction, not RANS. Sonic/shocked flows lack accuracy assurance.",
        "results_policy":"Only success cases have labels.npz; failed cases retain raw outputs and failure reason. No dummy or inverse-transformed 3-D labels.",
        "path_policy":"All task inputs, outputs, cwd, HOME, temporary files and caches are under project_root. Installed executable/OS libraries are external runtime resources."})
    jobs = []
    splits = ("train","test") if args.split == "both" else (args.split,)
    for split in splits:
        count = args.pilot if args.pilot is not None else args.limit
        chosen = choose_rows(data[split]["conditions"],count)
        require(args.start >= 0 and (args.stop is None or args.stop >= args.start),
                "Invalid input-row start/stop")
        chosen = chosen[chosen >= args.start]
        if args.stop is not None:
            chosen = chosen[chosen < args.stop]
        if args.shard_count is not None:
            require(args.shard_count > 0 and 0 <= args.shard_index < args.shard_count,
                    "Invalid shard index/count")
            chosen = chosen[chosen % args.shard_count == args.shard_index]
        for row in chosen:
            jobs.append((split,int(row)))
    summary = {"schema":SCHEMA,"input_root":str(inputs),"output_root":str(output),
               "selected_case_count":len(jobs),"input_counts":manifest["counts"],
               "selection":"pilot" if args.pilot is not None else "limited" if args.limit is not None else "all",
               "success":0,"failed":0,"resumed":0,"status":"running",
               "input_row_start":args.start,"input_row_stop":args.stop,
               "shard_count":args.shard_count,"shard_index":args.shard_index,
               "max_pending_futures":2*args.workers}
    require(jobs,"No selected cases")
    # A shard writes its own coordinator log/summary; case paths remain keyed by
    # original input row, so disjoint shards can be packed together afterward.
    run_tag = args.run_name
    if run_tag is None and args.shard_count is not None:
        run_tag = f"{args.split}-shard-{args.shard_index:04d}-of-{args.shard_count:04d}"
    if run_tag is not None:
        require(re.fullmatch(r"[A-Za-z0-9_.-]+",run_tag) is not None,"Unsafe --run-name")
        run_folder = inside(root,output/"runs")
        run_folder.mkdir(exist_ok=True)
        summary_path = inside(root,run_folder/(run_tag+".json"))
        status_path = inside(root,run_folder/(run_tag+".jsonl"))
    else:
        summary_path = inside(root,output/"run_summary.json")
        status_path = inside(root,output/"status.jsonl")
    json_write(summary_path,summary)
    begin = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def submit(pool,job):
            split,row = job
            values = data[split]
            return pool.submit(run_case,root,output,executable,x,
                               np.asarray(values["geometry"][row,0]),
                               np.asarray(values["conditions"][row]),values["provenance"][row],
                               split,row,config,args.retry_failed,args.compact_success)
        with status_path.open("a",encoding="utf-8") as log:
            results = bounded_results(pool,jobs,submit,2*args.workers)
            for done,result in enumerate(results,1):
                summary[result["status"]] += 1
                summary["resumed"] += int(result["resumed"])
                event = {key:result[key] for key in ("split","input_row","status","elapsed_seconds","resumed")}
                if "error" in result:
                    event["error"] = result["error"]
                log.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+"\n")
                log.flush()
                summary["elapsed_seconds"] = time.monotonic()-begin
                if done % 10 == 0 or done == len(jobs):
                    json_write(summary_path,summary)
                    print(json.dumps(dict(summary,completed=done),ensure_ascii=False),flush=True)
    summary["status"] = "selected_cases_finished"
    summary["all_requested_inputs_attempted"] = len(jobs) == sum(manifest["counts"].values())
    summary["labels_complete_for_selected"] = summary["failed"] == 0
    summary["final_dataset_train_test_ratio"] = "not_packed; use successful cases only"
    json_write(summary_path,summary)
    if args.pack_to is not None:
        summary["packed_dataset"] = pack_labels(root,inputs,output,args.pack_to,args.pack_seed)
        json_write(summary_path,summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,required=True)
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output-root",type=Path,default=Path("build/superwing_xfoil"))
    p.add_argument("--xfoil",type=Path,help="Explicit installed executable; required unless --pack-only")
    p.add_argument("--split",choices=("both","train","test"),default="both")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--pilot",type=int,help="Representative cases per selected split")
    group.add_argument("--limit",type=int,help="Maximum representative cases per selected split")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--panels",type=int,default=240)
    p.add_argument("--iterations",type=int,default=200)
    p.add_argument("--timeout",type=float,default=30.0)
    p.add_argument("--ncrit",type=float,default=9.0)
    p.add_argument("--trip-upper",type=float,default=1.0)
    p.add_argument("--trip-lower",type=float,default=1.0)
    p.add_argument("--cl-tolerance",type=float,default=5e-4)
    p.add_argument("--retry-failed",action="store_true")
    p.add_argument("--compact-success",action="store_true",help="Archive and byte-verify raw success outputs before removing redundant uncompressed copies")
    p.add_argument("--start",type=int,default=0,help="Inclusive original input row, applied to each selected split")
    p.add_argument("--stop",type=int,help="Exclusive original input row, applied to each selected split")
    p.add_argument("--shard-count",type=int,help="Run only input rows whose index modulo count equals shard-index")
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--run-name",help="Safe unique coordinator filename for concurrent disjoint row ranges")
    p.add_argument("--pack-to",type=Path,help="Fresh project subdirectory for successful 9:1 packed data")
    p.add_argument("--pack-only",action="store_true",help="Pack existing results without running XFOIL")
    p.add_argument("--pack-seed",type=int,default=42)
    return p


if __name__ == "__main__":
    try:
        args = parser().parse_args()
        if args.pack_only:
            require(args.pack_to is not None,"--pack-only requires --pack-to")
            result = pack_labels(args.project_root,args.input_root,args.output_root,args.pack_to,args.pack_seed)
        else:
            result = run(args)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        raise SystemExit(0 if result.get("failed",0) == 0 else 1)
    except (ValueError,OSError,KeyError) as error:
        print(f"error: {error}",file=sys.stderr,flush=True)
        raise SystemExit(2)
