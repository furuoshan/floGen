"""Pack checked compressible OpenFOAM SST solutions without split leakage.

Only the project's RANS runner success records and their genuine labels can be
included. Failed, running, unattempted and XFOIL cases never become labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


SOLVER = "OpenFOAM-v2412-rhoSimpleFoam+SST"
INDEX_COLUMNS = ("input_row", "source_row", "source_geometry_index", "original_id",
                 "source_condition_id", "family_id", "section_index")


def require(value, message):
    if not value:
        raise ValueError(message)


def inside(root, path):
    root = Path(root).resolve(strict=True)
    path = Path(path)
    if not path.is_absolute():
        path = root/path
    path = path.resolve()
    require(path == root or path.is_relative_to(root),f"Path escapes project root: {path}")
    return path


def json_write(path, value):
    temporary = path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def _finite_json(value):
    if isinstance(value,float):
        return math.isfinite(value)
    if isinstance(value,dict):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value,(list,tuple)):
        return all(_finite_json(item) for item in value)
    return True


def _load_inputs(root, folder):
    manifest = json.loads(inside(root,folder/"manifest.json").read_text(encoding="utf-8"))
    complete = json.loads(inside(root,folder/"COMPLETE.json").read_text(encoding="utf-8"))
    require(manifest.get("status") == complete.get("status") == "complete","Input build is incomplete")
    require(manifest["counts"] == complete["counts"],"Input manifest/completion counts disagree")
    x = np.load(inside(root,folder/"x.npy"),allow_pickle=False)
    require(x.ndim == 1 and len(x)%2 == 1 and np.isfinite(x).all(),"Invalid input x grid")
    data = {}
    for split in ("train","test"):
        n = manifest["counts"][split]
        require(isinstance(n,int) and n >= 0,"Invalid input counts")
        data[split] = {name:np.load(inside(root,folder/split/(name+".npy")),
                                   mmap_mode="r" if n else None,allow_pickle=False)
                       for name in ("geometry","conditions","provenance")}
        require(data[split]["geometry"].shape == (n,1,len(x)),"Invalid input geometry shape")
        require(data[split]["conditions"].shape == (n,3),"Invalid input conditions shape")
        require(data[split]["provenance"].shape == (n,),"Invalid input provenance shape")
        require(data[split]["provenance"].dtype.names is not None
                and all(name in data[split]["provenance"].dtype.names for name in INDEX_COLUMNS[1:]),
                "Input provenance lacks required source identity fields")
    a = np.unique(data["train"]["provenance"]["family_id"])
    b = np.unique(data["test"]["provenance"]["family_id"])
    require(not np.intersect1d(a,b).size,"Input train/test share a baseline-CST family")
    return x,data,manifest


def _solver_identity(config):
    solver = config.get("solver","rhoSimpleFoam")
    if solver == "rhoSimpleFoam":
        require(config.get("time_stepping","steadyState") == "steadyState",
                "rhoSimpleFoam configuration must use steadyState time treatment")
        return "OpenFOAM-v2412-rhoSimpleFoam",SOLVER,"steadyState"
    require(solver == "rhoPimpleFoam","Unsupported compressible RANS solver in config")
    require(config.get("time_stepping","localEuler") == "localEuler",
            "Only verified steady-state localEuler rhoPimpleFoam labels are supported")
    name = "OpenFOAM-v2412-rhoPimpleFoam-LTS"
    return name,name+"+SST","localEuler"


def _check_rans_record(record, split, row, expected_solver="OpenFOAM-v2412-rhoSimpleFoam"):
    require(record.get("split") == split and record.get("input_row") == row,"Result case identity disagrees")
    require(record.get("solver") in (expected_solver,expected_solver+"+SST")
            and record.get("turbulence_model") == "kOmegaSST",
            "Non-OpenFOAM-v2412 compressible SST record: refusing mixed/XFOIL labels")
    require(record.get("status") in ("success","failed","running"),"Unknown RANS case status")


def _fingerprint(x,y,conditions,provenance,config):
    return hashlib.sha256(x.tobytes()+y.tobytes()+conditions.tobytes()+provenance.tobytes()
                          +json.dumps(config,sort_keys=True).encode()).hexdigest()


def _checked_success(root, case, record, split, row, x, inputs, config):
    """Check current source identity, native acceptance and every NPZ operand."""
    _check_rans_record(record,split,row,_solver_identity(config)[0])
    require(record["status"] == "success","Only successful RANS cases may be packed")
    y = np.array(inputs["geometry"][row,0],dtype=float)
    conditions = np.array(inputs["conditions"][row],dtype=float)
    provenance = inputs["provenance"][row]
    require(np.isfinite(y).all() and np.isfinite(conditions).all(),"Nonfinite source input")
    require(record.get("fingerprint") == _fingerprint(x,y,conditions,provenance,config),
            "Success record is stale or refers to changed inputs/configuration")
    require(np.array_equal(np.asarray(record.get("conditions")),conditions),"Result conditions differ from source")
    expected_provenance = {name:provenance[name].item() for name in provenance.dtype.names}
    require(record.get("provenance") == expected_provenance,"Result provenance differs from complete source metadata")
    path = inside(root,case/"labels.npz")
    with np.load(path,allow_pickle=False) as raw:
        require(set(("geometry","labels","conditions","provenance")).issubset(raw.files),
                "RANS NPZ lacks required source/label fields")
        geometry,labels,oc,p = raw["geometry"],raw["labels"],raw["conditions"],raw["provenance"]
        require(geometry.shape == y.shape and np.array_equal(geometry,y),"NPZ geometry differs from source input")
        require(oc.shape == (3,) and np.array_equal(oc,conditions),"NPZ conditions differ from source input")
        require(p.shape in ((),(1,)) and np.array_equal(p.reshape(()),np.asarray(provenance)),
                "NPZ provenance differs from complete source metadata")
        require(labels.shape == (2,len(x)) and np.isfinite(labels).all(),"Invalid Cp/Cf label shape or values")
        require(np.isfinite(geometry).all() and np.isfinite(oc).all(),"Nonfinite NPZ inputs")
        labels = labels.astype(np.float32,copy=True)
    quality = json.loads(inside(root,case/"quality.json").read_text(encoding="utf-8"))
    require(quality.get("converged") is True and _finite_json(quality),"RANS quality is incomplete or nonfinite")
    achieved = float(record.get("cl",math.nan))
    require(math.isfinite(achieved) and abs(achieved-conditions[1]) <= config["cl_tolerance"],
            "Successful case did not achieve its prescribed CL")
    require(math.isfinite(float(record.get("cd",math.nan))),"Invalid achieved drag coefficient")
    require(abs(float(quality.get("cl",math.nan))-achieved) <= 1e-10,"Quality/record CL disagree")
    return y,labels,conditions,provenance,quality


def pack_dataset(project_root, input_root, result_root, dest, seed=42):
    root = Path(project_root).resolve(strict=True)
    inputs,results,destination = (inside(root,path) for path in (input_root,result_root,dest))
    require(destination != root and not destination.exists(),"Destination must be a fresh project subdirectory")
    for source in (inputs,results):
        require(destination != source and not destination.is_relative_to(source)
                and not source.is_relative_to(destination),"Packing paths must not overlap")
    x,data,input_manifest = _load_inputs(root,inputs)
    config = json.loads(inside(root,results/"config.json").read_text(encoding="utf-8"))
    require("cl_tolerance" in config and float(config["cl_tolerance"]) > 0,"Missing target CL acceptance threshold")
    expected_solver,solver_identity,time_stepping = _solver_identity(config)
    counts_in = input_manifest["counts"]
    states,successes = {},{}
    started = time.monotonic()
    for split in ("train","test"):
        states[split] = {"success":0,"failed":0,"running":0,"not_attempted":0}
        successes[split] = []
        for row in range(counts_in[split]):
            path = inside(root,results/"cases"/split/f"{row:07d}"/"result.json")
            if not path.is_file():
                states[split]["not_attempted"] += 1
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            _check_rans_record(record,split,row,expected_solver)
            states[split][record["status"]] += 1
            if record["status"] == "success":
                successes[split].append(row)
        print(json.dumps({"pack_scan_split":split,"source_states":states[split]}),flush=True)
    n_test = min(len(successes["test"]),len(successes["train"])//9)
    require(n_test > 0,"Strict 9:1 packing requires >=9 successful train and >=1 successful test cases")
    counts = {"train":9*n_test,"test":n_test}
    selection = {split:np.sort(np.random.default_rng(seed+offset).choice(
                    np.asarray(successes[split],dtype=np.int64),size=counts[split],replace=False))
                 for split,offset in (("train",0),("test",1))}
    selected_families = {split:np.unique(data[split]["provenance"][selection[split]]["family_id"])
                         for split in ("train","test")}
    require(not np.intersect1d(selected_families["train"],selected_families["test"]).size,"Packed family leakage")
    destination.mkdir(parents=True)
    json_write(destination/"PACK_STARTED.json",{"status":"packing","counts":counts,"solver":solver_identity})
    np.save(destination/"x.npy",x,allow_pickle=False)
    max_cl_error = 0.
    for split in ("train","test"):
        folder = inside(root,destination/split)
        folder.mkdir()
        n = counts[split]
        outputs = {
            "data":np.lib.format.open_memmap(folder/"data.npy",mode="w+",dtype="float32",shape=(n,3,len(x))),
            "conditions":np.lib.format.open_memmap(folder/"conditions.npy",mode="w+",dtype="float64",shape=(n,3)),
            "index":np.lib.format.open_memmap(folder/"index.npy",mode="w+",dtype="int64",shape=(n,len(INDEX_COLUMNS))),
            "provenance":np.lib.format.open_memmap(folder/"provenance.npy",mode="w+",dtype=data[split]["provenance"].dtype,shape=(n,)),
        }
        with (folder/"quality.jsonl").open("w",encoding="utf-8") as quality_file:
            for packed_row,input_row in enumerate(selection[split]):
                row = int(input_row)
                case = inside(root,results/"cases"/split/f"{row:07d}")
                record = json.loads(inside(root,case/"result.json").read_text(encoding="utf-8"))
                y,labels,conditions,provenance,quality = _checked_success(root,case,record,split,row,x,data[split],config)
                outputs["data"][packed_row] = np.vstack((y,labels[0],1000.0*labels[1]))
                outputs["conditions"][packed_row] = conditions
                outputs["provenance"][packed_row] = provenance
                outputs["index"][packed_row] = [row]+[int(provenance[name]) for name in INDEX_COLUMNS[1:]]
                max_cl_error = max(max_cl_error,abs(float(record["cl"])-float(conditions[1])))
                quality_record = {"packed_row":packed_row,"input_row":row,"split":split,"solver":solver_identity,
                                  "source_case":str(case.relative_to(results)),"quality":quality}
                quality_file.write(json.dumps(quality_record,ensure_ascii=False,allow_nan=False)+"\n")
                if (packed_row+1)%10000 == 0:
                    print(json.dumps({"pack_split":split,"packed":packed_row+1,"total":n}),flush=True)
        for output in outputs.values():
            output.flush()
        np.save(folder/"selected_input_rows.npy",selection[split],allow_pickle=False)
        np.save(folder/"unselected_successful_input_rows.npy",
                np.setdiff1d(np.asarray(successes[split],dtype=np.int64),selection[split]),allow_pickle=False)
    all_attempted = all(state["not_attempted"] == state["running"] == 0 for state in states.values())
    all_succeeded = all(state["success"] == counts_in[split] for split,state in states.items())
    manifest = {
        "schema":"superwing-effective-2d-rans-v1","status":"complete","labels_status":"complete",
        "solver":solver_identity,"solver_record_id":expected_solver,
        "time_stepping":time_stepping,"is_pseudo_time":time_stepping=="localEuler",
        "turbulence_model":"kOmegaSST","is_rans":True,
        "created_utc":datetime.now(timezone.utc).isoformat(),"counts":counts,"point_count":len(x),
        "train_test_ratio":[9,1],"exact_9_to_1":counts["train"]==9*counts["test"],
        "selection_seed":seed,"source_input_root":str(inputs),"source_result_root":str(results),
        "source_input_counts":counts_in,"source_case_states":states,
        "all_source_inputs_attempted":all_attempted,"all_source_inputs_succeeded":all_succeeded,
        "coverage_status":"all_inputs_attempted" if all_attempted else "partial_checkpoint",
        "checkpoint_only":not all_attempted,
        "completed_attempt_count":sum(state["success"]+state["failed"] for state in states.values()),
        "started_attempt_count":sum(state["success"]+state["failed"]+state["running"] for state in states.values()),
        "failed_source_case_count":sum(state["failed"] for state in states.values()),
        "data_channels":["effective_y_over_c","Cp","1000*signed_Cf"],
        "conditions_columns":["mach_2d","cl_target_2d","reynolds_2d_actual"],
        "index_columns":list(INDEX_COLUMNS),"geometry_order":"lower_TE_to_LE_then_upper_to_TE",
        "family_overlap_count":0,"selected_family_counts":{s:len(f) for s,f in selected_families.items()},
        "solver_config":config,"max_selected_target_cl_error":max_cl_error,
        "raw_solver_cases_preserved":True,"unselected_successful_input_rows_preserved":True,
        "source_metadata_restored_from":"complete_input_provenance.npy",
        "elapsed_seconds":time.monotonic()-started,
        "note":"Pack completion applies only to the selected converged labels. Source coverage, failures, running and unattempted counts are reported separately. No XFOIL, failed, dummy or inverse-transformed 3-D labels are included."
    }
    json_write(destination/"manifest.json",manifest)
    json_write(destination/"COMPLETE.json",{"status":"complete","labels_status":"complete","counts":counts,
                                             "coverage_status":manifest["coverage_status"]})
    (destination/"README.txt").write_text(
        "Effective SuperWing 2-D compressible RANS labels\n\n"
        "Solver: "+solver_identity+", compressible ideal-gas Sutherland transport, kOmegaSST.\n"
        "Time treatment: "+time_stepping+"; localEuler is steady pseudo-time, not an accepted physical unsteady label series.\n"
        "train|test/data.npy [N,3,P] float32: effective_y/c, Cp, 1000*signed_Cf.\n"
        "conditions.npy [N,3] float64: Mach, prescribed CL, actual Re (not millions).\n"
        "index.npy columns: "+", ".join(INDEX_COLUMNS)+".\n"
        "provenance.npy restores every source input metadata field; quality.jsonl retains acceptance metrics.\n"
        "selected_input_rows.npy and unselected_successful_input_rows.npy preserve selection traceability.\n"
        "Only successful native CFD solutions are included. The selected train/test count ratio is exactly 9:1.\n"
        "Read manifest.json coverage_status and source_case_states: a complete pack can be a PARTIAL CHECKPOINT of the source calculation.\n"
        "Failed and unattempted cases are never filled or reclassified as successful labels. All source cases/logs are retained.\n",
        encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--result-root",type=Path,required=True)
    parser.add_argument("--dest",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()
    print(json.dumps(pack_dataset(args.project_root,args.input_root,args.result_root,args.dest,args.seed),
                     indent=2,ensure_ascii=False,allow_nan=False),flush=True)
