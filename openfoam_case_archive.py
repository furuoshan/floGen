"""Verified, resumable compaction of completed project-owned OpenFOAM cases.

Wall solution arrays, labels, request identity, native diagnostics and exact
configuration text survive. Reconstructible mesh and internal-volume fields do
not. No deletion occurs before a CRC/SHA-verified archive and transaction record
exist. Symlinks anywhere in a case are rejected instead of followed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid
import zipfile

import numpy as np


FORMAT = "openfoam_completed_case_archive_v1"
NUMBER = re.compile(r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
KEEP = {"result.json", "quality.json", "labels.npz", "request.npz", "raw_surface.npz"}
METADATA = {"mesh_metadata.json", "mesh_acceptance.json", "case_conditions.json"}
CODE_NAMES = ("openfoam_airfoil_mesh.py", "openfoam_airfoil_case.py", "openfoam_airfoil_post.py",
              "openfoam_case_archive.py", "run_superwing_openfoam.py")
RAW_KEYS = ("face_centres", "tangent", "p_pa", "native_wall_shear", "tau_body_pa", "cp", "cf")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_jsonable)+"\n").encode("utf-8")


def _secure(project, case, path):
    """Validate components before following any link, including dangling links."""
    path = Path(os.path.abspath(path))
    _require(path == case or path.is_relative_to(case), "Case path escapes case directory")
    _require(path.is_relative_to(project), "Case path escapes authorized project")
    current = project
    for part in path.relative_to(project).parts:
        current = current/part
        _require(not current.is_symlink(), f"Refusing symlink in case path: {current.name}")
    _require(path.resolve().is_relative_to(case), "Resolved path escapes case directory")
    return path


def _scan(project, case, root):
    root = _secure(project,case,root)
    files, directories = {}, []
    if not root.exists():
        return files,directories
    _require(root.is_dir(),f"Expected owned case directory: {root.name}")
    for parent,dirs,names in os.walk(root,followlinks=False):
        folder=_secure(project,case,Path(parent))
        directories.append(folder.relative_to(case).as_posix())
        for name in dirs:
            _secure(project,case,folder/name)
        for name in names:
            path=_secure(project,case,folder/name)
            info=path.stat(follow_symlinks=False)
            _require(stat.S_ISREG(info.st_mode),"Nonregular case file is not archivable")
            files[path.relative_to(case).as_posix()]={"bytes":info.st_size,"mtime_ns":info.st_mtime_ns}
    return files,directories


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(project,case,path):
    """Persist publication metadata before any potentially irreversible cleanup."""
    if os.name=="posix":
        path=_secure(project,case,path)
        descriptor=os.open(path,os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_json(project,case,path,value):
    path=_secure(project,case,path)
    temporary=_secure(project,case,path.with_name(path.name+".partial-"+uuid.uuid4().hex))
    with temporary.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    _secure(project,case,path)
    os.replace(temporary,path)
    _sync_directory(project,case,path.parent)


def _request(case,result):
    path=case/"request.npz"
    _require(path.is_file(),"request.npz must be saved before case compaction")
    with np.load(path,allow_pickle=False) as archive:
        _require({"x","geometry","conditions","provenance"}.issubset(archive.files),
                 "request.npz lacks geometry/conditions/provenance reconstruction fields")
        request={name:archive[name] for name in archive.files}
    for name,value in request.items():
        _require(value.dtype.kind != "O",f"Object array is not permitted in request: {name}")
    _require(np.isfinite(request["x"]).all() and np.isfinite(request["geometry"]).all()
             and np.isfinite(request["conditions"]).all(),"Nonfinite reconstruction request")
    _require(request["x"].size == request["geometry"].size,"Request x/geometry shape mismatch")
    if "fingerprint" in request:
        _require(str(request["fingerprint"].item())==result["fingerprint"],"Request/result fingerprint mismatch")
    return request


def _save_raw(project,case,solution):
    path=_secure(project,case,case/"raw_surface.npz")
    if path.exists():
        with np.load(path,allow_pickle=False) as existing:
            values={name:existing[name] for name in existing.files}
        if solution is not None:
            raw=solution.get("raw_wall",{})
            for name in RAW_KEYS:
                _require(name in raw and name in values and np.array_equal(raw[name],values[name]),
                         "Saved raw wall arrays differ from the native solution")
    else:
        _require(solution is not None and solution.get("converged") is True,
                 "Successful compaction requires the genuine converged raw-wall solution")
        raw=solution.get("raw_wall",{})
        _require(all(name in raw for name in RAW_KEYS),"Native raw-wall fields are incomplete")
        values={name:np.asarray(raw[name]) for name in RAW_KEYS}
    n=len(values["p_pa"])
    _require(n>0,"Empty raw surface")
    for name in RAW_KEYS:
        shape=(n,3) if name in ("face_centres","tangent","native_wall_shear","tau_body_pa") else (n,)
        _require(values[name].shape==shape and values[name].dtype.kind in "fiu" and np.isfinite(values[name]).all(),
                 f"Invalid native wall array: {name}")
    _require((values["p_pa"]>0).all(),"Nonpositive native absolute wall pressure")
    if not path.exists():
        temporary=_secure(project,case,case/("raw_surface.partial-"+uuid.uuid4().hex+".npz"))
        with temporary.open("xb") as stream:
            np.savez_compressed(stream,**values)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temporary,allow_pickle=False) as checked:
            for name,value in values.items():
                _require(np.array_equal(checked[name],value),"Raw surface read-back differs")
        os.replace(temporary,path)
        _sync_directory(project,case,path.parent)
    return {"file":"raw_surface.npz","sha256":_sha(path),"wall_faces":n,
            "arrays":{name:{"shape":list(values[name].shape),"dtype":str(values[name].dtype)} for name in RAW_KEYS},
            "source":"genuine_parse_solution_raw_wall_no_numeric_resampling_or_placeholder_values"}


def _verify_archive(path):
    with zipfile.ZipFile(path,"r") as archive:
        _require(archive.testzip() is None,"Diagnostics ZIP CRC verification failed")
        names=archive.namelist()
        _require(len(names)==len(set(names)),"Duplicate ZIP entry")
        manifest=json.loads(archive.read("MANIFEST.json"))
        _require(manifest.get("format")==FORMAT,"Unexpected diagnostics archive format")
        _require(set(names)==set(manifest["files"])|{"MANIFEST.json"},"ZIP member/manifest mismatch")
        for name,expected in manifest["files"].items():
            _require(not name.startswith("/") and ".." not in Path(name).parts,"Unsafe ZIP member name")
            value=archive.read(name)
            _require(len(value)==expected["bytes"] and hashlib.sha256(value).hexdigest()==expected["sha256"],
                     f"Diagnostics SHA256 verification failed: {name}")
    return manifest


def _cleanup(project,case,record):
    for relative,snapshot in record["cleanup_directories"].items():
        target=_secure(project,case,case/relative)
        current,dirs=_scan(project,case,target)
        _require(set(current).issubset(snapshot["files"]),"New case files appeared during compaction")
        _require(set(dirs).issubset(snapshot["directories"]),"New case directories appeared during compaction")
        for name,info in current.items():
            _require(info==snapshot["files"][name],"Owned case file changed before cleanup")
        for name in sorted(current,key=lambda item:(item.count("/"),item),reverse=True):
            path=_secure(project,case,case/name)
            actual=path.stat(follow_symlinks=False)
            _require({"bytes":actual.st_size,"mtime_ns":actual.st_mtime_ns}==current[name],"Case file changed during cleanup")
            path.unlink()
        for name in sorted(dirs,key=lambda item:(item.count("/"),item),reverse=True):
            _secure(project,case,case/name).rmdir()
    for relative,expected in record["cleanup_logs"].items():
        path=_secure(project,case,case/relative)
        if path.exists():
            _require(path.is_file() and _sha(path)==expected,"Archived native log changed before cleanup")
            path.unlink()


def compact_case(project_root,case_dir,solution=None):
    """Compact a terminal, request-bound case; failure or verification error never deletes its fields."""
    project=Path(project_root).resolve(strict=True)
    case=Path(os.path.abspath(case_dir))
    _require(case!=project and case.is_relative_to(project),"Case must be inside the authorized project")
    case=_secure(project,case,case)
    _require(case.is_dir(),"Case directory is missing")
    inventory,_=_scan(project,case,case)  # Reject every symlink before any write.
    result_path=_secure(project,case,case/"result.json")
    result=json.loads(result_path.read_text(encoding="utf-8"))
    _require(result.get("status") in ("success","failed"),"Only terminal cases may be compacted")
    _require(result.get("solver")=="OpenFOAM-v2412-rhoSimpleFoam" and result.get("turbulence_model")=="kOmegaSST",
             "Case is not this task's compressible OpenFOAM/SST result")
    _require(re.fullmatch(r"[0-9a-f]{64}",str(result.get("fingerprint",""))) is not None,"Case fingerprint must be saved before compaction")
    _request(case,result)
    record_path=_secure(project,case,case/"archive_record.json")
    zip_path=_secure(project,case,case/"raw_diagnostics.zip")
    if record_path.exists():
        record=json.loads(record_path.read_text(encoding="utf-8"))
        _require(record.get("format")==FORMAT and record.get("fingerprint")==result["fingerprint"],"Archive record identity differs")
        _require(record.get("phase") in ("archive_verified","complete"),"Unrecognized compaction phase")
        if not zip_path.exists() and record.get("archive_pending_file"):
            pending_name=record["archive_pending_file"]
            _require(Path(pending_name).name==pending_name and pending_name.startswith("raw_diagnostics.zip.partial-"),
                     "Invalid pending diagnostics archive path")
            pending=_secure(project,case,case/pending_name)
            _require(pending.is_file() and _sha(pending)==record["archive_sha256"],"Pending diagnostics archive changed")
            _verify_archive(pending)
            os.replace(pending,zip_path)
            _sync_directory(project,case,case)
        _require(zip_path.is_file() and _sha(zip_path)==record["archive_sha256"],"Diagnostics archive changed")
        manifest=_verify_archive(zip_path)
        for name,sha in record["retained_files_sha256"].items():
            path=_secure(project,case,case/name)
            _require(path.is_file() and _sha(path)==sha,"A retained identity/label/wall file changed")
        if record["phase"]=="complete":
            return dict(record,already_compacted=True)
    else:
        _require(not zip_path.exists(),"Unowned existing diagnostics ZIP; original fields were retained")
        success=result["status"]=="success"
        if success:
            _require((case/"quality.json").is_file() and (case/"labels.npz").is_file(),"Successful quality/labels must be saved before compaction")
            raw=_save_raw(project,case,solution)
        else:
            _require(not (case/"labels.npz").exists(),"Failed case unexpectedly contains training labels")
            raw=None
        inventory,_=_scan(project,case,case)
        selected={}
        for name in inventory:
            parts=Path(name).parts
            top=parts[0]
            is_log=(len(parts)==1 and (name.endswith(".log") or name.startswith("log.")))
            if (name in KEEP|METADATA or is_log or top in ("system","postProcessing",".runtime")
                    or (top=="constant" and (len(parts)<2 or parts[1]!="polyMesh"))):
                selected[name]=_secure(project,case,case/name)
        code={}
        for name in CODE_NAMES:
            path=Path(__file__).resolve().parent/name
            if path.is_relative_to(project):
                path=_secure(project,project,path)
                if path.is_file():
                    code["reproduction_code/"+name]=path
        _require("reproduction_code/openfoam_airfoil_mesh.py" in code,
                 "Mesh generator code must be available inside the authorized project for reproducible compaction")
        virtual={}
        if solution is not None:
            histories={name:solution[name] for name in ("force_history","residual_history") if name in solution}
            if "residuals" in solution:
                histories["residuals"]=solution["residuals"]
            if histories:
                virtual["derived/native_solution_histories.json"]=_json_bytes(histories)
        cleanup_dirs={}
        for path in case.iterdir():
            name=path.name
            numeric=bool(NUMBER.fullmatch(name)) and math.isfinite(float(name))
            if path.is_dir() and (numeric or name in ("postProcessing",".runtime")):
                files,dirs=_scan(project,case,path)
                cleanup_dirs[name]={"files":files,"directories":dirs}
        poly=_secure(project,case,case/"constant/polyMesh")
        if poly.exists():
            files,dirs=_scan(project,case,poly)
            cleanup_dirs["constant/polyMesh"]={"files":files,"directories":dirs}
        cleanup_logs={name:_sha(path) for name,path in selected.items()
                      if len(Path(name).parts)==1 and (name.endswith(".log") or name.startswith("log."))
                      and name.startswith(("solver","checkMesh","potential","update","log."))}
        policy={"retained":"native wall pressure/shear/traction and Cp/Cf, successful labels, request, final result/quality, all native logs/history and exact case dictionaries",
                "omitted":"internal volumetric fields in numeric time directories and reconstructible constant/polyMesh",
                "reconstruction":"request.npz geometry/x/conditions/provenance plus exact archived reproduction code and case dictionaries; native potential/solver command logs remain",
                "failed_cases":"actual request, terminal error and native diagnostics; no invented wall fields or training labels",
                "temporary_runtime":"archived before removal; only this case's .runtime is eligible"}
        manifest={"format":FORMAT,"fingerprint":result["fingerprint"],"status":result["status"],
                  "files":{},"retention_policy":policy}
        partial=_secure(project,case,case/("raw_diagnostics.zip.partial-"+uuid.uuid4().hex))
        with zipfile.ZipFile(partial,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
            for name,path in sorted({**selected,**code}.items()):
                digest=hashlib.sha256()
                size=0
                with path.open("rb") as incoming,archive.open(name,"w",force_zip64=True) as outgoing:
                    for data in iter(lambda:incoming.read(1024*1024),b""):
                        outgoing.write(data)
                        digest.update(data)
                        size+=len(data)
                manifest["files"][name]={"bytes":size,"sha256":digest.hexdigest(),
                                         "source":"case_file" if name in selected else "reproduction_code"}
            for name,data in virtual.items():
                archive.writestr(name,data)
                manifest["files"][name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"source":"native_solution_object"}
            archive.writestr("MANIFEST.json",_json_bytes(manifest))
        with partial.open("r+b") as stream:
            os.fsync(stream.fileno())
        _verify_archive(partial)
        # Every selected case file must still be exactly what was archived.
        for name,path in selected.items():
            _require(_sha(path)==manifest["files"][name]["sha256"],"Diagnostic changed while archiving; original fields retained")
        _require(not zip_path.exists(),"Diagnostics destination appeared while archiving")
        retained={name:_sha(case/name) for name in KEEP if (case/name).is_file()}
        record={"format":FORMAT,"phase":"archive_verified","fingerprint":result["fingerprint"],
                "case_status":result["status"],"created_utc":datetime.now(timezone.utc).isoformat(),
                "archive_file":"raw_diagnostics.zip","archive_sha256":_sha(partial),
                "archive_pending_file":partial.name,
                "archive_bytes":partial.stat().st_size,"archive_file_count":len(manifest["files"]),
                "crc_verified":True,"per_file_sha256_verified":True,"retained_files_sha256":retained,
                "raw_surface":raw,"retention_policy":policy,"cleanup_directories":cleanup_dirs,
                "cleanup_logs":cleanup_logs,"bytes_before":sum(info["bytes"] for info in inventory.values()),
                "reproduction_code_sha256":{name.split("/")[-1]:manifest["files"][name]["sha256"] for name in code}}
        _atomic_json(project,case,record_path,record)
        # The verified transaction record precedes publication. A crash between
        # these operations can resume from its named, SHA-bound pending ZIP.
        os.replace(partial,zip_path)
        _sync_directory(project,case,case)
    _cleanup(project,case,record)
    remaining,_=_scan(project,case,case)
    record.update(phase="complete",completed_utc=datetime.now(timezone.utc).isoformat(),
                  bytes_after=sum(info["bytes"] for info in remaining.values()),already_compacted=False,
                  archive_pending_file=None)
    record["bytes_reclaimed"]=record["bytes_before"]-record["bytes_after"]
    _atomic_json(project,case,record_path,record)
    return record
