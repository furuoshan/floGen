"""SuperWing geometry prediction and incremental training worker (wing-loop-v1).

Run: python -B bridge.py --job absolute/path/to/job.json
Keeps checkpoint normalization and physical priors fixed. Source data is read-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

# A staged bridge can be tested without copying or changing the source repository.
_ENTRY = Path(__file__).resolve().with_name("main.py")
if not _ENTRY.is_file():
    _ENTRY = Path("D:/floGen/main.py")
if str(_ENTRY.parent) not in sys.path:
    sys.path.insert(0, str(_ENTRY.parent))
_MODULE = "flogen_closed_loop_entry"
if _MODULE in sys.modules:
    m = sys.modules[_MODULE]
else:
    _spec = importlib.util.spec_from_file_location(_MODULE, _ENTRY)
    m = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE] = m
    _spec.loader.exec_module(m)

SCHEMA_VERSION = "wing-loop-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute_path(value, label):
    require(isinstance(value, str) and bool(value), f"{label} must be an absolute path string")
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute: {value}")
    return path.resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def publish_new(temporary, destination):
    # Same-directory hard-link publication is atomic and never replaces a prior file.
    os.link(temporary, destination)
    Path(temporary).unlink()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("xb") as output:
        output.write(json_bytes(value))
        output.flush()
        os.fsync(output.fileno())
    try:
        publish_new(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def checked_rows(value, count, label):
    require(isinstance(value, list) and len(value) > 0, f"{label}.rows must be a nonempty list")
    require(all(type(row) is int and 0 <= row < count for row in value),
            f"{label}.rows must contain zero-based integer source rows")
    require(len(set(value)) == len(value), f"{label}.rows must not contain duplicates")
    return np.asarray(value, dtype=np.int64)


def reynolds_options(selection, label):
    scalar = selection.get("reynolds")
    filename = selection.get("reynolds_file")
    require(not (scalar is not None and filename is not None), f"{label}: use Reynolds scalar OR file")
    if scalar is not None:
        require(type(scalar) in (int, float) and math.isfinite(scalar) and scalar > 0,
                f"{label}.reynolds must be positive and finite")
    if filename is not None:
        filename = absolute_path(filename, f"{label}.reynolds_file")
    return {"reynolds":scalar, "reynolds_file":filename}


def selection_data(selection, spec, label):
    require(isinstance(selection, dict), f"{label} must be an object")
    root = absolute_path(selection.get("root"), f"{label}.root")
    options = reynolds_options(selection, label)
    dataset = m.SuperWingData(root, target="field", require_reynolds=spec.condition_dim == 3, **options)
    rows = checked_rows(selection.get("rows"), len(dataset), label)
    return dataset, rows


def digest_array(digest, name, values):
    values = np.asarray(values)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(values.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(values.shape)).encode("ascii") + b"\0")
    digest.update(np.ascontiguousarray(values).tobytes())


def validation_fingerprint(dataset, rows):
    digest = hashlib.sha256(b"wing-loop-v1:fixed-validation\0")
    digest_array(digest, "selected_rows", rows)
    for row in rows:
        row = int(row)
        g = int(dataset.shape_ids[row])
        digest_array(digest, "index", dataset.index[row])
        digest_array(digest, "config", dataset.configs[g])
        digest_array(digest, "geom0", dataset.centers[g])
        digest_array(digest, "origingeom", dataset.vertices[g])
        digest_array(digest, "target", dataset.data[row])
        if dataset.require_reynolds:
            digest_array(digest, "reynolds", np.asarray(dataset.reynolds_at(row), dtype=np.float64))
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest_array(digest, name, value.detach().cpu().numpy())
    return digest.hexdigest()


def normalizer_digest(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, m.FeatureNorm):
            state.update({name + "." + key:value for key, value in module.state_dict().items()})
    return state_digest(state)


def prior_digest(model):
    state = {"airfoil." + name:value for name, value in model.airfoil.state_dict().items()}
    state.update({"sld." + name:value for name, value in model.lift.state_dict().items()})
    return state_digest(state)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key:cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return deepcopy(value)


@torch.inference_mode()
def evaluate(model, dataset, rows, device, batch_size):
    model.eval()
    normalized_sum = 0.0
    raw_sum = np.zeros(3, dtype=np.float64)
    channel_count = count = 0
    loader = DataLoader(Subset(dataset, rows.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        batch = m.move_batch(batch, device)
        prediction = model(batch)
        difference = m.normalize_target(model, prediction, "wing") - m.normalize_target(model, batch["target"], "wing")
        require(bool(torch.isfinite(difference).all()), "Non-finite fixed validation prediction")
        normalized_sum += float(difference.double().square().sum())
        count += difference.numel()
        raw = (prediction.double() - batch["target"].double()).square().sum(dim=(0,2,3))
        raw_sum += raw.cpu().numpy()
        channel_count += prediction.shape[0] * prediction.shape[2] * prediction.shape[3]
    require(count > 0, "Empty fixed validation")
    return {"normalized_mse":normalized_sum / count,
            "raw_rmse":np.sqrt(raw_sum / channel_count).tolist()}


def check_no_validation_leakage(payload, train_sets, validation, validation_rows):
    heldout = set(map(int, validation.groups[validation_rows]))
    current = set()
    for dataset, rows in train_sets:
        current.update(map(int, dataset.groups[rows]))
    training = payload["training"]
    require("train_shape_ids" in training, "Input checkpoint lacks historical wing training shape IDs")
    require("sld_train_shape_ids" in training, "Input checkpoint lacks historical SLD training shape IDs")
    historical = set(map(int, training["train_shape_ids"]))
    sld = set(map(int, training["sld_train_shape_ids"]))
    for label, shapes in (("current training",current),("historical wing training",historical),("SLD training",sld)):
        overlap = heldout & shapes
        require(not overlap, f"Fixed validation shapes overlap {label}: {sorted(overlap)[:10]}")
    canonical = set()
    for dataset, rows in train_sets:
        for row in rows:
            key = (int(dataset.groups[row]), int(dataset.condition_ids[row]))
            require(key not in canonical, f"Duplicate training canonical case (original_id, condition): {key}; repeated weighting and conflicting source copies are not allowed")
            canonical.add(key)
    return sorted(historical | current), sorted(heldout)



def validate_prediction_job(job):
    require(job.get("schema_version") == SCHEMA_VERSION, "Unsupported job schema_version")
    require(type(job.get("round_id")) is int and job["round_id"] >= 0, "Invalid round_id")
    require(isinstance(job.get("job_id"),str) and job["job_id"], "Missing job_id")
    device = job.get("device","cpu")
    index = device.removeprefix("cuda:") if isinstance(device,str) else ""
    require(device in ("cpu","cuda") or (isinstance(device,str) and device.startswith("cuda:") and
            index.isascii() and index.isdigit()), "device must be cpu, cuda, or cuda:<nonnegative integer>")
    require(type(job.get("seed",42)) is int and 0 <= job.get("seed",42) < 2**32,
            "seed must be an integer in 0..2**32-1")
    require(type(job.get("batch_size",1)) is int and job.get("batch_size",1) > 0,
            "batch_size must be a positive integer")
    expected = job.get("checkpoint_in_sha256")
    require(expected is None or (isinstance(expected,str) and len(expected) == 64 and
            all(char in "0123456789abcdef" for char in expected)), "Invalid checkpoint_in_sha256")
    require(isinstance(job.get("geometry"),dict), "geometry must be an object")
    require(isinstance(job.get("prediction"),dict), "prediction must be an object")
    paths = {key:absolute_path(job.get(key),key) for key in ("checkpoint_in","result_path")}
    paths["geometry_root"] = absolute_path(job["geometry"].get("root"),"geometry.root")
    paths["prediction_output"] = absolute_path(job["prediction"].get("output_root"),"prediction.output_root")
    sources = [paths["checkpoint_in"],paths["geometry_root"]]
    destinations = [paths["result_path"],paths["result_path"].with_suffix(".started.json"),paths["prediction_output"]]
    require(len(set(destinations)) == len(destinations), "Output artifact paths must be distinct")
    for destination in destinations:
        m.protect_destination(destination,sources)
        require(not any(destination in source.parents for source in sources), "Output cannot be an ancestor of a source")
    for destination in destinations[:2]:
        require(paths["prediction_output"] not in destination.parents, "Result/claim cannot be inside prediction output")
    output = paths["prediction_output"]
    require(m.prediction_directory_available(output),
            f"Prediction output must be empty or contain only a regular format.md file: {output}")
    return paths


def geometry_manifest(root, expected_fingerprint=None):
    root = Path(root)
    path = root/"geometry_manifest.json"
    content = path.read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    require(expected_fingerprint is None or expected_fingerprint == fingerprint,
            "Geometry manifest fingerprint differs from expected_fingerprint")
    manifest = json.loads(content.decode("utf-8-sig"))
    require(isinstance(manifest,dict) and manifest.get("schema_version") == "wing-geometry-v1",
            "Unsupported geometry manifest schema")
    files = manifest.get("files")
    require(isinstance(files,dict), "Geometry manifest files must be an object")
    mandatory = {"configs.dat","index.npy","geom0.npy","origingeom.npy"}
    require(mandatory <= set(files), "Geometry manifest lacks required metadata/mesh files")
    re_file = manifest.get("reynolds_file")
    if re_file is not None:
        require(isinstance(re_file,str) and Path(re_file).name == re_file and
                re_file not in mandatory | {".","..","data.npy","geometry_manifest.json","running_info.jsonl"},
                "Manifest reynolds_file must be a local filename")
    allowed = mandatory | {"running_info.jsonl"} | ({re_file} if re_file is not None else set())
    require(set(files) <= allowed, "Geometry manifest contains unsupported files (flow data.npy is never read)")
    for filename,digest in files.items():
        require(isinstance(digest,str) and len(digest) == 64 and
                all(char in "0123456789abcdef" for char in digest), f"Invalid geometry SHA256: {filename}")
        require(sha256_file(root/filename) == digest, f"Geometry file SHA256 mismatch: {filename}")
    require(not (manifest.get("reynolds") is not None and re_file is not None),
            "Manifest must use Reynolds scalar OR file")
    if re_file is not None:
        require(re_file in files, "Manifest Reynolds file lacks a declared SHA256")
    require(manifest.get("reynolds") is not None or re_file is not None or "running_info.jsonl" in files,
            "Geometry manifest must declare actual Reynolds scalar, file, or original logs")
    return manifest,fingerprint


class GeometryData(torch.utils.data.Dataset):
    """Pure geometry/conditions; deliberately never opens any data.npy."""
    def __init__(self,root,manifest):
        self.root = Path(root).resolve()
        raw_index = m.read_npy(self.root/"index.npy")
        require(raw_index.ndim == 2 and raw_index.shape[1] in (6,12) and len(raw_index) > 0,
                "Geometry index must be nonempty (N,6) or compatible (N,12)")
        self.index = raw_index[:,:6]
        self.configs = np.loadtxt(self.root/"configs.dat",skiprows=1,ndmin=2)
        self.centers = m.read_npy(self.root/"geom0.npy")
        self.vertices = m.read_npy(self.root/"origingeom.npy")
        require(self.configs.ndim == 2 and self.configs.shape[1] == 55,"Geometry configs.dat must have 55 columns")
        require(self.centers.shape == (len(self.configs),3,128,256),"Geometry geom0 shape mismatch")
        require(self.vertices.shape == (len(self.configs),3,129,257),"Geometry origingeom shape mismatch")
        m.finite(self.index,"geometry index");m.finite(self.configs,"geometry configs")
        self.shape_ids = m.integer_values(self.index[:,0],len(self.configs)-1,"geometry shape index")
        self.condition_ids = m.integer_values(self.index[:,1],7,"geometry condition index")
        originals = m.integer_values(self.configs[:,0],2**53-1,"geometry original IDs")
        require(len(np.unique(originals)) == len(originals),"Duplicate geometry original IDs")
        self.groups = originals[self.shape_ids]
        require(len(set(zip(self.groups.tolist(),self.condition_ids.tolist()))) == len(self.index),
                "Duplicate geometry shape/condition samples")
        require(bool((self.index[:,4:6] > 0).all()),"Geometry reference area/half-span must be positive")
        require(np.allclose(self.index[:,2],self.configs[self.shape_ids,47+self.condition_ids],rtol=0,atol=1e-8),
                "Geometry AoA does not match configs")
        require(np.allclose(self.index[:,3],self.configs[self.shape_ids,39+self.condition_ids],rtol=0,atol=1e-8),
                "Geometry Mach does not match configs")
        scalar = manifest.get("reynolds")
        filename = manifest.get("reynolds_file")
        if scalar is not None:
            require(type(scalar) in (int,float) and math.isfinite(scalar) and scalar > 0,
                    "Manifest Reynolds scalar must be positive and finite")
            self.reynolds = np.full(len(self.index),scalar,dtype=np.float64)
        elif filename is not None:
            values = m.read_npy(self.root/filename)
            require(values.shape == (len(self.index),),"Manifest Reynolds file must be (N,) in geometry row order")
            self.reynolds = np.asarray(values,dtype=np.float64)
        else:
            records = [json.loads(line) for line in (self.root/"running_info.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            logs = {int(record["running_id"]):record for record in records}
            require(len(logs) == len(records),"Duplicate original IDs in geometry Reynolds logs")
            values = []
            for original,k in zip(self.groups,self.condition_ids):
                try:
                    values.append(float(logs[int(original)][f"{int(k)}_running_info"]["working_conditions"]["re"]))
                except KeyError as error:
                    raise ValueError(f"Geometry Reynolds missing for original_id={original}, condition={k}") from error
            self.reynolds = np.asarray(values,dtype=np.float64)
        m.finite(self.reynolds,"geometry actual Reynolds")
        require(bool((self.reynolds > 0).all()),"Geometry Reynolds must be positive")
        self._features = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self,row):
        g = int(self.shape_ids[row])
        if g not in self._features:
            self._features[g] = m.geometry_features(self.centers[g],self.vertices[g])
            if len(self._features) > 16:
                del self._features[next(iter(self._features))]
        result = {key:(torch.from_numpy(value.copy()) if value.dtype == bool else m.tensor(value))
                  for key,value in self._features[g].items()}
        result.update(parameters=m.tensor(self.configs[g,1:39]),conditions=m.tensor(self.index[row,[3,2]]),
                      reynolds=m.tensor(self.reynolds[row]),row=int(row))
        return result


@torch.inference_mode()
def execute_predict_job(job,paths,request_hash):
    root = paths["geometry_root"]
    manifest,fingerprint = geometry_manifest(root,job["geometry"].get("expected_fingerprint"))
    ancestor_sha = sha256_file(paths["checkpoint_in"])
    expected = job.get("checkpoint_in_sha256")
    require(expected is None or expected == ancestor_sha,"Input checkpoint SHA256 differs from checkpoint_in_sha256")
    model,spec,payload = m.load_checkpoint(paths["checkpoint_in"],"wing")
    require(payload["training"].get("airfoil_training_steps",0) > 0 and payload["training"].get("sld_training_steps",0) > 0,
            "Pure prediction requires a trained wing checkpoint with trained 2-D/SLD priors")
    dataset = GeometryData(root,manifest)
    # Optional caller Re parameters may confirm the manifest, never override it.
    options = reynolds_options(job["geometry"],"geometry")
    if options["reynolds"] is not None:
        require(bool((dataset.reynolds == options["reynolds"]).all()),"Job Reynolds differs from shared geometry manifest")
    if options["reynolds_file"] is not None:
        values = m.read_npy(options["reynolds_file"])
        require(values.shape == dataset.reynolds.shape and np.array_equal(values,dataset.reynolds),
                "Job Reynolds file differs from shared geometry manifest")
    m.seed_everything(job.get("seed",42))
    device = m.selected_device(job.get("device","cpu"))
    model.to(device).eval()
    normalization_before = normalizer_digest(model)
    model_before = state_digest(model.state_dict())
    output_root = paths["prediction_output"]
    output_root.mkdir(parents=True,exist_ok=True)
    marker = output_root/"prediction_info.json"
    metadata = {"status":"in_progress","source_root":str(root),"geometry_fingerprint":fingerprint,
                "geometry_manifest_schema":"wing-geometry-v1","sample_count":len(dataset),
                "channels":list(m.CHANNELS),"stored_multipliers":[1,150,300],"condition_dim":spec.condition_dim,
                "index_layout":"geometry_conditions_v1","index_coefficients_role":"unavailable_not_measured_not_predicted",
                "operation":"predict","checkpoint_sha256":ancestor_sha,"training_steps_performed":0,
                "source_geometry_manifest":"source_geometry_manifest.json"}
    marker.write_bytes(json_bytes(metadata))
    partial = output_root/"data.npy.partial"
    output = np.lib.format.open_memmap(partial,mode="w+",dtype=np.float32,shape=(len(dataset),3,128,256))
    offset = 0
    loader = DataLoader(dataset,batch_size=job.get("batch_size",1),shuffle=False,num_workers=0)
    for batch in loader:
        prediction = model(m.move_batch(batch,device))
        require(bool(torch.isfinite(prediction).all()),"Non-finite independent neural prediction")
        values = prediction.cpu().numpy()
        output[offset:offset+len(values)] = values
        offset += len(values)
    require(offset == len(dataset),"Independent prediction row count mismatch")
    output.flush();del output
    require(normalizer_digest(model) == normalization_before and state_digest(model.state_dict()) == model_before,
            "Model state changed during pure prediction")
    geometry_manifest(root,fingerprint)
    require(sha256_file(paths["checkpoint_in"]) == ancestor_sha,"Input checkpoint changed during pure prediction")
    for filename in manifest["files"]:
        if filename != "index.npy":
            m.shutil.copy2(root/filename,output_root/filename)
    # The geometry's first six columns are true metadata; no CFD coefficient is invented.
    np.save(output_root/"index.npy",np.array(dataset.index,copy=True),allow_pickle=False)
    m.shutil.copy2(root/"geometry_manifest.json",output_root/"source_geometry_manifest.json")
    partial.replace(output_root/"data.npy")
    metadata["status"] = "complete"
    marker.write_bytes(json_bytes(metadata))
    return {"schema_version":SCHEMA_VERSION,"job_id":job["job_id"],"round_id":job["round_id"],
            "operation":"predict","request_sha256":request_hash,"status":"completed",
            "checkpoint_in":str(paths["checkpoint_in"]),"checkpoint_sha256":ancestor_sha,
            "prediction_dir":str(output_root),"geometry_fingerprint":fingerprint,
            "steps_performed":0,"normalization_unchanged":True}


def validate_job(job):
    require(isinstance(job, dict), "Job JSON must be an object")
    if job.get("operation") == "predict":
        return validate_prediction_job(job)
    require(job.get("schema_version") == SCHEMA_VERSION, "Unsupported job schema_version")
    require(job.get("operation") == "train_predict", "Unsupported job operation")
    require(type(job.get("round_id")) is int and job["round_id"] >= 0, "Invalid round_id")
    require(isinstance(job.get("job_id"), str) and job["job_id"], "Missing job_id")
    require(job.get("training_data_policy") in ("cumulative","current"), "Invalid training_data_policy")
    device = job.get("device", "cpu")
    cuda_index = device.removeprefix("cuda:") if isinstance(device,str) else ""
    require(device in ("cpu","cuda") or
            (isinstance(device,str) and device.startswith("cuda:") and cuda_index.isascii() and cuda_index.isdigit()),
            "device must be cpu, cuda, or cuda:<nonnegative integer>")
    require(type(job.get("seed",42)) is int and 0 <= job.get("seed",42) < 2**32,
            "seed must be an integer in 0..2**32-1")
    checkpoint_hash = job.get("checkpoint_in_sha256")
    require(checkpoint_hash is None or (isinstance(checkpoint_hash,str) and len(checkpoint_hash) == 64 and
            all(char in "0123456789abcdef" for char in checkpoint_hash)),
            "checkpoint_in_sha256 must be a lowercase SHA256 hexadecimal string")
    plan = job.get("plan")
    require(isinstance(plan, dict), "Missing training plan")
    require(type(plan.get("learning_rate")) in (int,float) and
            math.isfinite(plan["learning_rate"]) and plan["learning_rate"] > 0, "Invalid learning_rate")
    for name in ("steps","batch_size"):
        require(type(plan.get(name)) is int and plan[name] > 0, f"plan.{name} must be positive integer")
    require(isinstance(job.get("train"), list) and len(job["train"]) > 0, "train must be nonempty selections")
    require(isinstance(job.get("validation"), dict), "validation must be a selection object")
    require(isinstance(job.get("prediction"), dict), "prediction must be an object")
    paths = {name:absolute_path(job.get(name), name) for name in ("checkpoint_in","checkpoint_out","result_path")}
    prediction = job["prediction"]
    paths["prediction_input"] = absolute_path(prediction.get("input_root"), "prediction.input_root")
    paths["prediction_output"] = absolute_path(prediction.get("output_root"), "prediction.output_root")
    sources = [paths["checkpoint_in"], paths["prediction_input"]]
    sources += [absolute_path(item.get("root"), "train.root") for item in job["train"]]
    sources += [absolute_path(job["validation"].get("root"), "validation.root")]
    for label, selection in [("prediction",prediction),("validation",job["validation"])] + [("train",s) for s in job["train"]]:
        options = reynolds_options(selection,label)
        if options["reynolds_file"] is not None:
            sources.append(options["reynolds_file"])
    destinations = [paths["checkpoint_out"], paths["result_path"], paths["result_path"].with_suffix(".started.json"), paths["prediction_output"]]
    require(len(set(destinations)) == len(destinations), "Output artifact paths must be distinct")
    for destination in destinations:
        m.protect_destination(destination, sources)
        require(not any(destination in source.parents for source in sources), "Output cannot be an ancestor of a source")
    for artifact in destinations[:3]:
        require(paths["prediction_output"] not in artifact.parents, "Checkpoint/result/claim cannot be inside prediction output")
    require(not paths["checkpoint_out"].exists(), f"Checkpoint output already exists: {paths['checkpoint_out']}")
    output = paths["prediction_output"]
    require(m.prediction_directory_available(output),
            f"Prediction output must be empty or contain only a regular format.md file: {output}")
    return paths


def train_increment(model, payload, train_sets, validation, val_rows, plan, device, seed):
    m.seed_everything(seed)
    model.to(device)
    model.freeze_priors()
    normalization_before = normalizer_digest(model)
    priors_before = prior_digest(model)
    before = evaluate(model, validation, val_rows, device, plan["batch_size"])
    parameters = list(model.residual.parameters())
    optimizer = torch.optim.Adam(parameters, lr=plan["learning_rate"])
    previous = payload["training"].get("bridge_optimizer")
    if previous is not None:
        optimizer.load_state_dict(previous)
    for group in optimizer.param_groups:
        group["lr"] = plan["learning_rate"]
    samples = ConcatDataset([Subset(dataset, rows.tolist()) for dataset, rows in train_sets])
    loader = DataLoader(samples, batch_size=plan["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    batches = iter(loader)
    total = 0.0
    for step in range(plan["steps"]):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        model.train()
        batch = m.move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = torch.nn.functional.mse_loss(m.normalize_target(model,prediction,"wing"),
                                            m.normalize_target(model,batch["target"],"wing"))
        require(bool(torch.isfinite(loss)), "Non-finite incremental training loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        print(json.dumps({"bridge_step":step+1,"steps":plan["steps"],"train_mse":float(loss.detach())}),flush=True)
    after = evaluate(model, validation, val_rows, device, plan["batch_size"])
    require(normalization_before == normalizer_digest(model), "Checkpoint normalization changed during incremental training")
    require(priors_before == prior_digest(model), "Frozen physical prior networks changed")
    return before, after, cpu_tree(optimizer.state_dict()), previous is None, total / plan["steps"]


def execute_job(job, paths, request_hash):
    ancestor_sha = sha256_file(paths["checkpoint_in"])
    expected_ancestor = job.get("checkpoint_in_sha256")
    require(expected_ancestor is None or ancestor_sha == expected_ancestor, "Input checkpoint SHA256 differs from checkpoint_in_sha256")
    model, spec, payload = m.load_checkpoint(paths["checkpoint_in"], "wing")
    require(payload["training"].get("airfoil_training_steps",0) > 0 and
            payload["training"].get("sld_training_steps",0) > 0, "Input wing checkpoint lacks trained prior provenance")
    train_sets = [selection_data(item,spec,f"train[{i}]") for i,item in enumerate(job["train"])]
    validation, val_rows = selection_data(job["validation"],spec,"validation")
    cumulative_shapes, validation_shapes = check_no_validation_leakage(payload,train_sets,validation,val_rows)
    fingerprint = validation_fingerprint(validation,val_rows)
    expected = job["validation"].get("expected_fingerprint")
    require(expected is None or expected == fingerprint, "Fixed validation fingerprint differs from job expected_fingerprint")
    prior_fingerprint = payload["training"].get("validation_fingerprint")
    require(prior_fingerprint is None or prior_fingerprint == fingerprint, "Fixed validation fingerprint differs from input checkpoint")
    device = m.selected_device(job.get("device","cpu"))
    before,after,optimizer_state,reset,mean_loss = train_increment(
        model,payload,train_sets,validation,val_rows,job["plan"],device,job.get("seed",42))
    require(validation_fingerprint(validation,val_rows) == fingerprint, "Fixed validation changed during worker execution")
    require(sha256_file(paths["checkpoint_in"]) == ancestor_sha, "Input checkpoint changed during worker execution")
    steps = job["plan"]["steps"]
    training = deepcopy(payload["training"])
    previous_total = int(training.get("cumulative_training_steps",training["training_steps"]))
    training.update(training_steps=previous_total+steps,cumulative_training_steps=previous_total+steps,
                    round_training_steps=steps,train_shape_ids=cumulative_shapes,
                    validation_shape_ids=validation_shapes,validation_fingerprint=fingerprint,
                    bridge_optimizer=optimizer_state,bridge_optimizer_reset=reset,
                    checkpoint_in=str(paths["checkpoint_in"]),checkpoint_in_sha256=ancestor_sha,
                    bridge_schema=SCHEMA_VERSION,job_id=job["job_id"],round_id=job["round_id"],
                    request_sha256=request_hash,training_data_policy=job["training_data_policy"],
                    plan=deepcopy(job["plan"]),fixed_validation_before=before,fixed_validation_after=after,
                    train_selections=deepcopy(job["train"]),validation_selection=deepcopy(job["validation"]),
                    normalization_preserved=True,prior_networks_preserved=True,
                    checkpoint_selection="final_optimizer_step",last_validation_mse=after["normalized_mse"])
    checkpoint_out = paths["checkpoint_out"]
    checkpoint_out.parent.mkdir(parents=True,exist_ok=True)
    temporary = checkpoint_out.with_name(checkpoint_out.name+".partial-"+uuid.uuid4().hex)
    roots = [dataset.root for dataset,_ in train_sets] + [validation.root,paths["prediction_input"]]
    m.save_checkpoint(temporary,"wing",model,spec,training,roots)
    # Reuse the original exporter: no CFD target is read for prediction, all source
    # metadata is copied unchanged, and the original channel multipliers remain.
    prediction_options = reynolds_options(job["prediction"],"prediction")
    export_info = {"stage":"wing","training":training}
    m.predict_directory(model,spec,export_info,paths["prediction_input"],paths["prediction_output"],
                        device=device,batch_size=job["plan"]["batch_size"],**prediction_options)
    publish_new(temporary,checkpoint_out)
    metrics = {"learning_rate":float(job["plan"]["learning_rate"]),"steps":float(steps),
               "train_normalized_mse":float(mean_loss),"optimizer_reset":float(reset)}
    for i,name in enumerate(m.CHANNELS):
        metrics["before_"+name+"_rmse"] = float(before["raw_rmse"][i])
        metrics["after_"+name+"_rmse"] = float(after["raw_rmse"][i])
    feedback = {"round_id":job["round_id"],"validation_error_before":float(before["normalized_mse"]),
                "validation_error_after":float(after["normalized_mse"]),"metrics":metrics,
                "notes":"Fixed validation; original checkpoint min-max ranges and frozen 2-D/SLD priors retained. Final-step model and matching Adam state saved."}
    return {"schema_version":SCHEMA_VERSION,"job_id":job["job_id"],"round_id":job["round_id"],
            "request_sha256":request_hash,"status":"completed","checkpoint_out":str(checkpoint_out),
            "checkpoint_sha256":sha256_file(checkpoint_out),"prediction_dir":str(paths["prediction_output"]),
            "validation_fingerprint":fingerprint,"steps_performed":steps,"normalization_unchanged":True,
            "feedback":feedback}


def run_job(job_path):
    job_path = Path(job_path).resolve()
    content = job_path.read_bytes()
    request_hash = hashlib.sha256(content).hexdigest()
    job = json.loads(content.decode("utf-8-sig"))
    require(isinstance(job,dict), "Job JSON must be an object")
    result_path = absolute_path(job.get("result_path"),"result_path")
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("request_sha256") == request_hash:
            return existing
        raise ValueError("Result already exists for failed or different request; cannot rerun")
    paths = validate_job(job)
    for destination in [path for path in (paths.get("checkpoint_out"),paths["prediction_output"],result_path) if path is not None]:
        require(destination != job_path and destination not in job_path.parents, "Output must not overwrite or contain job input")
    claim = result_path.with_suffix(".started.json")
    claim.parent.mkdir(parents=True,exist_ok=True)
    with claim.open("x",encoding="utf-8") as output:
        json.dump({"schema_version":SCHEMA_VERSION,"job_id":job["job_id"],"round_id":job["round_id"],
                   "request_sha256":request_hash,"status":"started"},output,indent=2)
    try:
        result = execute_predict_job(job,paths,request_hash) if job["operation"] == "predict" else execute_job(job,paths,request_hash)
    except Exception as error:
        result = {"schema_version":SCHEMA_VERSION,"job_id":job["job_id"],"round_id":job["round_id"],
                  "request_sha256":request_hash,"status":"failed",
                  "error":{"type":type(error).__name__,"message":str(error)}}
    write_json_new(result_path,result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        result = run_job(args.job)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if result["status"] == "completed" else 1
    except Exception as error:
        print(str(error),file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
