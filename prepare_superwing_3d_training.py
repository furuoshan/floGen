"""Prepare immutable SuperWing/VLM/frozen-airfoil inputs for 3-D training.

Only case geometry and operating conditions enter the official PyTornado SLD
calculation. CFD surface fields are retained as read-only training targets.
The expensive, frozen 2-D predictions are cached once and bound to their exact
checkpoint SHA256. Preparation can resume from per-geometry/per-case markers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import main as m
from low_fidelity_sld import estimate_spanwise_lift


ROOT = Path(__file__).resolve().parent
SCHEMA = "superwing_vlm_training_cache_v1"
SOURCE_NAMES = ("configs.dat", "index.npy", "geom0.npy", "origingeom.npy",
                "data.npy", "running_info.jsonl")


def require(value, message):
    if not value:
        raise ValueError(message)


def inside(path, project_root, label):
    path, project_root = Path(path).resolve(), Path(project_root).resolve()
    require(path == project_root or project_root in path.parents,
            f"{label} must remain inside project directory {project_root}: {path}")
    return path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_split(groups, seed=42):
    """Exact 9:1 samples, with entire geometries assigned to the test set.

    For N not divisible by ten, at most nine training-side rows are explicitly
    excluded. No source file is changed. A seeded subset-sum chooses whole test
    geometries whose number of cases is floor(N/10).
    """
    groups = np.asarray(groups)
    require(groups.ndim == 1 and len(groups) >= 10, "At least ten cases are required")
    require(np.issubdtype(groups.dtype, np.integer), "Geometry groups must be integer IDs")
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    require(len(unique) >= 2, "At least two geometries are required")
    ntest = len(groups) // 10
    order = np.random.default_rng(seed).permutation(len(unique))
    reachable = 1
    limit = (1 << (ntest + 1)) - 1
    parent_group = np.full(ntest + 1, -1, dtype=np.int64)
    previous = np.full(ntest + 1, -1, dtype=np.int64)
    for group_idx in order:
        weight = int(counts[group_idx])
        new = ((reachable << weight) & limit) & ~reachable
        pending = new
        while pending:
            bit = pending & -pending
            total = bit.bit_length() - 1
            parent_group[total] = group_idx
            previous[total] = total - weight
            pending ^= bit
        reachable |= new
        if (reachable >> ntest) & 1:
            break
    require(bool((reachable >> ntest) & 1),
            "Cannot make exact 9:1 using whole test geometries; no sample split was guessed")
    selected = []
    total = ntest
    while total:
        group_idx = int(parent_group[total])
        require(group_idx >= 0, "Invalid split reconstruction")
        selected.append(group_idx)
        total = int(previous[total])
    test_mask = np.isin(inverse, np.asarray(selected))
    test = np.flatnonzero(test_mask).astype(np.int64)
    train_pool = np.flatnonzero(~test_mask).astype(np.int64)
    nexcluded = len(groups) - ntest * 10
    excluded = np.sort(np.random.default_rng(seed + 1).choice(
        train_pool, size=nexcluded, replace=False)).astype(np.int64)
    train = train_pool[~np.isin(train_pool, excluded)]
    require(len(train) == 9 * len(test), "Internal sample ratio error")
    require(not np.intersect1d(groups[train], groups[test]).size, "Geometry leakage")
    manifest = {
        "schema": "superwing_geometry_heldout_exact_9_to_1_v1", "seed": int(seed),
        "excluded_seed": int(seed + 1), "source_samples": len(groups),
        "train_samples": len(train), "test_samples": len(test),
        "excluded_samples": len(excluded), "train_test_ratio": "9:1",
        "train_geometry_count": len(np.unique(groups[train])),
        "test_geometry_count": len(np.unique(groups[test])),
        "geometry_overlap_count": 0,
        "test_geometry_ids": np.unique(groups[test]).tolist(),
        "excluded_source_rows": excluded.tolist(),
        "excluded_reason": "Fixed training-side remainder needed for an exact sample ratio; source arrays unchanged",
        "test_used_for_training_or_normalization": False,
    }
    return {"train_rows": train, "test_rows": test, "excluded_rows": excluded}, manifest


def _save_or_verify(path, values):
    values = np.asarray(values)
    if path.exists():
        existing = np.load(path, mmap_mode="r", allow_pickle=False)
        require(existing.dtype == values.dtype and existing.shape == values.shape and
                np.array_equal(existing, values), f"Existing immutable cache differs: {path}")
        mapping = getattr(existing, "_mmap", None)
        if mapping is not None:
            mapping.close()
    else:
        np.save(path, values, allow_pickle=False)


def _open_array(path, shape, dtype, *, initial=None):
    dtype = np.dtype(dtype)
    if path.exists():
        array = np.load(path, mmap_mode="r+", allow_pickle=False)
        require(array.shape == shape and array.dtype == dtype,
                f"Incomplete cache array has incompatible shape/type: {path}")
    else:
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        if initial is not None:
            array[...] = initial
            array.flush()
    return array


def _source_manifest(source_root, project_root, log=print):
    result = {}
    for name in SOURCE_NAMES:
        path = inside(source_root / name, project_root, f"Native source {name}")
        require(path.is_file(), f"Missing native SuperWing source: {path}")
        log(json.dumps({"phase": "hash_source", "file": name, "bytes": path.stat().st_size}), flush=True)
        result[name] = {"path": str(path), "sha256": sha256_file(path),
                        "size": path.stat().st_size}
    return result


def _close_array(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _close_dataset(dataset):
    for name in ("index", "centers", "vertices", "data", "reynolds_array"):
        value = getattr(dataset, name, None)
        if value is not None:
            _close_array(value)


def prepare(source_root, cache_root, airfoil_checkpoint, *, project_root=None,
            device="cpu", seed=42, chord_panels=4, span_panels=32,
            prior_batch_size=2, section_batch_size=256, max_cases=None):
    """Prepare or resume cache. Explicit project_root supports isolated tests."""
    project_root = ROOT if project_root is None else Path(project_root).resolve()
    source_root = inside(source_root, project_root, "SuperWing source")
    cache_root = inside(cache_root, project_root, "Cache output")
    airfoil_checkpoint = inside(airfoil_checkpoint, project_root, "Airfoil checkpoint")
    require(source_root != cache_root and source_root not in cache_root.parents and
            cache_root not in source_root.parents, "Cache and source trees must be separate")
    require(prior_batch_size > 0 and section_batch_size > 0, "Prior batch sizes must be positive")
    require(max_cases is None or max_cases > 0, "max_cases must be positive")
    require(type(seed) is int and 0 <= seed < 2**32-1, "seed must be in [0,2^32-2]")
    manifest_path = cache_root / "manifest.json"
    if cache_root.exists():
        require(not any(path.is_symlink() for path in cache_root.rglob("*")),
                "Cache cannot contain symlinks; preparation never follows cache paths outside the project")
    require(not cache_root.exists() or manifest_path.is_file() or not any(cache_root.iterdir()),
            "Refusing an existing unrecognized cache directory")
    source_files = _source_manifest(source_root, project_root)
    checkpoint_hash = sha256_file(airfoil_checkpoint)
    airfoil, spec, checkpoint_info = m.load_checkpoint(airfoil_checkpoint, "airfoil")
    source = m.SuperWingData(source_root, target="none", require_reynolds=True)
    arrays = []
    recognized_manifest = False
    started = time.monotonic()
    try:
        targets = m.read_npy(source_root / "data.npy")
        require(targets.shape == (len(source), 3, 128, 256), "Invalid native CFD target shape")
        target_dtype = str(targets.dtype)
        _close_array(targets)
        split, split_info = build_split(source.groups, seed)
        immutable = {
            "schema": SCHEMA, "source_root": str(source_root), "source_files": source_files,
            "targets_path": str(source_root / "data.npy"), "targets_dtype": target_dtype,
            "airfoil_checkpoint": str(airfoil_checkpoint), "airfoil_checkpoint_sha256": checkpoint_hash,
            "airfoil_training_steps": checkpoint_info["training"]["training_steps"],
            "spec": asdict(spec), "sample_count": len(source), "geometry_count": len(source.configs),
            "split": split_info, "sld_source": "pytornado_vlm", "sld_uses_cfd": False,
            "sld_network_used": False, "vlm_chord_panels": chord_panels,
            "vlm_span_panels_per_half": span_panels,
            "vlm_implementation": "low_fidelity_sld.estimate_spanwise_lift using official PyTornado native routines",
            "cache_dtypes": {"geometry": "float32", "prior": "float32", "conditions": "float32",
                             "sld": "float64", "reynolds": "float64"},
            "field_channels": list(m.CHANNELS), "stored_multipliers": list(m.WING_MULTIPLIERS),
            "source_fields_used_for_sld": False,
        }
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, value in immutable.items():
                require(manifest.get(name) == value, f"Cache provenance changed: {name}")
        else:
            cache_root.mkdir(parents=True, exist_ok=True)
            manifest = {**immutable, "status": "preparing", "created_utc": datetime.now(timezone.utc).isoformat()}
            write_json(manifest_path, manifest)
        recognized_manifest = True
        split_root = cache_root / "split"
        split_root.mkdir(exist_ok=True)
        for name, rows in split.items():
            _save_or_verify(split_root / (name + ".npy"), rows)
        split_manifest_path = split_root / "manifest.json"
        if split_manifest_path.exists():
            require(json.loads(split_manifest_path.read_text(encoding="utf-8")) == split_info,
                    "Existing split manifest differs")
        else:
            write_json(split_manifest_path, split_info)
        _save_or_verify(cache_root / "conditions.npy", source.condition_matrix())
        _save_or_verify(cache_root / "geometry_indices.npy", source.shape_ids.astype(np.int64))
        _save_or_verify(cache_root / "groups.npy", source.groups.astype(np.int64))
        actual_re = np.asarray([source.reynolds_at(i) for i in range(len(source))], dtype=np.float64)
        _save_or_verify(cache_root / "reynolds.npy", actual_re)
        geometry = _open_array(cache_root / "geometry.npy", (len(source.configs), 3, 128, 256), np.float32)
        prior = _open_array(cache_root / "prior.npy", (len(source), 2, 128, 256), np.float32)
        sld = _open_array(cache_root / "sld.npy", (len(source), 128), np.float64)
        geometry_done = _open_array(cache_root / "geometry_done.npy", (len(source.configs),), np.bool_, initial=False)
        sld_done = _open_array(cache_root / "sld_done.npy", (len(source),), np.bool_, initial=False)
        prior_done = _open_array(cache_root / "prior_done.npy", (len(source),), np.bool_, initial=False)
        arrays.extend((geometry, prior, sld, geometry_done, sld_done, prior_done))
        selected = np.sort(np.r_[split["train_rows"], split["test_rows"]])
        selected_mask = np.zeros(len(source), dtype=bool)
        selected_mask[selected] = True
        if manifest.get("status") == "complete":
            require(bool(prior_done[selected].all() and sld_done[selected].all()),
                    "Complete manifest has missing case markers")
            require(bool(geometry_done[np.unique(source.shape_ids[selected])].all()),
                    "Complete manifest has missing geometry markers")
            return manifest
        # The frozen prior is computed once. The unused random residual/SLD
        # modules never execute, train, or contribute to the prepared fields.
        model = m.PhysicsEmbeddedWingPredictor(
            replace(spec, section_batch_size=section_batch_size), airfoil=airfoil).to(device).eval()
        airfoil_state_before = state_digest(model.airfoil)
        require(all(not p.requires_grad for p in model.airfoil.parameters()), "Airfoil was not frozen")
        conditions = source.condition_matrix()
        used_sld_info = manifest.get("vlm_details")
        done_this_call = 0
        pending = selected[~np.asarray(prior_done[selected])]
        if max_cases is not None:
            pending = pending[:max_cases]
        groups_in_order = np.unique(source.shape_ids[pending])
        for g in groups_in_order:
            g = int(g)
            rows = pending[source.shape_ids[pending] == g]
            features = source.geometry_at(g)
            if not geometry_done[g]:
                geometry[g] = features["geometry"]
                geometry.flush()
                geometry_done[g] = True
                geometry_done.flush()
            for row in rows:
                row = int(row)
                if not sld_done[row]:
                    values, info = estimate_spanwise_lift(
                        source.centers[g], source.vertices[g],
                        mach=float(source.index[row, 3]), aoa_deg=float(source.index[row, 2]),
                        chord_panels=chord_panels, span_panels=span_panels)
                    require(values.shape == (1, 128) and bool(np.isfinite(values).all()),
                            f"Invalid VLM output for source row {row}")
                    require(info.get("uses_cfd") is False and info.get("solver") == "pytornado",
                            "Only geometry-based official PyTornado SLD is allowed")
                    if used_sld_info is None:
                        used_sld_info = {key: info[key] for key in (
                            "solver", "version", "backend", "method", "uses_cfd", "chord_panels",
                            "span_panels_per_half", "total_panels", "compressibility_correction",
                            "camber_surface", "symmetry", "coordinate_mapping", "output_quantity",
                            "output_shape", "interpolation", "limitations")}
                    require(info.get("version") == used_sld_info["version"], "VLM version changed during preparation")
                    sld[row] = values[0]
                    sld.flush()
                    sld_done[row] = True
                    sld_done.flush()
            for start in range(0, len(rows), prior_batch_size):
                batch_rows = rows[start:start + prior_batch_size]
                count = len(batch_rows)
                batch = {key: torch.from_numpy(np.repeat(value[None], count, axis=0).copy()).to(device)
                         for key, value in features.items()}
                batch.update(conditions=m.tensor(conditions[batch_rows]).to(device),
                             reynolds=m.tensor(actual_re[batch_rows]).to(device),
                             sld=m.tensor(sld[batch_rows]).to(device))
                with torch.inference_mode():
                    prediction, _ = model.prior(batch)
                require(prediction.shape == (count, 2, 128, 256) and
                        bool(torch.isfinite(prediction).all()), "Invalid frozen 2-D prior")
                prior[batch_rows] = prediction.cpu().numpy()
                prior.flush()
                prior_done[batch_rows] = True
                prior_done.flush()
                done_this_call += count
            completed = int(np.asarray(prior_done[selected]).sum())
            manifest.update(status="preparing", geometry_completed=int(geometry_done.sum()),
                            cases_completed=completed, cases_required=len(selected),
                            vlm_details=used_sld_info, prior_section_batch_size=section_batch_size,
                            elapsed_seconds_this_call=time.monotonic() - started)
            write_json(manifest_path, manifest)
            print(json.dumps({"phase": "prepare", "geometry": g, "cases_completed": completed,
                              "cases_required": len(selected), "cases_this_call": done_this_call,
                              "elapsed_seconds_this_call": time.monotonic() - started}), flush=True)
        require(state_digest(model.airfoil) == airfoil_state_before,
                "Frozen airfoil parameters or normalization buffers changed")
        require(sha256_file(airfoil_checkpoint) == checkpoint_hash, "Source airfoil checkpoint changed")
        completed = bool(prior_done[selected].all() and sld_done[selected].all() and
                         geometry_done[np.unique(source.shape_ids[selected])].all())
        manifest.update(status="complete" if completed else "preparing",
                        cases_completed=int(np.asarray(prior_done[selected]).sum()),
                        geometry_completed=int(geometry_done.sum()), cases_required=len(selected),
                        airfoil_state_unchanged=True, vlm_details=used_sld_info,
                        elapsed_seconds_this_call=time.monotonic() - started)
        if completed:
            manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["cache_file_hashes"] = {}
            for name in ("geometry.npy", "prior.npy", "conditions.npy", "geometry_indices.npy",
                         "groups.npy", "sld.npy", "reynolds.npy", "geometry_done.npy",
                         "sld_done.npy", "prior_done.npy", "split/train_rows.npy",
                         "split/test_rows.npy", "split/excluded_rows.npy", "split/manifest.json"):
                path = cache_root / name
                manifest["cache_file_hashes"][name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
        write_json(manifest_path, manifest)
        return manifest
    except Exception as error:
        if recognized_manifest and manifest_path.is_file():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            current.update(status="failed", error_type=type(error).__name__, error=str(error))
            write_json(manifest_path, current)
        raise
    finally:
        for array in arrays:
            _close_array(array)
        _close_dataset(source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--airfoil-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chord-panels", type=int, default=4)
    parser.add_argument("--span-panels", type=int, default=32)
    parser.add_argument("--prior-batch-size", type=int, default=2)
    parser.add_argument("--section-batch-size", type=int, default=256)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args(argv)
    device = m.selected_device(args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result = prepare(args.source_root, args.cache_root, args.airfoil_checkpoint,
                     device=device, seed=args.seed, chord_panels=args.chord_panels,
                     span_panels=args.span_panels, prior_batch_size=args.prior_batch_size,
                     section_batch_size=args.section_batch_size, max_cases=args.max_cases)
    print(json.dumps({key: result.get(key) for key in (
        "status", "sample_count", "geometry_count", "cases_completed", "cases_required",
        "elapsed_seconds_this_call", "airfoil_checkpoint_sha256", "split")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
