"""Train the SuperWing residual with frozen 2-D weights and cached PyTornado priors.

Prepared inputs, outputs, temporary files and checkpoints must stay inside this
checkout. The fixed test set never fits normalization, updates parameters, changes
the learning-rate schedule or selects a checkpoint. Every saved epoch is the last
completed epoch, and immutable snapshots are retained at the requested interval.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch

import main as m


ROOT = Path(__file__).resolve().parent
RUN_KIND = "superwing_frozen_airfoil_vlm_fixed_test_v1"
CHANNELS = (("cp", 1., "Cp"), ("cf_tau", 150., "Cf_tau"), ("cf_z", 300., "Cf_z"))
METRIC_DEFINITIONS = {
    "nmae_range_percent": "100*mean(abs(prediction-reference))/(max(reference)-min(reference)), per complete sample surface and channel",
    "relative_l1_percent": "100*sum(abs(prediction-reference))/sum(abs(reference)), per sample surface and channel",
    "distribution_range_error_percent": "100*abs(range(prediction)-range(reference))/range(reference)",
    "error_range_percentage_points": "maximum sample error percentage minus minimum sample error percentage",
    "normalized_mse": "MSE after the fixed training-only target min/max transform",
    "zero_denominator_policy": "Undefined percentages are null, excluded from percentage summaries and counted; no epsilon substitution",
    "physical_cf_units": "stored Cf_tau/150 and stored Cf_z/300; all coefficients are dimensionless",
}


def project_path(value, base=None):
    path = Path(value)
    if not path.is_absolute():
        path = Path(base or ROOT) / path
    path = path.resolve()
    root = ROOT.resolve()
    m.require(path == root or root in path.parents, f"Path is outside the authorized project: {path}")
    return path


def digest_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_json(path, value):
    path = project_path(path)
    temporary = path.with_name("." + path.name + ".partial-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def normalizer_digest(model):
    return {name: state_digest(module) for name, module in model.residual.named_modules()
            if isinstance(module, m.FeatureNorm)}


class PreparedWingData:
    """Read-only cache made by prepare_superwing_vlm.py; target arrays are not copied."""
    def __init__(self, cache_root):
        self.root = project_path(cache_root)
        self.manifest_path = project_path(self.root / "manifest.json")
        manifest_bytes = self.manifest_path.read_bytes()
        self.manifest = json.loads(manifest_bytes)
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        m.require(self.manifest.get("schema") == "superwing_vlm_training_cache_v1" and
                  self.manifest.get("status") == "complete", "VLM cache is not complete or has an incompatible schema")
        m.require(self.manifest.get("sld_source") == "pytornado_vlm" and
                  self.manifest.get("sld_uses_cfd") is False, "Training requires geometry-only official PyTornado SLD")
        for name in ("geometry", "prior", "conditions", "geometry_indices", "groups", "sld", "reynolds"):
            setattr(self, name, m.read_npy(project_path(self.root / (name + ".npy"))))
        self.target_path = project_path(self.manifest["targets_path"], self.root)
        self.targets = m.read_npy(self.target_path)
        n, g = len(self.conditions), len(self.geometry)
        expected = {"geometry": (g, 3, 128, 256), "prior": (n, 2, 128, 256),
                    "conditions": (n, 2), "geometry_indices": (n,), "groups": (n,),
                    "sld": (n, 128), "reynolds": (n,), "targets": (n, 3, 128, 256)}
        for name, shape in expected.items():
            m.require(getattr(self, name).shape == shape, f"Prepared {name} shape mismatch")
        m.require(n == self.manifest["sample_count"] and g == self.manifest["geometry_count"], "Cache sample count mismatch")
        m.require(self.geometry_indices.dtype.kind in "iu" and self.groups.dtype.kind in "iu", "Geometry identifiers must be integer arrays")
        m.require(bool(((self.geometry_indices >= 0) & (self.geometry_indices < g)).all()), "Invalid geometry cache index")
        self.train = self._rows("train_rows")
        self.test = self._rows("test_rows")
        excluded = project_path(self.root / "split/excluded_rows.npy")
        self.excluded = np.asarray(np.load(excluded, allow_pickle=False), dtype=np.int64) if excluded.is_file() else np.empty(0, dtype=np.int64)
        m.require(len(self.train) == 9 * len(self.test), "Prepared train/test ratio must be exactly 9:1")
        combined = np.r_[self.train, self.test, self.excluded]
        m.require(len(combined) == n and np.array_equal(np.sort(combined), np.arange(n)), "Split must partition all source rows exactly once")
        m.require(not np.intersect1d(self.groups[self.train], self.groups[self.test]).size, "Training and test geometries overlap")
        self.split_signature = hashlib.sha256(self.train.tobytes() + self.test.tobytes() + self.excluded.tobytes()).hexdigest()

    def _rows(self, name):
        values = m.read_npy(project_path(self.root / "split" / (name + ".npy")))
        m.require(values.ndim == 1 and values.dtype.kind in "iu" and len(values) > 0, f"Invalid {name}")
        copied = np.array(values, dtype=np.int64, copy=True)
        values._mmap.close()
        m.require(len(np.unique(copied)) == len(copied) and bool(((copied >= 0) & (copied < len(self))).all()), f"Invalid {name} indices")
        return copied

    def __len__(self):
        return len(self.conditions)

    def iter_batches(self, rows, batch_size, device, shuffle=False, seed=42):
        chosen = np.array(rows, dtype=np.int64, copy=True)
        if shuffle:
            np.random.default_rng(seed).shuffle(chosen)
        for start in range(0, len(chosen), batch_size):
            selected = chosen[start:start + batch_size]
            geometry_rows = np.asarray(self.geometry_indices[selected], dtype=np.int64)
            values = {"geometry": self.geometry[geometry_rows], "prior": self.prior[selected],
                      "conditions": self.conditions[selected], "target": self.targets[selected]}
            result = {"row": torch.from_numpy(selected.copy())}
            for name, array in values.items():
                block = torch.from_numpy(np.array(array, dtype=np.float32, order="C", copy=True))
                if str(device).startswith("cuda"):
                    block = block.pin_memory()
                result[name] = block.to(device, non_blocking=True)
            yield result

    def close(self):
        for name in ("geometry", "prior", "conditions", "geometry_indices", "groups", "sld", "reynolds", "targets"):
            mapping = getattr(getattr(self, name, None), "_mmap", None)
            if mapping is not None:
                mapping.close()


def cached_forward(model, batch):
    """Exactly the full model after its already-frozen prior stage, not a new model."""
    prior = batch["prior"]
    delta = model.residual(torch.cat((batch["geometry"], prior), dim=1), batch["conditions"])
    return delta + torch.cat((prior, torch.zeros_like(prior[:, :1])), dim=1)


@torch.no_grad()
def fit_train_normalizers(model, data, rows, batch_size, device):
    model.to(device).eval()
    bounds = {}
    for batch in data.iter_batches(rows, batch_size, device):
        for normalizer, values in ((model.residual.normalizer, batch["conditions"]),
                                   (model.residual.input_normalizer, torch.cat((batch["geometry"], batch["prior"]), dim=1)),
                                   (model.residual.output_normalizer, batch["target"])):
            low, high = normalizer.bounds(values)
            if normalizer in bounds:
                low, high = torch.minimum(low, bounds[normalizer][0]), torch.maximum(high, bounds[normalizer][1])
            bounds[normalizer] = low, high
    m.require(len(bounds) == 3, "No training population for normalization")
    for normalizer, (low, high) in bounds.items():
        normalizer.fit_bounds(low, high)


def _percent(numerator, denominator):
    values = np.full(np.asarray(numerator).shape, np.nan, dtype=np.float64)
    np.divide(100 * numerator, denominator, out=values, where=denominator > 0)
    m.require(np.isfinite(values[denominator > 0]).all(), "Non-finite defined percentage")
    return values


def _summary(values, rows):
    valid = np.isfinite(values)
    good = values[valid]
    result = {"valid_samples": int(valid.sum()), "undefined_samples": int((~valid).sum()),
              "undefined_source_rows": np.asarray(rows)[~valid].astype(int).tolist()}
    if not len(good):
        return dict(result, mean=None, minimum=None, maximum=None, error_range_percentage_points=None,
                    p50=None, p95=None, p99=None, minimum_source_row=None, maximum_source_row=None)
    source = np.asarray(rows)[valid]
    return dict(result, mean=float(good.mean()), minimum=float(good.min()), maximum=float(good.max()),
                error_range_percentage_points=float(np.ptp(good)), p50=float(np.quantile(good, .5)),
                p95=float(np.quantile(good, .95)), p99=float(np.quantile(good, .99)),
                minimum_source_row=int(source[good.argmin()]), maximum_source_row=int(source[good.argmax()]))


def autocast_context(device, amp):
    if amp == "none":
        return nullcontext()
    m.require(str(device).startswith("cuda"), "AMP requires CUDA")
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bfloat16" else torch.float16)


@torch.inference_mode()
def evaluate_epoch(model, data, rows, batch_size, device, amp="none"):
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    m.require(len(rows) > 0, "Empty test population")
    blocks, seen = [], 0
    for batch in data.iter_batches(rows, batch_size, device):
        with autocast_context(device, amp):
            prediction = cached_forward(model, batch)
        target = batch["target"]
        m.require(prediction.shape == target.shape and prediction.shape[1:] == (3, 128, 256), "Wrong surface output shape")
        m.require(np.array_equal(batch["row"].cpu().numpy(), rows[seen:seen + len(prediction)]), "Test source order changed")
        predicted, reference = prediction.double().flatten(2), target.double().flatten(2)
        error = (predicted - reference).abs()
        normalized = (m.normalize_target(model, prediction.float(), "wing") -
                      m.normalize_target(model, target, "wing")).double().flatten(2)
        scores = torch.stack((error.mean(2), error.square().mean(2), reference.amax(2) - reference.amin(2),
                              reference.abs().mean(2), predicted.amax(2) - predicted.amin(2),
                              normalized.square().mean(2)), dim=-1).cpu().numpy()
        m.require(bool(np.isfinite(scores).all()), "Non-finite test predictions or labels")
        blocks.append(scores)
        seen += len(prediction)
    m.require(seen == len(rows), "Test did not evaluate every held-out record")
    scores = np.concatenate(blocks)
    nmae = _percent(scores[:, :, 0], scores[:, :, 2])
    relative_l1 = _percent(scores[:, :, 0], scores[:, :, 3])
    ranges = _percent(np.abs(scores[:, :, 4] - scores[:, :, 2]), scores[:, :, 2])
    result = {"test_samples": seen, "surface_cells": 128 * 256, "normalized_mse": float(scores[:, :, 5].mean()),
              "metric_definitions": METRIC_DEFINITIONS, "test_used_for_parameter_updates": False,
              "normalizers_fitted_here": False}
    for channel, (name, multiplier, physical_unit) in enumerate(CHANNELS):
        result[name] = {"nmae_range_percent": _summary(nmae[:, channel], rows),
                        "relative_l1_percent": _summary(relative_l1[:, channel], rows),
                        "distribution_range_error_percent": _summary(ranges[:, channel], rows),
                        "stored_mae": float(scores[:, channel, 0].mean()),
                        "stored_rmse": float(math.sqrt(scores[:, channel, 1].mean())),
                        "physical_mae": float(scores[:, channel, 0].mean() / multiplier),
                        "physical_rmse": float(math.sqrt(scores[:, channel, 1].mean()) / multiplier),
                        "stored_multiplier": multiplier, "physical_unit": physical_unit}
    return result


def _atomic_figure(fig, destination):
    temporary = destination.with_name("." + destination.stem + ".partial-" + uuid.uuid4().hex + ".png")
    try:
        fig.savefig(temporary, dpi=150, bbox_inches="tight", format="png", facecolor="white")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def plot_history(history, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build/mpl_wing_training"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [entry["epoch"] for entry in history]
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    train = np.array([entry["train_normalized_mse"] for entry in history])
    test = np.array([entry["test"]["normalized_mse"] for entry in history])
    ax.plot(epochs, train, label="Train: epoch-average normalized MSE")
    ax.plot(epochs, test, label="Test: end-of-epoch normalized MSE")
    if (train > 0).all() and (test > 0).all():
        ax.set_yscale("log")
    ax.set(xlabel="Epoch", ylabel="Normalized MSE", title=f"3-D wing residual loss | epoch {epochs[-1]}")
    ax.grid(alpha=.25, which="both"); ax.legend(); fig.tight_layout()
    loss = output / "loss_vs_epoch.png"
    _atomic_figure(fig, loss); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    panels = (("nmae_range_percent", "mean", "Mean test-surface error", "NMAE / reference range (%)"),
              ("nmae_range_percent", "maximum", "Worst test-surface error", "NMAE / reference range (%)"),
              ("nmae_range_percent", "error_range_percentage_points", "Spread of test errors", "Maximum - minimum (percentage points)"),
              ("distribution_range_error_percent", "mean", "Predicted surface range error", "Mean range error (%)"))
    for ax, (metric, stat, title, ylabel) in zip(axes.flat, panels):
        for name, _, label in CHANNELS:
            values = [entry["test"][name][metric][stat] for entry in history]
            ax.plot(epochs, [np.nan if v is None else v for v in values], label=label)
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title); ax.grid(alpha=.25); ax.legend()
    fig.suptitle(f"Fixed held-out SuperWing test set | epoch {epochs[-1]}")
    fig.tight_layout(rect=(0, 0, 1, .96))
    errors = output / "test_error_percent_vs_epoch.png"
    _atomic_figure(fig, errors); plt.close(fig)
    return {"loss_vs_epoch": str(loss), "test_error_percent_vs_epoch": str(errors)}


def save_epoch(output, model, spec, optimizer, scaler, info, history, interval):
    temporary = output / (".latest.partial-" + uuid.uuid4().hex)
    payload = {"format": m.CHECKPOINT_FORMAT, "stage": "wing", "spec": asdict(spec),
               "state_dict": cpu_tree(model.state_dict()), "training": dict(info),
               "conventions": {"wing_channels": list(m.CHANNELS), "wing_multipliers": [1, 150, 300],
                   "foil_channels": ["cp", "cf_times_1000"], "foil_cf_multiplier": 1000,
                   "reynolds_input_unit": "millions", "normalization": m.NORMALIZATION},
               "optimizer_state": cpu_tree(optimizer.state_dict()), "grad_scaler_state": scaler.state_dict(),
               "epoch_history": history, "cpu_rng_state": torch.get_rng_state(),
               "cuda_rng_states": [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
    torch.save(payload, temporary)
    temporary.replace(output / "latest.pt")
    if info["epochs_completed"] % interval == 0:
        snapshot = output / f"epoch_{info['epochs_completed']:03d}.pt"
        m.require(not snapshot.exists(), f"Immutable snapshot already exists: {snapshot}")
        # A hard link is constant-time and survives future atomic latest.pt replacements.
        os.link(output / "latest.pt", snapshot)
        publish_checkpoint_manifest(output, snapshot, info["epochs_completed"], info.get("run_id"), interval)
    return output / "latest.pt"


def publish_checkpoint_manifest(output, snapshot, epoch, run_id, interval=10):
    """Publish only complete immutable snapshots for the local SSH downloader."""
    path = project_path(output / "checkpoints.json")
    snapshot = project_path(snapshot)
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"checkpoints": []}
    m.require(manifest.get("run_id", run_id) == run_id, "Checkpoint transfer manifest belongs to a different run")
    entry = {"epoch": epoch, "path": snapshot.relative_to(output).as_posix(),
             "sha256": digest_file(snapshot), "size_bytes": snapshot.stat().st_size}
    previous = next((item for item in manifest["checkpoints"] if item["epoch"] == epoch), None)
    m.require(previous is None or previous == entry, "Immutable checkpoint changed")
    if previous is None:
        manifest["checkpoints"].append(entry)
    manifest.update(run_id=run_id, checkpoint_every=interval)
    atomic_json(path, manifest)


def verify_source_files(data):
    m.require(digest_file(data.manifest_path) == data.manifest_sha256,
              "Prepared manifest changed after it was loaded")
    sources = data.manifest.get("source_files", {})
    m.require(bool(sources) and any(project_path(item["path"], data.root) == data.target_path
                                  for item in sources.values()), "Cache must bind the original target file SHA256")
    for name, identity in sources.items():
        path = project_path(identity["path"], data.root)
        m.require(path.is_file() and path.stat().st_size == identity["size"] and
                  digest_file(path) == identity["sha256"], f"Read-only source changed: {name}")
    hashes = data.manifest.get("cache_file_hashes", {})
    required = {name + ".npy" for name in ("geometry", "prior", "conditions", "geometry_indices", "groups", "sld", "reynolds")}
    required.update("split/" + name + ".npy" for name in ("train_rows", "test_rows", "excluded_rows"))
    m.require(required.issubset(hashes), "Prepared cache lacks mandatory content checksums")
    for name, identity in hashes.items():
        path = project_path(name, data.root)
        m.require(path.is_file() and path.stat().st_size == identity["size"] and
                  digest_file(path) == identity["sha256"], f"Prepared cache changed: {name}")


def run(args):
    output = project_path(args.output_dir)
    cache_root = project_path(args.cache_root)
    airfoil_path = project_path(args.airfoil_checkpoint)
    m.require(output != cache_root and output not in cache_root.parents and cache_root not in output.parents,
              "Training output and prepared cache must be separate trees")
    m.require(args.epochs > 0 and args.batch_size > 0 and args.test_batch_size > 0 and args.checkpoint_every > 0
              and args.cpu_threads > 0 and args.progress_every > 0,
              "Invalid epoch, batch or checkpoint interval")
    m.require(args.learning_rate > 0 and math.isfinite(args.learning_rate), "Invalid learning rate")
    m.seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = m.selected_device(args.device)
    m.require(args.amp == "none" or str(device).startswith("cuda"), "AMP requires CUDA")
    if args.amp == "bfloat16":
        m.require(torch.cuda.is_bf16_supported(), "This GPU does not support bfloat16")
    data = PreparedWingData(cache_root)
    output.mkdir(parents=True, exist_ok=True)
    try:
        atomic_json(output / "status.json", {"status": "verifying_prepared_inputs",
                    "epochs_requested": args.epochs, "source_and_cache_sha256_required": True})
        verify_source_files(data)
        checkpoint_sha = digest_file(airfoil_path)
        m.require(checkpoint_sha == data.manifest["airfoil_checkpoint_sha256"], "Cached prior uses different 2-D weights")
        airfoil, spec, predecessor = m.load_checkpoint(airfoil_path, "airfoil")
        expected_airfoil = state_digest(airfoil)
        if "spec" in data.manifest:
            m.require(asdict(spec) == data.manifest["spec"], "Cache ModelSpec differs from the 2-D checkpoint")
        model = m.PhysicsEmbeddedWingPredictor(spec, airfoil=airfoil).to(device)
        optimizer = torch.optim.Adam(model.residual.parameters(), lr=args.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda") and args.amp == "float16")
        history, start_epoch = [], 0
        info = {"run_kind": RUN_KIND, "run_id": uuid.uuid4().hex, "started_utc": datetime.now(timezone.utc).isoformat(),
                "epochs_requested": args.epochs, "epochs_completed": 0, "training_steps": 0, "samples_seen": 0,
                "training_samples": len(data.train), "test_samples": len(data.test), "excluded_samples": len(data.excluded),
                "train_shape_ids": np.unique(data.groups[data.train]).astype(int).tolist(),
                "test_shape_ids": np.unique(data.groups[data.test]).astype(int).tolist(),
                "train_rows": data.train.tolist(), "test_rows": data.test.tolist(),
                "split_signature": data.split_signature, "cache_manifest_sha256": data.manifest_sha256,
                "airfoil_checkpoint": str(airfoil_path), "airfoil_checkpoint_sha256": checkpoint_sha,
                "airfoil_training_steps": predecessor["training"]["training_steps"],
                "airfoil_epochs_completed": predecessor["training"].get("epochs_completed"),
                "airfoil_reynolds_input": predecessor["training"].get("reynolds_input"),
                "airfoil_frozen": True, "sld_source": "pytornado_vlm", "sld_network_used": False,
                "sld_uses_cfd": False, "sld_training_steps": 0, "sld_train_shape_ids": [],
                "sld_prediction_mode": "external_low_fidelity", "source_files": data.manifest.get("source_files", {}),
                "prior_policy": "Precomputed frozen 2-D outputs plus geometry-only official PyTornado VLM; no CFD-derived SLD",
                "optimizer": "Adam", "batch_size": args.batch_size, "test_batch_size": args.test_batch_size,
                "learning_rate_max": args.learning_rate, "amp": args.amp, "device": str(device), "seed": args.seed,
                "normalization": m.NORMALIZATION, "loss_space": "minmax_normalized",
                "learning_rate_schedule": "60-epoch linear 0.1 to 1 warmup, followed by 0.95 exponential factor every 3 epochs",
                "test_used_for_gradients": False, "test_used_for_normalization": False,
                "test_used_for_checkpoint_selection": False, "checkpoint_selection": "last_completed_epoch_not_best_test",
                "checkpoint_every": args.checkpoint_every, "metric_definitions": METRIC_DEFINITIONS}
        if args.resume:
            payload = torch.load(project_path(output / "latest.pt"), map_location="cpu", weights_only=True)
            previous = payload["training"]
            for key in ("run_kind", "split_signature", "cache_manifest_sha256", "airfoil_checkpoint_sha256",
                        "epochs_requested", "batch_size", "test_batch_size", "learning_rate_max", "amp", "seed", "checkpoint_every"):
                m.require(previous.get(key) == info.get(key), f"Resume setting/source changed: {key}")
            model.load_state_dict(payload["state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state"])
            scaler.load_state_dict(payload["grad_scaler_state"])
            history = payload["epoch_history"]; start_epoch = previous["epochs_completed"]; info = previous
            m.require(len(history) == start_epoch and (not history or history[-1]["epoch"] == start_epoch), "Resume history is inconsistent")
            torch.set_rng_state(payload["cpu_rng_state"])
            if str(device).startswith("cuda") and payload["cuda_rng_states"]:
                torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
            if start_epoch and start_epoch % args.checkpoint_every == 0:
                snapshot = output / f"epoch_{start_epoch:03d}.pt"
                if not snapshot.exists():
                    os.link(output / "latest.pt", snapshot)
                publish_checkpoint_manifest(output, snapshot, start_epoch, info["run_id"], args.checkpoint_every)
        else:
            m.require(not (output / "latest.pt").exists() and not (output / "history.json").exists()
                      and not list(output.glob("epoch_*.pt")), "Training output already exists; use --resume")
            atomic_json(output / "status.json", {"status": "fitting_train_only_normalizers", "run_id": info["run_id"]})
            fit_train_normalizers(model, data, data.train, args.test_batch_size, device)
        m.require(state_digest(model.airfoil) == expected_airfoil, "Resumed 2-D state differs from the specified frozen checkpoint")
        frozen_lift = state_digest(model.lift)
        normalization = normalizer_digest(model)
        elapsed_before = float(info.get("elapsed_seconds", 0.)); began = time.monotonic()
        atomic_json(output / "verification.json", {"status": "pending", "run_id": info["run_id"], "epochs_requested": args.epochs})
        status = {"status": "training", "run_id": info["run_id"], "epoch": start_epoch,
                  "epochs_requested": args.epochs, "checkpoint": str(output / "latest.pt")}
        with (output / "train.log").open("a" if args.resume else "w", encoding="utf-8") as log:
            for epoch in range(start_epoch, args.epochs):
                epoch_start = time.monotonic(); model.train()
                lr = args.learning_rate * m.learning_rate_factor(epoch, "wing")
                for group in optimizer.param_groups: group["lr"] = lr
                loss_sum = torch.zeros((), dtype=torch.float64, device=device)
                seen, steps = 0, 0
                for batch in data.iter_batches(data.train, args.batch_size, device, shuffle=True, seed=args.seed + epoch):
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device, args.amp):
                        prediction = cached_forward(model, batch)
                        loss = torch.nn.functional.mse_loss(m.normalize_target(model, prediction.float(), "wing"),
                                                            m.normalize_target(model, batch["target"], "wing"))
                    m.require(bool(torch.isfinite(loss)), "Non-finite training loss")
                    scaler.scale(loss).backward()
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    # foreach performs one aggregate finite check instead of a GPU
                    # synchronization per parameter tensor; infinity means no clipping.
                    torch.nn.utils.clip_grad_norm_(model.residual.parameters(), float("inf"),
                                                   error_if_nonfinite=True, foreach=True)
                    scaler.step(optimizer); scaler.update()
                    count = len(prediction); seen += count; steps += 1
                    loss_sum += loss.detach().double() * count
                    if steps % args.progress_every == 0:
                        atomic_json(output / "status.json", {"status": "training_epoch", "run_id": info["run_id"],
                            "epoch": epoch + 1, "last_completed_epoch": epoch, "epochs_requested": args.epochs,
                            "epoch_samples_seen": seen, "epoch_training_samples": len(data.train),
                            "epoch_optimizer_steps": steps, "learning_rate": lr,
                            "elapsed_seconds": elapsed_before + time.monotonic() - began})
                m.require(seen == len(data.train), "Epoch did not use every selected training sample")
                train_loss = float(loss_sum.cpu()) / seen
                train_seconds = time.monotonic() - epoch_start
                evaluation_start = time.monotonic()
                metrics = evaluate_epoch(model, data, data.test, args.test_batch_size, device, args.amp)
                m.require(state_digest(model.airfoil) == expected_airfoil and state_digest(model.lift) == frozen_lift,
                          "Frozen 2-D or unused SLD state changed")
                m.require(normalizer_digest(model) == normalization, "Training-only normalization buffers changed")
                history.append({"epoch": epoch + 1, "learning_rate": lr, "train_normalized_mse": train_loss,
                                "test": metrics, "optimizer_steps": steps, "train_seconds": train_seconds,
                                "test_seconds": time.monotonic() - evaluation_start})
                info.update(epochs_completed=epoch + 1, training_steps=info["training_steps"] + steps,
                            samples_seen=info["samples_seen"] + seen, train_mse=train_loss, test=metrics,
                            learning_rate=lr, elapsed_seconds=elapsed_before + time.monotonic() - began,
                            frozen_airfoil_state_sha256=expected_airfoil)
                latest = save_epoch(output, model, spec, optimizer, scaler, info, history, args.checkpoint_every)
                atomic_json(output / "history.json", history)
                atomic_json(output / "training_report.json", info)
                plots = plot_history(history, output)
                status = {"status": "final_verification" if epoch + 1 == args.epochs else "training",
                          "run_id": info["run_id"], "epoch": epoch + 1, "epochs_requested": args.epochs,
                          "training_samples": len(data.train), "test_samples": len(data.test),
                          "train_normalized_mse": train_loss, "test_normalized_mse": metrics["normalized_mse"],
                          "learning_rate": lr, "epoch_wall_seconds": time.monotonic() - epoch_start,
                          "elapsed_seconds": elapsed_before + time.monotonic() - began,
                          "checkpoint": str(latest), "plots": plots,
                          "test_mean_error_percent": {name: metrics[name]["nmae_range_percent"]["mean"] for name, _, _ in CHANNELS},
                          "test_error_range_percentage_points": {name: metrics[name]["nmae_range_percent"]["error_range_percentage_points"] for name, _, _ in CHANNELS}}
                atomic_json(output / "status.json", status)
                log.write(json.dumps(status, ensure_ascii=False, allow_nan=False) + "\n"); log.flush()
                print(json.dumps(status, ensure_ascii=False, allow_nan=False), flush=True)
        m.require(info["epochs_completed"] == args.epochs, "Requested epochs were not completed")
        if start_epoch == args.epochs:
            # A restart after the last atomic checkpoint may need to regenerate
            # sidecar plots/history and finish verification without training again.
            atomic_json(output / "history.json", history)
            status.update(plots=plot_history(history, output), test_normalized_mse=info["test"]["normalized_mse"])
        verify_source_files(data)
        loaded, _, saved = m.load_checkpoint(output / "latest.pt", "wing")
        m.require(saved["training"]["epochs_completed"] == args.epochs, "Final checkpoint is not the requested last epoch")
        m.require(state_digest(loaded.airfoil) == expected_airfoil, "Final checkpoint did not preserve the 2-D weights")
        loaded.to(device)
        reread = evaluate_epoch(loaded, data, data.test, args.test_batch_size, device, args.amp)
        m.require(abs(reread["normalized_mse"] - info["test"]["normalized_mse"]) <=
                  max(1e-8, 1e-5 * abs(info["test"]["normalized_mse"])),
                  "Final checkpoint reload did not reproduce normalized test loss")
        comparison = {name: abs(reread[name]["nmae_range_percent"]["mean"] - info["test"][name]["nmae_range_percent"]["mean"])
                      if reread[name]["nmae_range_percent"]["mean"] is not None else 0. for name, _, _ in CHANNELS}
        m.require(max(comparison.values()) < 1e-4, "Final checkpoint reload did not reproduce test metrics")
        atomic_json(output / "final_test_metrics.json", reread)
        atomic_json(output / "verification.json", {"status": "passed", "run_id": info["run_id"],
                    "epochs_completed": args.epochs, "train_test_ratio": "9:1", "geometry_overlap_count": 0,
                    "frozen_2d_unchanged": True, "normalizers_fitted_train_only": True,
                    "source_files_unchanged": True, "checkpoint_selection": "last_epoch",
                    "checkpoint_reload_mean_percent_abs_difference": comparison,
                    "checkpoint_sha256": digest_file(output / "latest.pt")})
        info.update(status="complete", elapsed_seconds=elapsed_before + time.monotonic() - began)
        status.update(status="complete", elapsed_seconds=info["elapsed_seconds"])
        atomic_json(output / "training_report.json", info); atomic_json(output / "status.json", status)
        print(json.dumps(status, ensure_ascii=False, allow_nan=False), flush=True)
        return output / "latest.pt"
    except Exception as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                   "time_utc": datetime.now(timezone.utc).isoformat()}
        if "info" in locals():
            failure.update(run_id=info["run_id"], last_completed_epoch=info["epochs_completed"],
                           epochs_requested=args.epochs)
        atomic_json(output / "failure.json", failure)
        atomic_json(output / "status.json", failure)
        raise
    finally:
        data.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--airfoil-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--test-batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", choices=("none", "bfloat16", "float16"), default="none")
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    print(str(run(parser().parse_args())), flush=True)
