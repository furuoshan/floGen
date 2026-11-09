"""Split geometry/condition/sectional-CL inputs for prediction without CFD data.

This adapter preserves the native SuperWing coordinates and field conventions.
External sectional CL is mandatory and is bound to the first six native index
columns; it is never inferred from a CFD surface field or silently replaced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

import main as m


FOILPARA_ROOT = Path("D:/data/foilpara")


class FoilParaData(m.SuperWingData):
    """One geometry/condition with 41-column configs; no repeated Mach/AoA slots.

    ``sectional_cl.npy`` contains dimensionless local lift coefficients at the
    128 spanwise cell centers, in each sample's index order. It is not lift in
    newtons, wing-total CL, or the sweep-corrected two-dimensional CL.
    """

    def __init__(self, root=FOILPARA_ROOT, require_reynolds=True):
        self.input_root = Path(root).resolve()
        self.condition_root = self.input_root / "foiloc"
        self.geometry_root = self.input_root / "foilgeo"
        self.sld_root = self.input_root / "foilsld"
        required = (
            self.condition_root / "index.npy",
            self.geometry_root / "configs.dat",
            self.geometry_root / "geom0.npy",
            self.geometry_root / "origingeom.npy",
            self.sld_root / "index.npy",
            self.sld_root / "sectional_cl.npy",
        )
        missing = [str(path.relative_to(self.input_root)) for path in required if not path.is_file()]
        m.require(not missing,
                  f"Incomplete foilpara input at {self.input_root}; missing: {', '.join(missing)}. "
                  "Supply native geometry, conditions and externally estimated sectional CL; "
                  "empty directories cannot be used for prediction.")
        self.reynolds_path = self.condition_root / "reynolds.npy"
        m.require(not require_reynolds or self.reynolds_path.is_file(),
                  f"Missing actual Reynolds numbers: {self.reynolds_path}; expected an (N,) NPY array")
        super().__init__(
            self.condition_root, target="none", require_reynolds=require_reynolds,
            reynolds_file=self.reynolds_path if self.reynolds_path.is_file() else None,
            geometry_root=self.geometry_root, index_columns=(6,), config_columns=(41,))
        m.require(bool((self.index[:, 4:6] > 0).all()),
                  "Foilpara reference half-area and half-span must be positive")
        if self.reynolds_array is not None:
            m.finite(self.reynolds_array, "foiloc actual Reynolds numbers")
            m.require(bool((self.reynolds_array > 0).all()), "Foilpara actual Reynolds numbers must be positive")
        self.sld = m.read_npy(self.sld_root / "sectional_cl.npy")
        self.sld_index = m.read_npy(self.sld_root / "index.npy")
        m.require(self.sld.shape == (len(self.index), 128),
                  "foilsld/sectional_cl.npy must be (N,128) dimensionless sectional CL")
        m.require(self.sld_index.ndim == 2 and self.sld_index.shape == (len(self.index), 6),
                  "foilsld/index.npy must be (N,6), matching foiloc/index.npy's first six columns")
        m.finite(self.sld, "external sectional CL")
        m.finite(self.sld_index, "external sectional CL index")
        m.require(np.array_equal(self.sld_index, self.index[:, :6]),
                  "foilsld/index.npy does not exactly match foiloc/index.npy's first six columns; "
                  "external sectional CL sample identity/order is ambiguous")

    def __getitem__(self, row):
        result = super().__getitem__(row)
        result["sld"] = m.tensor(self.sld[row])
        return result

    def close(self):
        """Release read-only memory maps, including Windows file handles."""
        for name in ("index", "centers", "vertices", "reynolds_array", "sld", "sld_index"):
            value = getattr(self, name, None)
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _destination(output_root, input_root):
    output_root = Path(output_root).resolve()
    input_root = Path(input_root).resolve()
    m.protect_destination(output_root, [input_root])
    m.require(output_root not in input_root.parents,
              "Foilpara output cannot be an ancestor of input data")
    m.require(m.prediction_directory_available(output_root),
              f"Prediction directory must be empty or contain only a regular format.md file: {output_root}")
    return output_root


@torch.inference_mode()
def _export_foilpara(model, spec, dataset, output_root, device, batch_size, provenance):
    """Shared writer; diagnostic callers explicitly label untrained output."""
    m.require(type(batch_size) is int and batch_size > 0, "batch_size must be a positive integer")
    output_root = _destination(output_root, dataset.input_root)
    model.to(device).eval()
    state_before = _state_digest(model)
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / "prediction_info.json"
    metadata = {
        "status": "in_progress",
        "source_root": str(dataset.input_root),
        "sample_count": len(dataset),
        "condition_dim": spec.condition_dim,
        "channels": list(m.CHANNELS),
        "stored_multipliers": list(m.WING_MULTIPLIERS),
        "index_layout": "geometry_conditions_v1",
        "index_coefficients_role": "unavailable_not_measured_not_predicted",
        "sld_source": "external_foilsld_sectional_cl",
        "sld_quantity": "dimensionless_sectional_cl_before_sweep_transform",
        "sld_shape": [len(dataset), 128],
        "sld_sample_binding": "exact_first_six_native_index_columns",
        "source_sectional_cl": "source_sectional_cl.npy",
        "source_reynolds": "source_reynolds.npy" if dataset.reynolds_array is not None else None,
        "training_steps_performed": 0,
        **provenance,
    }
    marker.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    partial = output_root / "data.npy.partial"
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(len(dataset), 3, 128, 256))
    offset = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        values = model(m.move_batch(batch, device))
        m.require(values.shape == (len(batch["row"]), 3, 128, 256),
                  "Foilpara prediction has an unexpected shape")
        m.require(bool(torch.isfinite(values).all()),
                  "Non-finite foilpara prediction; export remains incomplete")
        array = values.cpu().numpy()
        output[offset:offset + len(array)] = array
        offset += len(array)
    m.require(offset == len(dataset), "Foilpara prediction row count mismatch")
    output.flush()
    del output
    m.require(_state_digest(model) == state_before,
              "Model parameters or normalization buffers changed during foilpara prediction")
    # Pure predictions have no measured or predicted integral coefficients.
    np.save(output_root / "index.npy", np.asarray(dataset.index[:, :6]), allow_pickle=False)
    for name in ("configs.dat", "geom0.npy", "origingeom.npy"):
        shutil.copy2(dataset.geometry_root / name, output_root / name)
    shutil.copy2(dataset.sld_root / "sectional_cl.npy", output_root / "source_sectional_cl.npy")
    if dataset.reynolds_array is not None:
        shutil.copy2(dataset.reynolds_path, output_root / "source_reynolds.npy")
    partial.replace(output_root / "data.npy")
    metadata.update(status="complete", model_state_unchanged=True, normalization_unchanged=True)
    marker.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_root / "data.npy"


def predict_foilpara(model, spec, checkpoint_info, input_root=FOILPARA_ROOT,
                    output_root=m.NETWORK_ROOT, device="cpu", batch_size=1):
    """Export a trained wing model using mandatory externally estimated SLD."""
    trained = checkpoint_info.get("training", {})
    m.require(checkpoint_info.get("format") == m.CHECKPOINT_FORMAT and
              checkpoint_info.get("stage") == "wing" and trained.get("training_steps", 0) > 0,
              "Foilpara prediction requires a trained v2 wing checkpoint")
    m.require(trained.get("airfoil_training_steps", 0) > 0 and trained.get("sld_training_steps", 0) > 0,
              "Wing checkpoint must record trained airfoil and SLD predecessors")
    dataset = FoilParaData(input_root, require_reynolds=spec.condition_dim == 3)
    try:
        return _export_foilpara(
            model, spec, dataset, output_root, device, batch_size,
            {"prediction_kind": "trained_checkpoint", "training_steps": trained["training_steps"]})
    finally:
        dataset.close()


def forward_test(device="cpu"):
    """Real 3-D forward and file export only; no optimizer or training steps.

    All native-format inputs and clearly labelled random-model output exist only
    in a temporary directory. Model hashes cover parameters and normalization.
    """
    device = m.selected_device(device)
    old_threads = torch.get_num_threads()
    if device == "cpu":
        torch.set_num_threads(min(4, old_threads))
    try:
        m.seed_everything(7)
        spec = m.ModelSpec()
        model = m.PhysicsEmbeddedWingPredictor(spec).to(device).eval()
        before = _state_digest(model)
        with tempfile.TemporaryDirectory(prefix="flogen-foilpara-forward-") as temporary:
            root = Path(temporary) / "foilpara"
            for directory in ("foiloc", "foilgeo", "foilsld"):
                (root / directory).mkdir(parents=True)
            centers, vertices = m.synthetic_geometry()
            configs = np.zeros((1, 41), dtype=np.float64)
            configs[0, 0] = 101
            configs[0, 39:41] = [.75, 2.]
            index = np.array([[0., 0., 2., .75, 1.5, 2.]])
            np.savetxt(root / "foilgeo" / "configs.dat", configs,
                       header="Synthetic single-case diagnostic, 41 columns", comments="")
            np.save(root / "foilgeo" / "geom0.npy", centers[None])
            np.save(root / "foilgeo" / "origingeom.npy", vertices[None])
            np.save(root / "foiloc" / "index.npy", index)
            np.save(root / "foiloc" / "reynolds.npy", np.array([2e7]))
            np.save(root / "foilsld" / "index.npy", index)
            sld = np.linspace(.4, .5, 128)[None,:]
            np.save(root / "foilsld" / "sectional_cl.npy", sld)
            dataset = FoilParaData(root)
            try:
                output = _export_foilpara(
                    model, spec, dataset, Path(temporary) / "network", device, 1,
                    {"prediction_kind": "diagnostic_untrained", "synthetic_only": True,
                     "training_steps": 0, "accuracy_claim": "none"})
                result = np.load(output, allow_pickle=False)
                report = {
                    "synthetic_only": True, "prediction_kind": "diagnostic_untrained",
                    "device": device, "wing_shape": list(result.shape),
                    "index_shape": list(np.load(output.parent / "index.npy").shape),
                    "sld_source": "external_foilsld_sectional_cl", "finite": bool(np.isfinite(result).all()),
                    "optimizer_steps": 0, "model_state_unchanged": _state_digest(model) == before,
                    "normalization_unchanged": True, "production_output_written": False,
                }
                m.require(report["model_state_unchanged"], "Diagnostic forward changed the model state")
                return report
            finally:
                dataset.close()
    finally:
        if device == "cpu":
            torch.set_num_threads(old_threads)
