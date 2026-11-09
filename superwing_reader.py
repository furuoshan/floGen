"""Read native SuperWing samples one at a time as the single-case 41-column format.

Only the requested surface field and geometry are copied. The native arrays stay
read-only memory maps; log records are scanned lazily and held in a bounded LRU.
SLD obtained from CFD is explicitly labelled reference data, never an independent
estimate suitable for blind prediction or model-accuracy validation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Iterator

import numpy as np

import main as m


GEOMETRY_FIELDS = (
    "original_id", "SA", "DA", "DA_kink", "AR", "TR", "kink", "rootadj",
    "rtcs_0", "rtcs_1", "rtcs_2", "cambers_0", "cambers_1", "cambers_2",
    "twists_0", "twists_1", "twists_2", "twists_3",
    *(f"upper_U{i}" for i in range(10)),
    *(f"lower_U{i}" for i in range(10)), "tcroot", "mach", "aoa_deg",
)
INDEX_FIELDS = ("shape_index", "condition_index", "aoa_deg", "mach", "ref_area", "half_span")
CFD_INDEX_FIELDS = INDEX_FIELDS + (
    "cl_solver", "cd_solver", "cm_solver", "cl_surface", "cd_surface", "cm_surface")


@dataclass
class SuperWingCase:
    """An independently owned single-case snapshot, retaining all source units."""

    configs: np.ndarray         # (1,41), original ID + 38 geometry parameters + Mach/AoA
    index: np.ndarray           # (1,6), local shape and condition IDs both zero
    cfd_index: np.ndarray       # (1,12), source CFD coefficients preserved
    centers: np.ndarray         # (1,3,128,256)
    vertices: np.ndarray        # (1,3,129,257)
    field: np.ndarray           # (1,3,128,256): Cp,150*Cf_tau,300*Cf_z
    reynolds: np.ndarray        # (1,), actual Re
    sectional_cl: np.ndarray    # (1,128), before sweep-theory correction
    metadata: dict


def _integer(value, name):
    m.require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)),
              f"{name} must be an integer, not bool or a floating-point value")
    return int(value)


def _actual_reynolds(value):
    m.require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
              "Actual Reynolds number must be numeric")
    value = float(value)
    m.require(math.isfinite(value) and value > 0, "Actual Reynolds number must be finite and positive")
    return value


def _copy64(values):
    return np.array(values, dtype=np.float64, copy=True, order="C")


class SuperWingReader:
    """Context-managed native reader, with explicit random and lazy sequential access.

    ``read(n)`` returns one case using a geometry-only low-fidelity VLM estimate.
    ``read(n, sld=...)`` instead accepts one externally supplied CL distribution.
    No source arrays or logs are changed, and no optimizer/model training is used.
    """

    def __init__(self, root="D:/Superwing", reynolds=None, reynolds_file=None, log_cache_size=16):
        m.require(not (reynolds is not None and reynolds_file is not None),
                  "Provide an actual-Reynolds scalar OR a source-row-order file")
        self.root = Path(root).resolve()
        self.closed = False
        self.data = None
        self.reynolds_array = None
        self._log_handle = None
        self._log_scan_offset = 0
        self._log_offsets = {}  # Small byte offsets only, not full geometry records.
        self._log_records = OrderedDict()
        self.log_cache_size = _integer(log_cache_size, "log_cache_size")
        m.require(self.log_cache_size > 0, "log_cache_size must be positive")
        self.reynolds_override = _actual_reynolds(reynolds) if reynolds is not None else None
        self.reynolds_path = Path(reynolds_file).resolve() if reynolds_file is not None else None
        try:
            self.data = m.SuperWingData(
                self.root, target="field", require_reynolds=False,
                config_columns=(55,), index_columns=(12,))
            if self.reynolds_path is not None:
                self.reynolds_array = m.read_npy(self.reynolds_path)
                m.require(self.reynolds_array.shape == (len(self.data),),
                          "Reynolds file must have shape (N,) in native source sample order")
            elif self.reynolds_override is None:
                m.require((self.root / "running_info.jsonl").is_file(),
                          "Actual Reynolds unavailable: supply reynolds/reynolds_file or native running_info.jsonl")
        except BaseException:
            self.close()
            raise

    def _ensure_open(self):
        m.require(not self.closed, "SuperWingReader has been closed")

    def __len__(self):
        self._ensure_open()
        return len(self.data)

    def _cache_record(self, original_id, record):
        self._log_records[original_id] = record
        self._log_records.move_to_end(original_id)
        while len(self._log_records) > self.log_cache_size:
            self._log_records.popitem(last=False)
        return record

    @staticmethod
    def _parse_log_line(line, offset):
        try:
            record = json.loads(line.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid native Reynolds log at byte offset {offset}") from error
        m.require(isinstance(record, dict), f"Reynolds log must contain objects at byte offset {offset}")
        original = record.get("running_id")
        m.require(isinstance(original, Integral) and not isinstance(original, bool) and original >= 0,
                  f"Invalid running_id in Reynolds log at byte offset {offset}")
        return int(original), record

    def _log_record(self, original_id):
        if original_id in self._log_records:
            self._log_records.move_to_end(original_id)
            return self._log_records[original_id]
        if self._log_handle is None:
            self._log_handle = (self.root / "running_info.jsonl").open("rb")
        stream = self._log_handle
        if original_id in self._log_offsets:
            offset = self._log_offsets[original_id]
            stream.seek(offset)
            found, record = self._parse_log_line(stream.readline(), offset)
            m.require(found == original_id, "Native Reynolds log changed during reading")
            return self._cache_record(original_id, record)
        # Continue the previous scan. Sequential dataset traversal reads log lines
        # once, while random access can seek to an already discovered byte offset.
        stream.seek(self._log_scan_offset)
        while True:
            offset = stream.tell()
            line = stream.readline()
            self._log_scan_offset = stream.tell()
            if not line:
                raise ValueError(f"Reynolds log missing original_id={original_id}")
            if not line.strip():
                continue
            found, record = self._parse_log_line(line, offset)
            m.require(found not in self._log_offsets, f"Duplicate running_id in Reynolds log: {found}")
            self._log_offsets[found] = offset
            if found == original_id:
                return self._cache_record(original_id, record)

    def _reynolds_at(self, row, original_id, condition_id):
        if self.reynolds_override is not None:
            return self.reynolds_override, {"kind": "explicit_scalar", "unit": "actual_reynolds"}
        if self.reynolds_array is not None:
            return _actual_reynolds(self.reynolds_array[row]), {
                "kind": "explicit_file", "path": str(self.reynolds_path),
                "source_row": row, "unit": "actual_reynolds"}
        record = self._log_record(original_id)
        key = f"{condition_id}_running_info"
        try:
            value = record[key]["working_conditions"]["re"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Reynolds missing for original_id={original_id}, condition={condition_id}") from error
        return _actual_reynolds(value), {
            "kind": "native_running_info", "path": str(self.root / "running_info.jsonl"),
            "running_id": original_id, "condition_id": condition_id, "unit": "actual_reynolds"}

    def read(self, row, sld=None, sld_mode="vlm"):
        """Copy exactly one sample, converting only metadata layout and local IDs.

        CFD channel values, signs, multipliers and geometry coordinates are copied
        as float64. The default SLD depends on geometry and conditions only.
        CFD integration is available solely through explicit cfd-reference mode.
        """
        self._ensure_open()
        m.require(sld_mode in ("vlm", "cfd-reference"), "Unknown SLD mode")
        row = _integer(row, "row")
        m.require(0 <= row < len(self.data), f"row must be in [0,{len(self.data)-1}]")
        native = self.data
        g = int(native.shape_ids[row])
        k = int(native.condition_ids[row])
        original_id = int(native.original_ids[g])
        source_index = _copy64(native.index[row:row+1])
        m.require(bool((source_index[0,4:6] > 0).all()), "Reference half-area and half-span must be positive")
        m.require(source_index[0,3] > 0, "Mach must be positive")
        centers = _copy64(native.centers[g:g+1])
        vertices = _copy64(native.vertices[g:g+1])
        field = _copy64(native.data[row:row+1])
        m.finite(field, f"SuperWing field row {row}")
        features = m.geometry_features(centers[0], vertices[0])
        reynolds, re_source = self._reynolds_at(row, original_id, k)
        configs = _copy64(np.concatenate((native.configs[g,:39], source_index[0,[3,2]]))[None,:])
        cfd_index = source_index.copy()
        cfd_index[0,:2] = 0  # The destination contains only this geometry/condition.
        conditions_index = _copy64(cfd_index[:,:6])
        sld_details = None
        if sld is None and sld_mode == "vlm":
            from low_fidelity_sld import estimate_spanwise_lift
            sectional_cl, sld_details = estimate_spanwise_lift(
                centers[0], vertices[0], float(source_index[0,3]), float(source_index[0,2]))
            sectional_cl = _copy64(sectional_cl)
            sld_source = "pytornado_vlm"
        elif sld is None:
            sectional_cl = _copy64(m.sectional_lift(
                vertices[0], field[0], features["chord"], source_index[0,2])[None,:])
            sld_source = "cfd_reference_integral"
        else:
            values = np.asarray(sld)
            m.require(values.dtype.kind in "fiu", "External SLD must be a numeric array")
            m.require(values.shape in ((128,), (1,128)), "External SLD must have shape (128,) or (1,128)")
            m.finite(values, "external sectional CL")
            sectional_cl = _copy64(values.reshape(1,128))
            sld_source = "external_provided"
        metadata = {
            "schema_version": "superwing-single-case-v1",
            "source_root": str(self.root), "source_row": row,
            "source_geometry_index": g, "source_condition_id": k,
            "original_id": original_id, "shape_index": 0, "condition_index": 0,
            "sample_count": 1, "geometry_count": 1,
            "mach": float(source_index[0,3]), "aoa_deg": float(source_index[0,2]),
            "actual_Re": reynolds, "reynolds_source": re_source,
            "sld_source": sld_source, "sld_requires_cfd": sld_source == "cfd_reference_integral",
            "sld_usage": ("reference_diagnostic_only_not_blind_inference" if sld_source == "cfd_reference_integral"
                          else "geometry_only_low_fidelity_estimate" if sld_source == "pytornado_vlm"
                          else "caller_supplied_external_estimate"),
            "sld_details": sld_details,
            "field_source": "native_cfd_reference_not_network_prediction",
            "columns_and_units": {
                "configs": list(GEOMETRY_FIELDS), "index": list(INDEX_FIELDS),
                "cfd_index": list(CFD_INDEX_FIELDS),
                "field_channels": ["Cp", "150*Cf_tau", "300*Cf_z"],
                "stored_multipliers": [1,150,300],
                "angle": "degree", "reynolds": "actual_dimensionless_number",
                "sectional_cl": "dimensionless_before_sweep_transform",
                "geometry_axes": ["geometry", "xyz", "span", "contour"],
                "field_axes": ["sample", "channel", "span", "contour"],
                "coordinates": "native_model_units_reference_chord_1_not_implicitly_metres",
            },
        }
        return SuperWingCase(configs, conditions_index, cfd_index, centers, vertices, field,
                             np.array([reynolds], dtype=np.float64), sectional_cl, metadata)

    def iter_cases(self, start=0, stop=None, sld_mode="vlm") -> Iterator[SuperWingCase]:
        """Yield only the next requested case; no list or future fields are loaded."""
        self._ensure_open()
        start = _integer(start, "start")
        stop = len(self) if stop is None else _integer(stop, "stop")
        m.require(0 <= start <= stop <= len(self), "Require 0 <= start <= stop <= sample count")
        for row in range(start, stop):
            yield self.read(row, sld_mode=sld_mode)

    def close(self):
        """Release read-only mappings and log handles, including Windows locks."""
        if self.closed:
            return
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if self.reynolds_array is not None:
            self.reynolds_array._mmap.close()
        if self.data is not None:
            for name in ("index", "centers", "vertices", "data", "reynolds_array"):
                value = getattr(self.data, name, None)
                mapping = getattr(value, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            self.data._features.clear()
        self._log_records.clear()
        self.closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_):
        self.close()
