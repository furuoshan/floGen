"""Read one SuperWing sample and export the project's single-case file layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

import numpy as np

import main as m
from foilpara import predict_foilpara
from superwing_reader import SuperWingCase, SuperWingReader


CONFIG_FIELDS = (
    ["original_id", "SA", "DA", "DA_kink", "AR", "TR", "kink", "rootadj"]
    + [f"rtcs_{i}" for i in range(3)] + [f"cambers_{i}" for i in range(3)]
    + [f"twists_{i}" for i in range(4)] + [f"upper_U{i}" for i in range(10)]
    + [f"lower_U{i}" for i in range(10)] + ["tcroot", "mach", "aoa_deg"]
)
LEAVES = ("foilpara/foilgeo", "foilpara/foiloc", "foilpara/foilsld", "foilperf/cfd")


def _validate_case(case):
    expected = {"configs": (1, 41), "index": (1, 6), "cfd_index": (1, 12),
                "centers": (1, 3, 128, 256), "vertices": (1, 3, 129, 257),
                "field": (1, 3, 128, 256), "reynolds": (1,), "sectional_cl": (1, 128)}
    for name, shape in expected.items():
        values = np.asarray(getattr(case, name))
        m.require(values.shape == shape, f"Single-case {name} must have shape {shape}")
        m.require(values.dtype.kind in "fiu", f"Non-numeric {name}")
        m.finite(values, name)
    m.require(bool((case.index[0, :2] == 0).all()), "Local shape/condition IDs must both be 0")
    m.require(np.array_equal(case.cfd_index[:, :6], case.index), "CFD and input sample identity differ")
    m.require(bool((case.reynolds > 0).all() and (case.index[:, 4:6] > 0).all()),
              "Reynolds, reference area and half-span must be positive")
    m.require(np.allclose(case.configs[0, [40, 39]], case.index[0, [2, 3]], rtol=0, atol=1e-8),
              "Single Mach/AoA does not match the condition index")
    m.require(isinstance(case.metadata, dict) and "source_root" in case.metadata,
              "Source provenance is required")
    m.require(float(case.configs[0, 0]) == case.metadata["original_id"], "Original geometry ID differs")
    m.geometry_features(case.centers[0], case.vertices[0])


def _preflight(output_root, source_root, leaves):
    m.protect_destination(output_root, [source_root])
    m.require(output_root not in source_root.parents, "Output cannot be an ancestor of source data")
    m.require(not output_root.exists() or output_root.is_dir(), "Output root must be a directory")
    for relative in leaves:
        target = output_root / relative
        m.require(output_root in target.resolve().parents,
                  f"Output leaf resolves outside the selected root: {target}")
        m.require(m.prediction_directory_available(target),
                  f"Case export refuses existing data; only format.md is allowed: {target}")


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_geometry(folder, case):
    np.savetxt(folder / "configs.dat", case.configs, fmt="%.17g",
               header="VARIABLES = " + " ".join(CONFIG_FIELDS), comments="")
    np.save(folder / "geom0.npy", np.array(case.centers, dtype=np.float64, order="C"), allow_pickle=False)
    np.save(folder / "origingeom.npy", np.array(case.vertices, dtype=np.float64, order="C"), allow_pickle=False)


def export_case(case: SuperWingCase, output_root, checkpoint=None, device="cpu"):
    """Publish one case, without overwrites; optionally run an actual trained model.

    Default SLD from SuperWingReader is the paper's low-fidelity PyTornado VLM
    estimate. Explicit cfd-reference mode is labelled separately and cannot be
    treated as a blind prediction benchmark. No network result is generated when
    checkpoint is absent.
    """
    _validate_case(case)
    output_root = Path(output_root).resolve()
    source_root = Path(case.metadata["source_root"]).resolve()
    leaves = list(LEAVES)
    model = spec = payload = None
    if checkpoint is not None:
        leaves.append("foilperf/network")
    _preflight(output_root, source_root, leaves)
    if checkpoint is not None:
        model, spec, payload = m.load_checkpoint(checkpoint, "wing")
        device = m.selected_device(device)
    export_id = uuid.uuid4().hex
    provenance = dict(case.metadata, export_id=export_id, status="complete",
                      local_shape_index=0, local_condition_id=0,
                      configs_columns=CONFIG_FIELDS, geometry_count=1, sample_count=1,
                      data_origin="existing_superwing_cfd", cfd_computed_in_this_export=False,
                      network_prediction_generated=checkpoint is not None,
                      reference_sld_uses_cfd=case.metadata.get("sld_source") == "cfd_reference_integral")
    with tempfile.TemporaryDirectory(prefix="flogen-single-case-") as temporary:
        stage = Path(temporary)
        for relative in leaves:
            (stage / relative).mkdir(parents=True)
        geo, oc, sld, cfd = [stage / relative for relative in LEAVES]
        _write_geometry(geo, case)
        _write_geometry(cfd, case)
        np.save(oc / "index.npy", case.index.astype(np.float64), allow_pickle=False)
        np.save(oc / "reynolds.npy", case.reynolds.astype(np.float64), allow_pickle=False)
        np.save(sld / "index.npy", case.index.astype(np.float64), allow_pickle=False)
        np.save(sld / "sectional_cl.npy", case.sectional_cl.astype(np.float64), allow_pickle=False)
        np.save(cfd / "index.npy", case.cfd_index.astype(np.float64), allow_pickle=False)
        np.save(cfd / "data.npy", case.field.astype(np.float64), allow_pickle=False)
        # The single exported condition is always ID 0. Preserve its original
        # source condition ID independently in source_case.json.
        log = {"running_id": int(case.metadata["original_id"]),
               "0_running_info": {"working_conditions": {"re": float(case.reynolds[0])}}}
        (cfd / "running_info.jsonl").write_text(json.dumps(log, allow_nan=False) + "\n", encoding="utf-8")
        if checkpoint is not None:
            predict_foilpara(model, spec, payload, stage / "foilpara", stage / "foilperf/network", device)
            info_path = stage / "foilperf/network/prediction_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info.update(source_root=str(output_root / "foilpara"),
                        input_prepared_in_temporary_directory=True,
                        source_case="source_case.json", export_id=export_id,
                        reference_sld_uses_cfd=provenance["reference_sld_uses_cfd"])
            _write_json(info_path, info)
        for relative in leaves:
            record = dict(provenance, output_role=relative)
            _write_json(stage / relative / "source_case.json", record)
        # Recheck all destinations after preparation, before writing any output.
        _preflight(output_root, source_root, leaves)
        for relative in leaves:
            target = output_root / relative
            target.mkdir(parents=True, exist_ok=True)
            for source in (stage / relative).iterdir():
                if source.name == "source_case.json":
                    continue
                with source.open("rb") as incoming, (target / source.name).open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
        # Completion/source records are published only after all arrays exist.
        for relative in leaves:
            with (output_root / relative / "source_case.json").open("xb") as handle:
                handle.write((stage / relative / "source_case.json").read_bytes())
    return {"status": "complete", "export_id": export_id, "source_row": case.metadata["source_row"],
            "original_id": case.metadata["original_id"], "output_root": str(output_root),
            "foilpara_root": str(output_root / "foilpara"), "cfd_root": str(output_root / "foilperf/cfd"),
            "network_root": str(output_root / "foilperf/network") if checkpoint is not None else None,
            "sld_source": case.metadata.get("sld_source"), "configs_shape": [1, 41],
            "network_prediction_generated": checkpoint is not None, "source_files_modified": False}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "export"):
        p = commands.add_parser(command)
        p.add_argument("--source-root", type=Path, default=Path("D:/Superwing"))
        p.add_argument("--row", type=int, required=True, help="Source index/data row, not geometry ID")
        re = p.add_mutually_exclusive_group()
        re.add_argument("--reynolds", type=float)
        re.add_argument("--reynolds-file", type=Path)
        p.add_argument("--sld-file", type=Path, help="One external sectional-CL array, (128,) or (1,128)")
        p.add_argument("--sld-mode", choices=("vlm", "cfd-reference"), default="vlm",
                       help="Default: paper's PyTornado VLM; CFD integration must be explicit")
        if command == "export":
            p.add_argument("--output-root", type=Path, default=Path("D:/data"))
            p.add_argument("--checkpoint", type=Path, help="Optional trained wing checkpoint for actual prediction")
            p.add_argument("--device", default="cpu")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    sld = np.load(args.sld_file, allow_pickle=False) if args.sld_file is not None else None
    with SuperWingReader(args.source_root, reynolds=args.reynolds, reynolds_file=args.reynolds_file) as reader:
        case = reader.read(args.row, sld=sld, sld_mode=args.sld_mode)
        if args.command == "inspect":
            report = {"source_sample_count": len(reader), "source": case.metadata,
                      "shapes": {name: list(getattr(case, name).shape) for name in
                                 ("configs", "index", "cfd_index", "centers", "vertices", "field",
                                  "reynolds", "sectional_cl")}}
        else:
            report = export_case(case, args.output_root, args.checkpoint, args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, IndexError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
