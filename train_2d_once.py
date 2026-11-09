"""Train the paper's 2-D conditional U-Net for exactly one pass over VT records."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import main as m

SUBSETS = ("VT-BUF", "VT-DD", "VT-GU")
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}


class JointSuperFoil(Dataset):
    """Share VT Running_ID families across all subsets, including GU variants."""
    def __init__(self, root=m.FOIL_ROOT, subsets=SUBSETS):
        self.root = Path(root).resolve()
        m.require(bool(subsets) and len(set(subsets)) == len(subsets), "Select distinct VT subsets")
        self.datasets = [m.SuperFoilData(self.root, name, condition_dim=3) for name in subsets]
        self.offsets = np.r_[0, np.cumsum([len(d) for d in self.datasets])].tolist()
        self.groups = np.concatenate([d.groups for d in self.datasets])
        official = []
        test_ids = []
        self.audit = []
        for offset, data in zip(self.offsets, self.datasets):
            # Full streaming validation before an optimizer or checkpoint exists.
            for start in range(0, len(data), 512):
                m.finite(data.data[start:start+512], f"{data.subset} surface data")
            m.finite(data.condition_matrix(), f"{data.subset} conditions")
            train = data.official_train()
            test = np.setdiff1d(np.arange(len(data)), train)
            m.require(not np.intersect1d(data.groups[train], data.groups[test]).size,
                      f"{data.subset} official split mixes a geometry across train/test")
            official.append(train+offset)
            test_ids.extend(np.unique(data.groups[test]).tolist())
            self.audit.append({"subset": data.subset, "samples": len(data),
                               "official_train": len(train), "official_test": len(test),
                               "all_finite": True, "data_shape": list(data.data.shape),
                               "reynolds_input": data.reynolds_info})
        self.protected_test_ids = np.unique(test_ids)
        self.official_rows = np.concatenate(official)
        self.candidate_rows = self.official_rows[
            ~np.isin(self.groups[self.official_rows], self.protected_test_ids)]

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, row):
        position = bisect_right(self.offsets, int(row))-1
        data = self.datasets[position]
        result = data[int(row)-self.offsets[position]]
        result["row"] = int(row)
        result["subset"] = data.subset
        return result

    def split(self, fraction=.1, seed=42):
        train, val = m.group_split(self.candidate_rows, self.groups, fraction, seed)
        m.require(not np.intersect1d(self.groups[train], self.groups[val]).size, "Validation shape leakage")
        m.require(not np.intersect1d(self.groups[np.r_[train, val]], self.protected_test_ids).size,
                  "Protected official-test family leaked into training/validation")
        return train, val

    def counts(self, rows):
        return {d.subset: int(((rows >= self.offsets[i]) & (rows < self.offsets[i+1])).sum())
                for i, d in enumerate(self.datasets)}


def write_json(path, value):
    temporary = path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_latest(output_dir, model, spec, training, source_root):
    """Publish a verified checkpoint before deleting older in-directory weights."""
    output_dir = Path(output_dir).resolve()
    m.protect_destination(output_dir, [source_root])
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / (".latest-"+uuid.uuid4().hex+".partial")
    m.save_checkpoint(temporary, "airfoil", model, spec, training, [source_root])
    loaded, _, payload = m.load_checkpoint(temporary, "airfoil")
    m.require(payload["training"]["training_steps"] == training["training_steps"], "Checkpoint step mismatch")
    m.require(all(bool(torch.isfinite(p).all()) for p in loaded.parameters()), "Non-finite checkpoint")
    del loaded, payload
    latest = output_dir / "latest.pt"
    temporary.replace(latest)
    # User explicitly requested keeping only the newest 2-D weight file.
    for old in output_dir.iterdir():
        if old == latest or old.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        m.require(old.resolve().parent == output_dir and not old.is_symlink(),
                  f"Refusing to remove a weight outside the selected folder: {old}")
        if old.is_file():
            old.unlink()
    return latest


@torch.inference_mode()
def evaluate(model, dataset, rows, batch_size, device):
    model.eval()
    normalized_sse = 0.
    raw_sse = np.zeros(2, dtype=float)
    points = 0
    for batch in DataLoader(Subset(dataset, rows.tolist()), batch_size=batch_size, shuffle=False):
        batch = m.move_batch(batch, device)
        prediction = model(batch["geometry"], batch["conditions"])
        error = prediction-batch["target"]
        normalized = m.normalize_target(model, prediction, "airfoil")-m.normalize_target(model, batch["target"], "airfoil")
        m.require(bool(torch.isfinite(normalized).all()), "Non-finite validation prediction")
        normalized_sse += float(normalized.square().sum())
        raw_sse += error.double().square().sum(dim=(0, 2)).cpu().numpy()
        points += len(prediction)*401
    return {"normalized_mse": normalized_sse/(2*points),
            "cp_rmse": float(math.sqrt(raw_sse[0]/points)),
            "cf_times_1000_rmse": float(math.sqrt(raw_sse[1]/points))}


def run(root=m.FOIL_ROOT, output_dir=Path("weights/2d"), device="auto", subsets=SUBSETS,
        batch_size=16, seed=42, validation_fraction=.1, checkpoint_every=100):
    m.require(batch_size > 0 and checkpoint_every > 0, "Positive batch/checkpoint interval required")
    device = m.selected_device(device)
    m.seed_everything(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output_dir = Path(output_dir).resolve()
    root = Path(root).resolve()
    m.protect_destination(output_dir, [root])
    data = JointSuperFoil(root, subsets)
    train, val = data.split(validation_fraction, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = m.ModelSpec(condition_dim=3)
    model = m.AirfoilPredictor(spec).to(device)
    actual_lr = 1e-4*m.learning_rate_factor(0, "airfoil")
    info = {"run_id": uuid.uuid4().hex, "stage": "airfoil", "epochs_requested": 1,
            "epochs_completed": 0, "training_steps": 0, "samples_seen": 0,
            "training_samples": len(train), "validation_samples": len(val),
            "train_counts": data.counts(train), "validation_counts": data.counts(val),
            "source_samples": len(data), "source_root": str(root), "foil_subsets": list(subsets),
            "data_audit": data.audit, "seed": seed, "batch_size": batch_size,
            "device": device, "optimizer": "Adam", "loss": "MSE", "learning_rate": actual_lr,
            "normalization": m.NORMALIZATION, "loss_space": "minmax_normalized",
            "train_rows": train.tolist(), "validation_rows": val.tolist(),
            "train_shape_ids": np.unique(data.groups[train]).tolist(),
            "validation_shape_ids": np.unique(data.groups[val]).tolist(),
            "protected_official_test_shape_ids": data.protected_test_ids.tolist(),
            "validation_split": "VT-global Running_ID families; protect union of all official-test IDs",
            "reynolds_input": {"source": "user_confirmed_default", "value": m.FOIL_REYNOLDS,
                               "unit": "actual_reynolds"},
            "checkpoint_selection": "latest_optimizer_step_not_best_validation",
            "duplicate_policy": "each selected source record once; cross-subset near-duplicates retained",
            "paper_override": "one epoch, one initialization requested instead of 300 epochs/five initializations"}
    status = {"status": "fitting_normalization", "training_samples": len(train),
              "validation_samples": len(val), "device": device, "steps_expected": math.ceil(len(train)/batch_size)}
    write_json(output_dir / "status.json", status)
    print(json.dumps(status), flush=True)
    started = time.monotonic()
    m.fit_normalizers(model, data, train, "airfoil", device, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=actual_lr)
    loader = DataLoader(Subset(data, train.tolist()), batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=False,
                        generator=torch.Generator().manual_seed(seed), pin_memory=device.startswith("cuda"))
    visited = np.zeros(len(data), dtype=np.uint8)
    total = samples = 0
    log = output_dir / "train.log"
    with log.open("w", encoding="utf-8") as stream:
        model.train()
        for step, batch in enumerate(loader, 1):
            original_rows = batch["row"].numpy()
            m.require(not bool(visited[original_rows].any()), "Training sample was repeated")
            visited[original_rows] += 1
            batch = m.move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["geometry"], batch["conditions"])
            loss = torch.nn.functional.mse_loss(
                m.normalize_target(model, prediction, "airfoil"),
                m.normalize_target(model, batch["target"], "airfoil"))
            m.require(bool(torch.isfinite(loss)), "Non-finite training loss")
            loss.backward()
            optimizer.step()
            count = len(original_rows)
            samples += count
            total += float(loss.detach())*count
            info.update(training_steps=step, samples_seen=samples, train_mse=total/samples,
                        elapsed_seconds=time.monotonic()-started)
            if step == 1 or step % 50 == 0 or samples == len(train):
                status.update(status="training", step=step, samples_seen=samples,
                              training_fraction=samples/len(train), train_mse=total/samples,
                              elapsed_seconds=info["elapsed_seconds"])
                write_json(output_dir / "status.json", status)
                record = json.dumps(status)
                print(record, flush=True)
                stream.write(record+"\n"); stream.flush()
            if step == 1 or step % checkpoint_every == 0:
                save_latest(output_dir, model, spec, info.copy(), root)
        m.require(samples == len(train) and bool((visited[train] == 1).all()) and
                  int(visited.sum()) == len(train), "Single-pass training coverage mismatch")
        metrics = evaluate(model, data, val, batch_size, device)
        info.update(epochs_completed=1, every_selected_training_record_seen_exactly_once=True,
                    validation=metrics, elapsed_seconds=time.monotonic()-started)
        latest = save_latest(output_dir, model, spec, info, root)
        write_json(output_dir / "training_report.json", info)
        status.update(status="complete", epochs_completed=1, validation=metrics,
                      every_selected_training_record_seen_exactly_once=True,
                      checkpoint=str(latest), elapsed_seconds=info["elapsed_seconds"])
        write_json(output_dir / "status.json", status)
        print(json.dumps(status), flush=True)
        stream.write(json.dumps(status)+"\n")
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foil-root", type=Path, default=m.FOIL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("weights/2d"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    args = parser.parse_args()
    run(args.foil_root, args.output_dir, args.device, args.subsets)


if __name__ == "__main__":
    main()
