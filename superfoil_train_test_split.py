"""Deterministic, disjoint baseline-CST train/test split for native SuperFoil.

The three VT subsets are addressed by their existing concatenated row numbers.
Original arrays are never changed. Exact 9:1 sample counts exclude at most nine
seeded training indices to make the selected sample count divisible by ten,
rather than splitting a family or making artificial copies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_META = Path(__file__).resolve().parent / "build" / "foil_meta_for_condition_audit.npz"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def family_ids(dataset, meta_path=DEFAULT_META):
    """Bind all Running_IDs to one native metadata CST family, including aliases."""
    with np.load(meta_path, allow_pickle=False) as meta:
        running = np.asarray(meta["run_ids"], dtype=np.int64)
        cst = np.asarray(meta["cst"], dtype=np.float64)
    require(cst.shape == (len(running), 20), "Expected native metadata with 20 CST coefficients")
    require(np.isfinite(cst).all() and len(np.unique(running)) == len(running), "Invalid CST metadata IDs")
    unique, inverse = np.unique(cst, axis=0, return_inverse=True)
    binding = {int(run): int(family) for run, family in zip(running, inverse)}
    native_ids = np.asarray(dataset.groups)
    require(np.isfinite(native_ids).all() and (native_ids == native_ids.astype(np.int64)).all(),
            "Dataset Running_IDs must be finite integers")
    missing = set(map(int, np.unique(native_ids))) - binding.keys()
    require(not missing, f"Native metadata lacks Running_IDs: {sorted(missing)[:8]}")
    groups = np.asarray([binding[int(run)] for run in native_ids], dtype=np.int64)
    present = np.unique(groups)
    # Dense IDs make selection independent of unused rows in a synthetic fixture.
    dense = np.searchsorted(present, groups)
    aliases = []
    for family in present:
        ids = running[inverse == family]
        if len(ids) > 1:
            aliases.append({"family_id": int(np.searchsorted(present, family)),
                            "running_ids": ids.astype(int).tolist()})
    return dense, {"method": "exact equality of 20 float64 CST coefficients in one native metadata file",
                   "source_meta": str(Path(meta_path).resolve()),
                   "metadata_sha256": hashlib.sha256(Path(meta_path).read_bytes()).hexdigest(),
                   "native_metadata_rows": len(running), "present_cst_families": len(present),
                   "present_running_ids": len(np.unique(native_ids)), "aliased_running_ids": aliases}


def choose_test_families(weights, target, seed=42, test_family_count=None):
    """Use a seeded family sample, then exact equal-count swaps if possible."""
    weights = np.asarray(weights, dtype=np.int64)
    require(weights.ndim == 1 and (weights > 0).all(), "Family weights must be positive")
    n = len(weights)
    wanted = max(1, int(round(n / 10))) if test_family_count is None else int(test_family_count)
    wanted = min(wanted, n-1)
    require(n >= 2 and 0 < target < int(weights.sum()), "Nonempty train and test populations required")
    order = np.random.default_rng(seed).permutation(n)
    test, train = order[:wanted].copy(), order[wanted:].copy()
    initial = int(weights[test].sum())
    delta = int(target)-initial
    swaps = []
    if delta:
        for a in train:
            hits = test[weights[a]-weights[test] == delta]
            if len(hits):
                swaps = [(int(a), int(hits[0]))]
                break
    if delta and not swaps:
        # At most N_train*N_test candidate pairs (181k for the real database).
        by_difference = {}
        for a in train:
            for b in test:
                by_difference.setdefault(int(weights[a]-weights[b]), []).append((int(a),int(b)))
        for a in train:
            for b in test:
                complement = delta-int(weights[a]-weights[b])
                second = next(((c,d) for c,d in by_difference.get(complement, ()) if c != a and d != b), None)
                if second:
                    swaps = [(int(a),int(b)),second]
                    break
            if swaps:
                break
    if delta and not swaps:
        # A seeded subset-sum fallback still preserves whole families, although
        # its family count need not be exactly 10% when family sizes are uneven.
        parent = {0: None}
        for family in order:
            for total in list(parent)[::-1]:
                nxt = total+int(weights[family])
                if nxt <= target and nxt not in parent:
                    parent[nxt] = (total,int(family))
            if target in parent:
                break
        require(target in parent, "Cannot obtain the exact test count without splitting a CST family")
        found, remaining = [],target
        while remaining:
            remaining,family = parent[remaining]
            found.append(family)
        test = np.asarray(found,dtype=np.int64)
        method = "seeded whole-family subset sum"
    else:
        for added,removed in swaps:
            test[test == removed] = added
        method = "seeded 10-percent family sample with at most two equal-count family swaps"
    require(int(weights[test].sum()) == target, "Exact test-family weight check failed")
    require(len(np.unique(test)) == len(test), "Repeated selected test family")
    return np.sort(test), {"method":method,"seed":seed,"initial_test_samples":initial,
                          "target_test_samples":int(target),"test_family_count":len(test),
                          "family_swaps":[{"add_to_test":a,"remove_from_test":b,
                                           "added_samples":int(weights[a]),"removed_samples":int(weights[b])}
                                          for a,b in swaps]}


def excluded_training_indices(dataset, candidates, groups, needed, seed):
    """Select only training indices, retaining at least one row of each family."""
    if not needed:
        return np.empty(0,dtype=np.int64),[]
    remaining=np.bincount(groups[candidates],minlength=int(groups.max())+1)
    excluded,details=[],[]
    for row in np.random.default_rng(seed).permutation(candidates):
        row=int(row);family=int(groups[row])
        if remaining[family] <= 1:
            continue
        position=int(np.searchsorted(dataset.offsets,row,side="right")-1)
        excluded.append(row);remaining[family]-=1
        details.append({"excluded_joint_row":row,"excluded_subset":dataset.datasets[position].subset,
                        "excluded_subset_row":row-int(dataset.offsets[position]),"family_id":family,
                        "reason":"Source sample count is not divisible by ten; this seeded training index is unused only to make the selected sample ratio exactly 9:1. The original source record is not deleted.",
                        "redundancy_claimed":False})
        if len(excluded)==needed:
            return np.sort(np.asarray(excluded,dtype=np.int64)),details
    raise ValueError("Cannot exclude the remainder without emptying a training CST family")


def build_split(dataset, meta_path=DEFAULT_META, seed=42):
    n = len(dataset)
    target_test = n//10
    target_train = 9*target_test
    remainder = n-target_train-target_test
    require(target_test > 0, "At least ten records are required")
    groups,family_manifest = family_ids(dataset,meta_path)
    weights = np.bincount(groups)
    test_families,selection = choose_test_families(weights,target_test,seed)
    is_test = np.isin(groups,test_families)
    test = np.flatnonzero(is_test)
    initial_train = np.flatnonzero(~is_test)
    excluded,exclusion_details = excluded_training_indices(dataset,initial_train,groups,remainder,seed+1)
    train = np.setdiff1d(initial_train,excluded,assume_unique=True)
    require(len(train)==target_train and len(test)==target_test and len(train)==9*len(test), "Train/test ratio is not exact")
    require(len(np.intersect1d(groups[train],groups[test]))==0, "CST family leakage")
    all_rows = np.r_[train,test,excluded]
    require(len(all_rows)==n and np.array_equal(np.sort(all_rows),np.arange(n)), "Rows must partition the full source exactly")
    subset_counts = {name:dataset.counts(rows) for name,rows in (("train",train),("test",test),("excluded",excluded))}
    manifest = {"source_samples":n,"selected_samples":len(train)+len(test),
        "counts":{"train":len(train),"test":len(test),"excluded":len(excluded)},
        "exact_train_test_ratio":"9:1","seed":seed,"family_mapping":family_manifest,
        "selection":selection,"subset_counts":subset_counts,
        "family_counts":{"train":len(np.unique(groups[train])),"test":len(test_families)},
        "family_overlap":0,"exclusion_seed":seed+1,"excluded_record_details":exclusion_details,
        "source_databases_modified":False,"training_or_normalization_uses_test":False,
        "old_official_split_policy":"User explicitly requested a new full-library 9:1 train/test split; older official/protected/validation assignments are not used for this new experiment.",
        "duplicate_policy":"No deduplication is claimed or performed. All shared-CST aliases stay in one set; only the listed seeded training indices are unused to make the sample count divisible by ten. No test row is dropped."}
    return {"train_rows":train,"test_rows":test,"excluded_rows":excluded,
            "family_ids":groups,"test_family_ids":test_families,
            "counts":manifest["counts"],"subset_counts":subset_counts,"manifest":manifest}


def save_split(result, destination):
    destination=Path(destination)
    targets=[destination/(key+".npy") for key in ("train_rows","test_rows","excluded_rows","family_ids","test_family_ids")]
    targets.append(destination/"manifest.json")
    require(not any(p.exists() for p in targets),"Split destination already contains a target file")
    destination.mkdir(parents=True,exist_ok=True)
    for key in ("train_rows","test_rows","excluded_rows","family_ids","test_family_ids"):
        path=destination/(key+".npy")
        require(not path.exists(),f"Split file already exists: {path}")
        np.save(path,result[key],allow_pickle=False)
    (destination/"manifest.json").write_text(json.dumps(result["manifest"],ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    from train_2d_once import JointSuperFoil
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foil-root",type=Path,default=Path("D:/SuperFoil"))
    parser.add_argument("--meta-path",type=Path,default=DEFAULT_META)
    parser.add_argument("--review-output",type=Path,default=Path("build/superfoil_new_split_review.json"))
    args=parser.parse_args()
    result=build_split(JointSuperFoil(args.foil_root),args.meta_path)
    args.review_output.parent.mkdir(parents=True,exist_ok=True)
    args.review_output.write_text(json.dumps(result["manifest"],ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result["manifest"],ensure_ascii=False,indent=2,allow_nan=False))
