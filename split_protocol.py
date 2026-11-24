"""Create and verify a shared train/validation/test split for paired curve CSVs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


VALIDATION_SPLIT = "validation"
METRICS = ("R2", "MAE", "RMSE")
TIME_METRICS = (*METRICS, "MAPE(%)")
CONC_METRICS = (*METRICS, "R2_C1", "R2_C2")
SUBSETS = ("train", "validation", "test")


def read_dataset(x_csv, y_csv):
    x = pd.read_csv(x_csv, header=None, float_precision="round_trip").to_numpy(dtype=float)
    y = pd.read_csv(y_csv, header=None, float_precision="round_trip").to_numpy(dtype=float)
    if x.shape[1] != 24 or y.shape != (len(x), 300):
        raise ValueError("Expected aligned (N, 24) inputs and (N, 300) curve targets.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Dataset contains non-finite values.")
    return x, y


def dataset_id(x, y):
    digest = hashlib.sha256()
    for array in (x, y):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def isotherm_groups(x):
    """Group identical binary LF curves, independent of site and component order."""
    keys = {}
    groups = []
    for row in x:
        components = [tuple(sorted((tuple(row[start:start + 3]), tuple(row[start + 3:start + 6]))))
                      for start in (9, 18)]
        key = tuple(sorted(components))
        groups.append(keys.setdefault(key, len(keys)))
    return np.asarray(groups)


def create_split(x, y, mode, seed, groups_csv=None):
    if mode == "system":
        if groups_csv is None:
            raise ValueError("System mode requires a one-column --groups-csv aligned to the cleaned rows.")
        labels = pd.read_csv(groups_csv, header=None, dtype=str, keep_default_na=False,
                             skip_blank_lines=False)
        if labels.shape != (len(x), 1) or labels.iloc[:, 0].str.strip().eq("").any():
            raise ValueError("Supply one nonempty system ID per cleaned data row, without a header.")
        _, groups = np.unique(labels.iloc[:, 0].str.strip(), return_inverse=True)
    elif groups_csv is not None:
        raise ValueError("--groups-csv applies to system mode only.")
    elif mode == "isotherm":
        groups = isotherm_groups(x)
        if len(np.unique(groups)) == len(x):
            warnings.warn("Every LF pair is unique. This split is equivalent in scope to a random row holdout.",
                          stacklevel=2)
    elif mode == "sample":
        groups = np.arange(len(x))
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    shuffled = np.random.default_rng(seed).permutation(np.unique(groups))
    n_test = math.ceil(0.20 * len(shuffled))
    n_val = math.ceil(0.15 * (len(shuffled) - n_test))
    partition = dict(zip(SUBSETS, (shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val], shuffled[:n_test])))
    indices = {part: np.flatnonzero(np.isin(groups, selected)).tolist() for part, selected in partition.items()}
    if any(len(rows) < 2 for rows in indices.values()):
        raise ValueError("Each subset needs at least two rows. Supply more data or independent groups.")
    return {
        "schema_version": 1,
        "dataset_id": dataset_id(x, y),
        "mode": mode,
        "seed": seed,
        "group_counts": {part: len(selected) for part, selected in partition.items()},
        "indices": indices,
    }


def split_id(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


def load_split(path, x, y):
    with open(path, encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest.get("schema_version") != 1 or manifest.get("dataset_id") != dataset_id(x, y):
        raise ValueError("Split manifest does not match these cleaned CSVs and their row order.")
    subsets = manifest["indices"]
    if set(subsets) != set(SUBSETS) or any(len(rows) < 2 for rows in subsets.values()):
        raise ValueError("Split must have nonempty train, validation, and test subsets with at least two rows each.")
    combined = [index for rows in subsets.values() for index in rows]
    if any(type(index) is not int for index in combined) or sorted(combined) != list(range(len(x))):
        raise ValueError("Split must assign every row exactly once.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-csv", default="curve_dataset_clean.csv")
    parser.add_argument("--y-csv", default="curve_clean.csv")
    parser.add_argument("--mode", required=True, choices=("sample", "isotherm", "system"))
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("split_manifest.json"))
    args = parser.parse_args()
    x, y = read_dataset(args.x_csv, args.y_csv)
    manifest = create_split(x, y, args.mode, args.seed, args.groups_csv)
    with args.out.open("x", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2)
    for subset, rows in manifest["indices"].items():
        print(f"{subset}: {len(rows)} rows / {manifest['group_counts'][subset]} groups")
    print(f"Split ID: {split_id(manifest)}")


if __name__ == "__main__":
    main()
