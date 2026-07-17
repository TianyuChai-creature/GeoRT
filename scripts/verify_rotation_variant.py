#!/usr/bin/env python3
"""Verify a local-motion target cloud only adds robot link rotations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def _array_report(base: np.ndarray, variant: np.ndarray) -> dict:
    report = {"base_dtype": str(base.dtype), "variant_dtype": str(variant.dtype), "base_shape": list(base.shape), "variant_shape": list(variant.shape)}
    if base.dtype != variant.dtype or base.shape != variant.shape:
        report.update({"array_equal": False, "max_abs_diff": None})
        return report
    if base.dtype == object and base.shape == ():
        left, right = base.item(), variant.item()
        left_keys, right_keys = sorted(left), sorted(right)
        children, values = {}, []
        equal = left_keys == right_keys
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                equal = False
                children[key] = {"missing": "base" if key not in left else "variant"}
                continue
            child = _array_report(np.asarray(left[key]), np.asarray(right[key]))
            children[key] = child
            equal = equal and child["array_equal"]
            if child["max_abs_diff"] is not None:
                values.append(child["max_abs_diff"])
        report.update({"object_dict_keys_base": left_keys, "object_dict_keys_variant": right_keys, "children": children, "array_equal": equal, "max_abs_diff": max(values, default=0.0)})
        return report
    max_abs = float(np.max(np.abs(base - variant))) if np.issubdtype(base.dtype, np.number) and base.size else (0.0 if np.issubdtype(base.dtype, np.number) else None)
    report.update({"array_equal": bool(np.array_equal(base, variant)), "max_abs_diff": max_abs})
    return report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_rotation_variant(base_path: Path, variant_path: Path) -> dict:
    with np.load(base_path, allow_pickle=True) as base, np.load(variant_path, allow_pickle=True) as variant:
        base_fields, variant_fields = set(base.files), set(variant.files)
        reports = {name: _array_report(base[name], variant[name]) for name in sorted(base_fields & variant_fields)}
    equivalent = all(item["array_equal"] for item in reports.values())
    numeric_diffs = [item["max_abs_diff"] for item in reports.values() if item["max_abs_diff"] is not None]
    return {"base": str(base_path), "base_sha256": sha256(base_path), "variant": str(variant_path), "variant_sha256": sha256(variant_path), "base_fields": sorted(base_fields), "variant_fields": sorted(variant_fields), "added_fields": sorted(variant_fields - base_fields), "removed_fields": sorted(base_fields - variant_fields), "shared_fields": reports, "shared_fields_equivalent": equivalent, "shared_max_abs_diff": max(numeric_diffs, default=0.0), "rotation_only_addition": equivalent and not (base_fields - variant_fields) and (variant_fields - base_fields) == {"link_rotation"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_rotation_variant(args.base, args.variant)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
