# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

# GeoRT Package Path Helper.
# No more pain configuring paths!
def get_package_root():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / "..").resolve()

def to_package_root(path):
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / "..").resolve() / path

def get_data_root():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / ".." / "data").resolve()

def get_checkpoint_root():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / ".." / "checkpoint").resolve()

def get_human_data_output_path(human_data):
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / ".." / "data" / human_data).resolve()

def get_human_data(name):
    data_root = Path(get_data_root())
    requested = Path(name)
    if requested.suffix == ".npy" and requested.is_file():
        return requested.resolve()

    exact_name = requested.name if requested.suffix == ".npy" else f"{name}.npy"
    exact_path = data_root / exact_name
    if exact_path.exists():
        return exact_path.resolve()

    stem = requested.stem if requested.suffix == ".npy" else str(name)
    partial_matches = sorted(
        path.name
        for path in data_root.glob("*.npy")
        if stem in path.stem
    )
    hint = f" Partial matches: {partial_matches}" if partial_matches else ""
    raise FileNotFoundError(f"No exact human dataset {exact_name!r} found in {data_root}.{hint}")


if __name__ == '__main__':
    print(get_package_root())
    print(get_human_data_output_path("human"))

