# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""Collect HTS frames as GeoRT-compatible human mocap data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np

from geort.utils.path import get_data_root
from geort.frame_convention import COORDINATE_CONVENTION

EXPECTED_HTS_LANDMARKS = 21
CAPTURE_COORDINATE_CONVENTION = COORDINATE_CONVENTION
HandSideName = Literal["left", "right"]


def normalize_hand_side(hand_side: str) -> HandSideName:
    side = hand_side.lower()
    if side not in ("left", "right"):
        raise ValueError(f"Expected hand_side to be 'left' or 'right', got {hand_side!r}")
    return side


def make_output_path(name: str, hand_side: str, data_dir: Path | str | None = None) -> Path:
    """Return ``data/<name>_<left|right>.npy`` for an HTS recording."""
    side = normalize_hand_side(hand_side)
    root = Path(data_dir) if data_dir is not None else Path(get_data_root())
    stem = name[:-4] if name.endswith(".npy") else name
    suffix = f"_{side}"
    if not stem.endswith(suffix):
        stem = f"{stem}{suffix}"
    return root / f"{stem}.npy"


def make_right_output_path(name: str, data_dir: Path | str | None = None) -> Path:
    """Return ``data/<name>_right.npy`` for a right-hand HTS recording."""
    return make_output_path(name, hand_side="right", data_dir=data_dir)


def frame_to_geort_points(frame, converter: Callable | None = None) -> np.ndarray:
    """Convert one HTS frame into GeoRT's right-handed ``[21, 3]`` float32 layout."""
    if converter is None:
        from hand_tracking_sdk.convert import convert_hand_frame_unity_left_to_right

        converter = convert_hand_frame_unity_left_to_right

    converted = converter(frame)
    points = np.asarray(converted.landmarks.points, dtype=np.float32)
    if points.shape != (EXPECTED_HTS_LANDMARKS, 3):
        raise ValueError(
            f"Expected {EXPECTED_HTS_LANDMARKS} HTS landmarks with shape "
            f"({EXPECTED_HTS_LANDMARKS}, 3), got {points.shape}"
        )
    return points


def save_human_data(
    frames: Iterable[np.ndarray],
    name: str,
    hand_side: str,
    data_dir: Path | str | None = None,
) -> Path:
    """Save collected HTS frames as a GeoRT-compatible ``.npy`` file."""
    frame_array = np.asarray(list(frames), dtype=np.float32)
    if frame_array.ndim != 3 or frame_array.shape[1:] != (EXPECTED_HTS_LANDMARKS, 3):
        raise ValueError(
            f"Expected collected frames with shape [T, {EXPECTED_HTS_LANDMARKS}, 3], "
            f"got {frame_array.shape}"
        )

    output_path = make_output_path(name, hand_side=hand_side, data_dir=data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, frame_array)
    return output_path


def save_right_human_data(
    frames: Iterable[np.ndarray],
    name: str,
    data_dir: Path | str | None = None,
) -> Path:
    """Save collected right-hand frames as a GeoRT-compatible ``.npy`` file."""
    return save_human_data(frames, name=name, hand_side="right", data_dir=data_dir)


def iter_hts_points(
    *,
    hand_side: str,
    transport: str,
    host: str,
    port: int,
    timeout_s: float,
):
    """Yield GeoRT-ready points from selected-hand HTS frame events."""
    from hand_tracking_sdk import (
        ErrorPolicy,
        HTSClient,
        HTSClientConfig,
        HandFilter,
        HandFrame,
        HandSide,
        StreamOutput,
        TransportMode,
    )

    side = normalize_hand_side(hand_side)
    sdk_side_name = side.upper()
    hand_filter = getattr(HandFilter, sdk_side_name)
    expected_side = getattr(HandSide, sdk_side_name)
    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode(transport),
            host=host,
            port=port,
            timeout_s=timeout_s,
            output=StreamOutput.FRAMES,
            hand_filter=hand_filter,
            error_policy=ErrorPolicy.TOLERANT,
        )
    )

    for event in client.iter_events():
        if not isinstance(event, HandFrame):
            continue
        if event.side != expected_side:
            continue
        yield frame_to_geort_points(event)


def iter_right_hts_points(
    *,
    transport: str,
    host: str,
    port: int,
    timeout_s: float,
):
    """Yield GeoRT-ready points from right-hand HTS frame events."""
    yield from iter_hts_points(
        hand_side="right",
        transport=transport,
        host=host,
        port=port,
        timeout_s=timeout_s,
    )


def collect_human_data(
    *,
    name: str,
    hand_side: str,
    transport: str = "udp",
    host: str = "0.0.0.0",
    port: int = 9000,
    timeout_s: float = 1.0,
    max_frames: int | None = None,
) -> Path:
    """Collect selected-hand HTS frames until interrupted or ``max_frames`` is reached."""
    side = normalize_hand_side(hand_side)
    frames: list[np.ndarray] = []
    print(
        f"Listening for HTS {side}-hand frames on {transport}://{host}:{port}. "
        "Press Ctrl-C to stop and save."
    )

    try:
        for points in iter_hts_points(
            hand_side=side,
            transport=transport,
            host=host,
            port=port,
            timeout_s=timeout_s,
        ):
            frames.append(points)
            print(f"{side.title()}-hand frames collected: {len(frames)}", end="\r", flush=True)
            if max_frames is not None and len(frames) >= max_frames:
                break
    except KeyboardInterrupt:
        print("\nStopping collection.")

    if not frames:
        raise RuntimeError(f"No {side}-hand HTS frames were collected; nothing was saved.")

    output_path = save_human_data(frames, name=name, hand_side=side)
    print(f"\nSaved {len(frames)} frames to {output_path}")
    return output_path


def collect_right_human_data(
    *,
    name: str,
    transport: str = "udp",
    host: str = "0.0.0.0",
    port: int = 9000,
    timeout_s: float = 1.0,
    max_frames: int | None = None,
) -> Path:
    """Collect HTS right-hand frames until interrupted or ``max_frames`` is reached."""
    return collect_human_data(
        name=name,
        hand_side="right",
        transport=transport,
        host=host,
        port=port,
        timeout_s=timeout_s,
        max_frames=max_frames,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="hts", help="Dataset name; saved as data/<name>_<hand-side>.npy.")
    parser.add_argument(
        "--hand-side",
        choices=("left", "right"),
        default="right",
        help="HTS hand stream to collect. Saved frames are converted into GeoRT's right-handed coordinates.",
    )
    parser.add_argument(
        "--transport",
        choices=("udp", "tcp_server", "tcp_client"),
        default="udp",
        help="HTS transport mode. Defaults to UDP broadcast listening.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind/connect host for the selected transport.")
    parser.add_argument("--port", type=int, default=9000, help="Bind/connect port for HTS streaming.")
    parser.add_argument("--timeout-s", type=float, default=1.0, help="Socket receive timeout in seconds.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for scripted captures.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    collect_human_data(
        name=args.name,
        hand_side=args.hand_side,
        transport=args.transport,
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
