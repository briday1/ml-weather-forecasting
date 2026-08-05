"""Convert the native polar tensor once into a regular Cartesian tensor."""

import json
import os
from pathlib import Path

import numpy as np

DATASET = Path("data/datasets/ktlx-reflectivity-120km-2016-2025-v1")
POLAR_TENSOR = os.environ.get("ML_WEATHER_POLAR_TENSOR", "polar-180x64.npy")
GRID_SIZE = int(os.environ.get("ML_WEATHER_GRID_SIZE", "64"))
RADIUS_KM = float(os.environ.get("ML_WEATHER_RADIUS_KM", "120"))
CARTESIAN_TENSOR = os.environ.get(
    "ML_WEATHER_TENSOR",
    f"cartesian-z-{RADIUS_KM:g}km-{GRID_SIZE}x{GRID_SIZE}.npy",
)
MISSING_LINEAR_Z = 0.0


def polar_lookup(
    rays: int,
    gates: int,
    grid_size: int,
    radius_km: float,
    source_radius_km: float | None = None,
):
    """Map Cartesian grid centers to their nearest native polar bin."""
    spacing = 2 * radius_km / grid_size
    axis = np.linspace(
        -radius_km + spacing / 2,
        radius_km - spacing / 2,
        grid_size,
        dtype=np.float32,
    )
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    theta = np.mod(np.arctan2(x, y), 2 * np.pi)
    ray = np.rint(theta * rays / (2 * np.pi)).astype(np.int64) % rays
    source_radius_km = radius_km if source_radius_km is None else source_radius_km
    gate = np.rint(radius * gates / source_radius_km - 0.5).astype(np.int64)
    gate = np.clip(gate, 0, gates - 1)
    return axis, ray, gate, radius <= radius_km


def main() -> None:
    tensor_directory = DATASET / "tensors"
    source_path = tensor_directory / POLAR_TENSOR
    destination_path = tensor_directory / CARTESIAN_TENSOR
    metadata_path = destination_path.with_suffix(".json")
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing {source_path}; run scripts/get_data.py first")

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    source_metadata = json.loads(
        source_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    examples, frames, rays, gates = source.shape
    source_radius_km = float(source_metadata.get("radius_km", 120.0))
    use_native_scans = RADIUS_KM > source_radius_km
    if use_native_scans:
        from build_dataset import _sample_frame

        manifest = json.loads((DATASET / "dataset.json").read_text(encoding="utf-8"))
        lake_root = (DATASET / manifest["lake_root"]).resolve()

        def polar_frames(example: int) -> np.ndarray:
            return np.stack(
                [
                    _sample_frame(
                        lake_root / frame["lake_path"],
                        radius_km=RADIUS_KM,
                        azimuth_rays=rays,
                        range_gates=gates,
                    )
                    for frame in manifest["examples"][example]["frames"]
                ]
            )

        lookup_radius_km = RADIUS_KM
        print(
            f"Rebuilding {RADIUS_KM:g} km coverage from cached native scans",
            flush=True,
        )
    else:

        def polar_frames(example: int) -> np.ndarray:
            return source[example]

        lookup_radius_km = source_radius_km
    axis, ray, gate, inside = polar_lookup(
        rays, gates, GRID_SIZE, RADIUS_KM, lookup_radius_km
    )

    temporary_path = destination_path.with_name(destination_path.name + ".tmp")
    destination = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(examples, frames, GRID_SIZE, GRID_SIZE),
    )
    failures = dict(source_metadata.get("failed_examples", {}))
    for example in range(examples):
        identifier = source_metadata["examples"][example]
        if identifier in failures:
            destination[example] = MISSING_LINEAR_Z
        else:
            try:
                dbz = np.asarray(polar_frames(example)[:, ray, gate], dtype=np.float32)
                valid = np.isfinite(dbz) & inside[None, ...]
                cartesian_z = np.zeros_like(dbz, dtype=np.float32)
                cartesian_z[valid] = np.power(10.0, dbz[valid] / 10.0)
                destination[example] = cartesian_z
            except Exception as error:  # noqa: BLE001 - isolate corrupt examples.
                destination[example] = MISSING_LINEAR_Z
                failures[identifier] = str(error) or type(error).__name__
                print(
                    f"Skipped {identifier}: {failures[identifier]}",
                    flush=True,
                )
        if (example + 1) % 50 == 0 or example + 1 == examples:
            print(f"Cartesianized {example + 1:,}/{examples:,} examples", flush=True)
    destination.flush()
    del destination
    os.replace(temporary_path, destination_path)

    metadata = {
        "format": "Cartesian radar grid",
        "source_tensor": "cached native N0B scans"
        if use_native_scans
        else POLAR_TENSOR,
        "shape": [examples, frames, GRID_SIZE, GRID_SIZE],
        "dtype": "float32",
        "units": "linear reflectivity Z (mm^6 m^-3)",
        "radius_km": RADIUS_KM,
        "x_km": axis.tolist(),
        "y_km": axis.tolist(),
        "resampling": "nearest native polar bin",
        "missing_linear_z": MISSING_LINEAR_Z,
        "outside_radius": MISSING_LINEAR_Z,
        "contains_nan": False,
        "examples": source_metadata["examples"],
        "failed_examples": failures,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Cartesian tensor: {destination_path.resolve()}")


if __name__ == "__main__":
    main()
