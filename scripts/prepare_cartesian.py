"""Convert the native polar tensor once into a regular Cartesian tensor."""

import json
import os
from pathlib import Path

import numpy as np

DATASET = Path("data/datasets/ktlx-reflectivity-120km-2016-2025-v1")
POLAR_TENSOR = "polar-180x64.npy"
CARTESIAN_TENSOR = "cartesian-z-64x64.npy"
GRID_SIZE = 64
RADIUS_KM = 120.0
MISSING_LINEAR_Z = 0.0


def polar_lookup(rays: int, gates: int, grid_size: int, radius_km: float):
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
    gate = np.rint(radius * gates / radius_km - 0.5).astype(np.int64)
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
    examples, frames, rays, gates = source.shape
    axis, ray, gate, inside = polar_lookup(rays, gates, GRID_SIZE, RADIUS_KM)

    temporary_path = destination_path.with_name(destination_path.name + ".tmp")
    destination = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(examples, frames, GRID_SIZE, GRID_SIZE),
    )
    for example in range(examples):
        dbz = np.asarray(source[example][:, ray, gate], dtype=np.float32)
        valid = np.isfinite(dbz) & inside[None, ...]
        cartesian_z = np.zeros_like(dbz, dtype=np.float32)
        cartesian_z[valid] = np.power(10.0, dbz[valid] / 10.0)
        destination[example] = cartesian_z
        if (example + 1) % 50 == 0 or example + 1 == examples:
            print(f"Cartesianized {example + 1:,}/{examples:,} examples", flush=True)
    destination.flush()
    del destination
    os.replace(temporary_path, destination_path)

    source_metadata = json.loads(
        source_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    metadata = {
        "format": "Cartesian radar grid",
        "source_tensor": POLAR_TENSOR,
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
        "failed_examples": source_metadata.get("failed_examples", {}),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Cartesian tensor: {destination_path.resolve()}")


if __name__ == "__main__":
    main()
